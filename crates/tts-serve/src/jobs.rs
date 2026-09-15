//! Narration runs: submitting a book, watching it, and stopping it.
//!
//! # Why this lives in the service
//!
//! A thousand-page book is hours of synthesis, and the expensive thing — the resident engine
//! — is already here. A separate job daemon would mean two processes wanting one GPU, so the
//! queue goes where the engine is and the same semaphore that serialises one-off requests
//! serialises chapters.
//!
//! The *records* deliberately do not live here: [`tts_jobs::Store`] writes them to the data
//! directory so a process that is not this one — a status bar, a shell script, the CLI
//! before it has started a server — can answer "is something already narrating?".
//!
//! # What a caller sees
//!
//! Three shapes, because three different things watch a run. `GET /v1/jobs` is a list for a
//! decision ("is the machine busy?"). `GET /v1/jobs/{id}` is a snapshot for a check.
//! `GET /v1/jobs/{id}/events` is a stream for a display, and it opens with the current state
//! so a watcher that attaches late is not blank until the next segment finishes.

use crate::{bad, require_key, ApiError, App};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, Notify};
use tts_core::{ProgressEvent, Sampling, SynthesisRequest, Voice};
use tts_jobs::{
    ChapterState, Job, Progress, Request as JobRequest, StageProgress, State as JobState, Store,
};

/// How many events a slow watcher may fall behind before it is dropped.
///
/// A browser tab that stops reading must not stall the run, so the channel is bounded and a
/// lagging receiver loses events rather than applying back-pressure. Progress is a snapshot,
/// not a ledger — a watcher that misses some is still correct at the next one.
const EVENT_BUFFER: usize = 256;

/// What the runner publishes.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Update {
    /// Where the running chapter is, several times a second.
    Progress(Progress),
    /// A job changed state, or a chapter finished. Carries the whole job, because a watcher
    /// that reacts to this wants the new totals anyway.
    Job(Box<Job>),
}

/// The runner's half of the world: the store, the wake-up, and the broadcast.
pub struct Jobs {
    pub store: Store,
    /// Set to stop the chapter in flight. An atomic rather than a channel because the
    /// engine's interrupt hook is a plain `Fn() -> bool` asked between segments.
    pub interrupt: Arc<std::sync::atomic::AtomicBool>,
    /// Woken when a job is submitted or a control request arrives, so the loop reacts at
    /// once rather than at its next poll.
    wake: Notify,
    events: broadcast::Sender<Update>,
}

impl Jobs {
    pub fn open(data_dir: &std::path::Path) -> anyhow::Result<Self> {
        let store = Store::open(data_dir)?;
        // A previous server killed mid-run leaves jobs claiming to run. Turn them into
        // resumable paused jobs at startup, before anyone can read a lie.
        let reaped = store.reap_stale()?;
        if !reaped.is_empty() {
            eprintln!(
                "note: {} job(s) left running by an earlier process are now paused and \
                 resumable: {}",
                reaped.len(),
                reaped.join(", ")
            );
        }
        let (events, _) = broadcast::channel(EVENT_BUFFER);
        Ok(Self {
            store,
            interrupt: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            wake: Notify::new(),
            events,
        })
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Update> {
        self.events.subscribe()
    }

    fn publish(&self, update: Update) {
        // An error means nobody is listening, which is the normal case.
        let _ = self.events.send(update);
    }

    pub fn wake(&self) {
        self.wake.notify_one();
    }
}

// ---------------------------------------------------------------- the runner

/// One chapter at a time, for ever.
///
/// Polls as well as waiting on the notify: a job can also become runnable because a file
/// appeared or another process edited the store, and a loop that only reacts to its own
/// notifications would miss that.
pub async fn run(app: Arc<App>) {
    loop {
        match next_runnable(&app) {
            Some(job) => {
                if let Err(e) = run_one_chapter(&app, job).await {
                    eprintln!("job runner: {e:#}");
                    // Back off rather than spin on a persistent failure.
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
            }
            None => {
                tokio::select! {
                    _ = app.jobs.wake.notified() => {}
                    _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {}
                }
            }
        }
    }
}

/// The oldest job with work to do. Oldest, so a queue is first-in-first-out and a long book
/// cannot be starved by newer submissions.
fn next_runnable(app: &Arc<App>) -> Option<Job> {
    let mut jobs = app.jobs.store.list().ok()?;
    jobs.sort_by_key(|j| j.created);
    jobs.into_iter()
        .find(|j| j.state.is_active() && j.next_chapter().is_some())
}

async fn run_one_chapter(app: &Arc<App>, mut job: Job) -> anyhow::Result<()> {
    // Re-read under no lock but check the request each time round: a pause that arrived
    // while the previous chapter ran takes effect here, which is the boundary that keeps a
    // half-written chapter from ever existing.
    match job.request {
        JobRequest::Pause => {
            job.state = JobState::Paused;
            job.request = JobRequest::None;
            job.pid = None;
            return finish(app, job);
        }
        JobRequest::Cancel => {
            job.state = JobState::Cancelled;
            job.request = JobRequest::None;
            job.pid = None;
            return finish(app, job);
        }
        JobRequest::None => {}
    }

    let Some(index) = job.next_chapter() else {
        job.state = JobState::Done;
        job.pid = None;
        return finish(app, job);
    };

    job.state = JobState::Running;
    job.pid = Some(std::process::id());
    job.chapters[index].state = ChapterState::Running;
    job.updated = tts_jobs::now();
    app.jobs.store.put(&job)?;
    app.jobs.publish(Update::Job(Box::new(job.clone())));

    let chapter = job.chapters[index].clone();
    let text = std::fs::read_to_string(&chapter.text_path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", chapter.text_path))?;

    // A job records the voice it was submitted with, which for an engine that cannot clone
    // is the built-in name rather than a path — there is nothing to load.
    let voice = app
        .voice
        .is_some()
        .then(|| {
            Voice::load(&job.voice).map_err(|e| anyhow::anyhow!("loading voice {}: {e}", job.voice))
        })
        .transpose()?;

    // The same permit one-off requests take: chapters queue behind them and each other.
    let _permit = app.gpu.acquire().await?;

    let started = Instant::now();
    let progress = ChapterProgress::new(app, &job, index, started);
    let engine = Arc::clone(app);
    let segment_chars = app.segment_chars;
    let seed = job.seed;
    // Read by the engine between segments. A pause that must wait out a 1900-word chapter
    // is a pause that takes eight minutes, so `--now` sets this: the chapter in flight is
    // discarded and redone on resume. The cost, stated, rather than a slow control.
    let stop = Arc::clone(&app.jobs.interrupt);
    let outcome = tokio::task::spawn_blocking(move || {
        let mut request = crate::with_optional_voice(SynthesisRequest::new(text), voice)
            .with_progress(Arc::new(move |event| progress.on(event)))
            .with_interrupt(Arc::new(move || {
                stop.load(std::sync::atomic::Ordering::Relaxed)
            }));
        request.max_chars = segment_chars;
        if let Some(s) = seed {
            request.sampling = Sampling {
                seed: s,
                ..request.sampling
            };
        }
        engine.engine.validate(&request)?;
        engine.engine.synthesize(&request)
    })
    .await?;

    let wall = started.elapsed().as_secs_f64();
    let was_interrupted = app
        .jobs
        .interrupt
        .swap(false, std::sync::atomic::Ordering::Relaxed);
    // Re-read: a control request may have arrived while the chapter ran, and it is on the
    // stored record rather than on the copy this task has been holding.
    let stored = app.jobs.store.get(&job.id)?.unwrap_or_else(|| job.clone());
    job.request = stored.request;

    match outcome {
        Ok(synth) => {
            let seconds = synth.audio.seconds();
            tts_core::wav::write_mono(
                std::path::Path::new(&chapter.wav_path),
                &synth.audio.samples,
                synth.audio.sample_rate,
            )
            .map_err(|e| anyhow::anyhow!("writing {}: {e}", chapter.wav_path))?;
            let c = &mut job.chapters[index];
            c.state = ChapterState::Done;
            c.audio_seconds = Some(seconds);
            c.wall_seconds = Some(wall);
            c.error = None;
        }
        // An interrupted chapter is not a failure: a caller stopped it. Left pending, so
        // resuming redoes it from the start — its partial waveform was never audio.
        Err(e) if was_interrupted || e.downcast_ref::<tts_core::Interrupted>().is_some() => {
            job.chapters[index].state = ChapterState::Pending;
            job.state = match job.request {
                JobRequest::Cancel => JobState::Cancelled,
                _ => JobState::Paused,
            };
            job.request = JobRequest::None;
            job.pid = None;
            return finish(app, job);
        }
        Err(e) => {
            let c = &mut job.chapters[index];
            c.state = ChapterState::Failed;
            c.error = Some(format!("{e:#}"));
            job.state = JobState::Failed;
            job.error = Some(format!("{} failed: {e:#}", chapter.name));
            job.pid = None;
            return finish(app, job);
        }
    }

    if job.next_chapter().is_none() {
        job.state = JobState::Done;
        job.pid = None;
    }
    finish(app, job)
}

fn finish(app: &Arc<App>, mut job: Job) -> anyhow::Result<()> {
    job.updated = tts_jobs::now();
    app.jobs.store.put(&job)?;
    app.jobs.publish(Update::Job(Box::new(job)));
    Ok(())
}

/// Turns the engine's per-segment callback into published progress.
///
/// Holds only what the snapshot needs, so the callback — which runs on the synthesis thread
/// between segments — neither locks nor touches the disk.
struct ChapterProgress {
    jobs: Arc<App>,
    job: String,
    chapter: String,
    chapter_index: usize,
    chapters_total: usize,
    words_done: usize,
    words_total: usize,
    eta: Option<f64>,
    started: Instant,
}

impl ChapterProgress {
    fn new(app: &Arc<App>, job: &Job, index: usize, started: Instant) -> Self {
        Self {
            jobs: Arc::clone(app),
            job: job.id.clone(),
            chapter: job.chapters[index].name.clone(),
            chapter_index: index,
            chapters_total: job.chapters.len(),
            words_done: job.words_done(),
            words_total: job.words_total(),
            eta: job.eta_seconds(),
            started,
        }
    }

    fn on(&self, event: ProgressEvent) {
        let stage = match event {
            // `Planned` says how many segments there are but not which stage; the first
            // `Advanced` names one. Publishing it keeps a watcher from sitting at zero
            // through a long first stage with nothing to show.
            ProgressEvent::Planned { segments } => Some(StageProgress {
                stage: "planned".into(),
                done: 0,
                total: segments,
            }),
            ProgressEvent::Advanced { stage, done, total } => Some(StageProgress {
                stage: stage.to_string(),
                done,
                total,
            }),
        };
        self.jobs.jobs.publish(Update::Progress(Progress {
            job: self.job.clone(),
            chapter_index: self.chapter_index,
            chapter: self.chapter.clone(),
            chapters_total: self.chapters_total,
            stage,
            elapsed: self.started.elapsed().as_secs_f64(),
            eta_seconds: self.eta,
            words_done: self.words_done,
            words_total: self.words_total,
        }));
    }
}

// ---------------------------------------------------------------- the routes

#[derive(Deserialize)]
pub struct Submit {
    /// What to call this run, for a human. The document's path, usually.
    pub source: String,
    /// Narration text files, in reading order.
    pub texts: Vec<String>,
    pub out_dir: String,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

pub async fn post_jobs(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(req): Json<Submit>,
) -> Result<Json<Job>, ApiError> {
    require_key(&app, &headers)?;

    let plan = tts_jobs::Plan {
        source: req.source,
        texts: req.texts.iter().map(std::path::PathBuf::from).collect(),
        out_dir: std::path::PathBuf::from(&req.out_dir),
        engine: app.engine_id.clone(),
        voice: req.voice.unwrap_or_else(|| app.voice_name.clone()),
        quant: req.quant,
        seed: req.seed,
    };
    let fresh = plan
        .into_job()
        .map_err(|e| bad(StatusCode::BAD_REQUEST, format!("{e:#}")))?;

    std::fs::create_dir_all(&req.out_dir).map_err(|e| {
        bad(
            StatusCode::BAD_REQUEST,
            format!("creating {}: {e}", req.out_dir),
        )
    })?;

    // Submitting the same book again *is* the resume path: the id is a hash of what will be
    // spoken, so this finds the earlier run and adopts its finished chapters.
    let job = match app.jobs.store.get(&fresh.id).map_err(internal)? {
        Some(stored) => {
            let mut job = tts_jobs::adopt(fresh, &stored);
            // A terminal job resubmitted is a request to continue it, except when it is
            // genuinely finished — then it stays done and the caller sees that nothing is
            // left to do.
            job.state = if job.next_chapter().is_none() {
                JobState::Done
            } else {
                JobState::Queued
            };
            job.request = JobRequest::None;
            job.error = None;
            job
        }
        None => fresh,
    };
    app.jobs.store.put(&job).map_err(internal)?;
    app.jobs.publish(Update::Job(Box::new(job.clone())));
    app.jobs.wake();
    Ok(Json(job))
}

pub async fn get_jobs(State(app): State<Arc<App>>) -> Result<Json<serde_json::Value>, ApiError> {
    let jobs = app.jobs.store.list().map_err(internal)?;
    let active = jobs
        .iter()
        .filter(|j| j.state.is_active() && !j.is_stale())
        .count();
    Ok(Json(json!({ "active": active, "jobs": jobs })))
}

pub async fn get_job(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
) -> Result<Json<Job>, ApiError> {
    Ok(Json(lookup(&app, &id)?))
}

/// `pause`, `resume` or `cancel`.
pub async fn post_job_control(
    State(app): State<Arc<App>>,
    Path((id, action)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_key(&app, &headers)?;
    let mut job = lookup(&app, &id)?;

    // What the caller is told, because "pause" on a running chapter does not take effect
    // now and a control that appears to do nothing is worse than one that refuses.
    let note = match action.as_str() {
        "pause" | "pause-now" => match job.state {
            JobState::Running => {
                job.request = JobRequest::Pause;
                if action == "pause-now" {
                    app.jobs
                        .interrupt
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    format!(
                        "stopping {} now; it will be narrated again from the start on resume",
                        job.chapters
                            .get(job.next_chapter().unwrap_or(0))
                            .map_or("the chapter in flight", |c| c.name.as_str())
                    )
                } else {
                    "pausing after the chapter in flight, so its work is kept".to_string()
                }
            }
            JobState::Queued => {
                job.state = JobState::Paused;
                "paused before starting".to_string()
            }
            other => {
                return Err(bad(
                    StatusCode::CONFLICT,
                    format!("cannot pause a {} job", other.name()),
                ))
            }
        },
        "resume" => match job.state {
            JobState::Paused | JobState::Failed | JobState::Cancelled => {
                if job.next_chapter().is_none() {
                    job.state = JobState::Done;
                    "nothing left to narrate".to_string()
                } else {
                    job.state = JobState::Queued;
                    job.request = JobRequest::None;
                    job.error = None;
                    format!(
                        "resuming at {}",
                        job.chapters[job.next_chapter().unwrap_or(0)].name
                    )
                }
            }
            JobState::Running => "already running".to_string(),
            other => {
                return Err(bad(
                    StatusCode::CONFLICT,
                    format!("cannot resume a {} job", other.name()),
                ))
            }
        },
        "cancel" | "cancel-now" => match job.state {
            JobState::Running => {
                job.request = JobRequest::Cancel;
                if action == "cancel-now" {
                    app.jobs
                        .interrupt
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    "cancelling now; the chapter in flight is discarded".to_string()
                } else {
                    "cancelling after the chapter in flight".to_string()
                }
            }
            s if !s.is_terminal() => {
                job.state = JobState::Cancelled;
                "cancelled".to_string()
            }
            other => {
                return Err(bad(
                    StatusCode::CONFLICT,
                    format!("cannot cancel a {} job", other.name()),
                ))
            }
        },
        other => {
            return Err(bad(
                StatusCode::NOT_FOUND,
                format!(
                    "no such action `{other}`; use pause, pause-now, resume, cancel or \
                     cancel-now"
                ),
            ))
        }
    };

    job.updated = tts_jobs::now();
    app.jobs.store.put(&job).map_err(internal)?;
    app.jobs.publish(Update::Job(Box::new(job.clone())));
    app.jobs.wake();
    Ok(Json(json!({ "job": job, "note": note })))
}

/// The progress stream.
///
/// Opens with the job as it stands, so a watcher attaching to a long chapter is not blank
/// until the next segment. Keep-alive comments stop an idle proxy from closing it.
pub async fn get_job_events(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let job = lookup(&app, &id)?;
    let receiver = app.jobs.subscribe();
    let opening = futures::stream::once(async move { Update::Job(Box::new(job)) });
    let live = tokio_stream::wrappers::BroadcastStream::new(receiver)
        .filter_map(|update| futures::future::ready(update.ok()));

    use futures::StreamExt;
    let stream = opening.chain(live).filter_map(move |update| {
        let keep = match &update {
            Update::Progress(p) => p.job == id,
            Update::Job(j) => j.id == id,
        };
        futures::future::ready(keep.then(|| {
            Event::default()
                .json_data(&update)
                .map_err(|_| unreachable!())
        }))
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn lookup(app: &Arc<App>, id: &str) -> Result<Job, ApiError> {
    app.jobs
        .store
        .get(id)
        .map_err(internal)?
        .ok_or_else(|| bad(StatusCode::NOT_FOUND, format!("no job {id}")))
}

fn internal(e: anyhow::Error) -> ApiError {
    bad(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
}
