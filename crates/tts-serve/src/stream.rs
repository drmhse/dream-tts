//! Incremental synthesis: audio out while the rest is still being made.
//!
//! # Why this is not just the buffered path with chunked encoding
//!
//! `Engine::synthesize` returns one finished waveform, so a route built on it cannot emit
//! anything until the last segment is done — for a paragraph that is tens of seconds, and
//! for a chapter it is minutes. What a realtime caller needs is not more throughput but a
//! short **time to first audio**, and that is a different quantity.
//!
//! So this segments the text itself and synthesizes one segment at a time, emitting each as
//! it lands. First audio arrives after one segment rather than after all of them.
//!
//! **It is slower overall, on purpose.** `qwen3tts` batches across segments and that is
//! worth 2x on book-length text; one segment at a time gives that up. The trade is right
//! here and wrong for a book, which is why the job runner does not use this path — a
//! listener waiting on a live response cares about the first second, and a book cares about
//! the last hour.
//!
//! # The gaps
//!
//! Segmenting outside the engine means reproducing what it does between segments: 90 ms
//! within a paragraph, 320 ms between paragraphs. Without that, streamed audio would run its
//! sentences together while the buffered route did not — the same text giving two different
//! renders depending on which URL was called.

use crate::{bad, require_key, ApiError, App, TtsRequest};
use axum::body::Body;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use std::sync::Arc;
use tokio::sync::mpsc;
use tts_core::{Gaps, Sampling, SynthesisRequest};

/// Chunks queued ahead of a slow reader before synthesis waits for it.
///
/// Small on purpose: a client that stops reading should stop the work rather than let the
/// server run a chapter ahead into memory. Two is enough to keep the GPU busy across the
/// hand-off without buffering an appreciable amount of audio.
const CHUNK_QUEUE: usize = 2;

pub async fn post_tts_stream(
    app: Arc<App>,
    headers: HeaderMap,
    req: TtsRequest,
) -> Result<Response, ApiError> {
    require_key(&app, &headers)?;

    let voice = crate::request_voice(&app, &req.voice)?;
    let sample_rate = app.sample_rate;

    // Paragraphs of segments, exactly as the engine would split it, so the streamed render
    // matches the buffered one.
    let paragraphs = tts_core::text::segment(&req.text, app.segment_chars);
    let segments: Vec<(usize, String)> = paragraphs
        .iter()
        .enumerate()
        .flat_map(|(p, sentences)| sentences.iter().map(move |s| (p, s.clone())))
        .collect();
    if segments.is_empty() {
        return Err(bad(StatusCode::BAD_REQUEST, "no text to speak"));
    }

    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, String>>(CHUNK_QUEUE);
    let seed = req.seed;
    let worker = Arc::clone(&app);

    tokio::spawn(async move {
        // Taken once for the whole stream, not per segment: releasing between segments
        // would let another request interleave and stall a live listener mid-sentence.
        let Ok(_permit) = worker.gpu.acquire().await else {
            return;
        };
        let gaps = Gaps::default();
        let mut previous_paragraph: Option<usize> = None;

        for (paragraph, text) in segments {
            let gap = match previous_paragraph {
                None => Vec::new(),
                Some(p) if p != paragraph => {
                    tts_core::wav::silence(sample_rate as usize, gaps.paragraph_ms)
                }
                Some(_) => tts_core::wav::silence(sample_rate as usize, gaps.segment_ms),
            };
            previous_paragraph = Some(paragraph);

            let engine = Arc::clone(&worker);
            let voice = voice.clone();
            let rendered = tokio::task::spawn_blocking(move || {
                let mut request = crate::with_optional_voice(SynthesisRequest::new(text), voice);
                // One segment per call, so the engine must not split it again.
                request.max_chars = usize::MAX;
                if let Some(s) = seed {
                    request.sampling = Sampling {
                        seed: s,
                        ..request.sampling
                    };
                }
                engine.engine.validate(&request)?;
                engine.engine.synthesize(&request)
            })
            .await;

            let chunk = match rendered {
                Ok(Ok(synth)) => {
                    let mut samples = gap;
                    samples.extend_from_slice(&synth.audio.samples);
                    Ok(tts_core::wav::pcm_s16le(&samples))
                }
                Ok(Err(e)) => Err(format!("{e:#}")),
                Err(e) => Err(format!("synthesis task failed: {e}")),
            };
            let failed = chunk.is_err();
            // A closed receiver means the client hung up; stop rather than finish the book
            // for nobody.
            if tx.send(chunk).await.is_err() || failed {
                return;
            }
        }
    });

    let body = Body::from_stream(
        tokio_stream::wrappers::ReceiverStream::new(rx).map(|chunk| match chunk {
            Ok(bytes) => Ok::<_, std::io::Error>(bytes),
            // The response has already begun, so the status is long gone. Ending the body
            // early is the only signal left, and a truncated stream is what a client sees.
            Err(message) => {
                eprintln!("stream: {message}");
                Err(std::io::Error::other(message))
            }
        }),
    );

    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header("x-sample-rate", sample_rate.to_string())
        .header("x-audio-format", "pcm_s16le_mono")
        // Was `buffered`, and a client that trusted it would have waited for the whole
        // render before playing anything.
        .header("x-streaming", "incremental")
        .body(body)
        .map_err(|e| bad(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

use futures::StreamExt;
