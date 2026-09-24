//! `POST /v1/batch`: several texts in one engine run, answered as one WAV per text.
//!
//! An engine that batches across segments (qwen3tts) renders eight chunks for the wall time of
//! about two, but a caller that needs each chunk's own audio could only send them one at a time.
//! The join already knows where every segment landed, so the split is exact.

use crate::{bad, render, require_key, validate, ApiError, App, TtsRequest};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tts_core::{Audio, SegmentTime, WordTime};

pub const MAX_TEXTS: usize = 16;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchRequest {
    texts: Vec<String>,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Serialize)]
pub struct Part {
    /// A complete WAV, base64.
    audio: String,
    seconds: f64,
    words: Option<Vec<(String, f64, f64)>>,
    segments: Vec<(String, f64, f64)>,
}

#[derive(Serialize)]
pub struct BatchResponse {
    /// Lets a client refuse audio from an engine other than the one it thinks it is caching.
    engine: String,
    parts: Vec<Part>,
    audio_seconds: f64,
    wall_seconds: f64,
}

pub async fn post_batch(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(req): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, ApiError> {
    require_key(&app, &headers)?;
    if req.texts.is_empty() || req.texts.len() > MAX_TEXTS {
        return Err(bad(
            StatusCode::BAD_REQUEST,
            format!("between 1 and {MAX_TEXTS} texts, got {}", req.texts.len()),
        ));
    }
    // One paragraph per text: the segmenter splits paragraphs on lines.
    let texts: Vec<String> = req
        .texts
        .iter()
        .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let single = |text: String| TtsRequest {
        text,
        mode: "zero_shot".into(),
        instruct_text: None,
        speed: 1.0,
        voice: req.voice.clone(),
        seed: req.seed,
    };
    for t in &texts {
        validate(&app, &single(t.clone()))?;
    }
    let r = render(&app, single(texts.join("\n"))).await?;
    let segments = r.segments.as_deref().unwrap_or_default();
    let parts = split(&r.audio, segments, r.words.as_deref(), texts.len()).ok_or_else(|| {
        bad(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "{} texts came back as a different number of paragraphs; send them singly",
                texts.len()
            ),
        )
    })?;
    Ok(Json(BatchResponse {
        engine: app.engine_id.clone(),
        parts,
        audio_seconds: r.seconds,
        wall_seconds: r.wall,
    }))
}

/// Cut the joined audio back into one part per paragraph. A paragraph starts wherever the
/// silence before a segment is the paragraph gap rather than the segment gap.
fn split(
    audio: &Audio,
    segments: &[SegmentTime],
    words: Option<&[WordTime]>,
    expected: usize,
) -> Option<Vec<Part>> {
    let gaps = tts_core::Gaps::default();
    let threshold = (gaps.segment_ms + gaps.paragraph_ms) as f64 / 2000.0;
    let mut groups: Vec<Vec<&SegmentTime>> = Vec::new();
    for (i, s) in segments.iter().enumerate() {
        if i == 0 || s.start - segments[i - 1].end > threshold {
            groups.push(Vec::new());
        }
        groups.last_mut()?.push(s);
    }
    if groups.len() != expected {
        return None;
    }
    let rate = audio.sample_rate as f64;
    let at = |t: f64| ((t * rate).round() as usize).min(audio.samples.len());
    Some(
        groups
            .iter()
            .map(|g| {
                let (start, end) = (g[0].start, g[g.len() - 1].end);
                let samples = audio.samples[at(start)..at(end)].to_vec();
                let shifted = |text: &str, s: f64, e: f64| (text.to_string(), s - start, e - start);
                Part {
                    seconds: samples.len() as f64 / rate,
                    audio: base64(&tts_core::wav::to_bytes(&Audio {
                        samples: samples.clone(),
                        sample_rate: audio.sample_rate,
                    })),
                    words: words.map(|w| {
                        w.iter()
                            .filter(|w| w.start >= start - 1e-6 && w.start < end)
                            .map(|w| shifted(&w.text, w.start, w.end.min(end)))
                            .collect()
                    }),
                    segments: g.iter().map(|s| shifted(&s.text, s.start, s.end)).collect(),
                }
            })
            .collect(),
    )
}

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            out.push(if i <= c.len() {
                T[(n >> (18 - 6 * i) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str, start: f64, end: f64) -> SegmentTime {
        SegmentTime {
            text: text.into(),
            start,
            end,
        }
    }

    #[test]
    fn paragraphs_are_found_by_their_gap() {
        let audio = Audio {
            samples: vec![0.0; 1000],
            sample_rate: 100,
        };
        // Segment gap 90 ms inside a paragraph, 320 ms between them.
        let segments = [seg("a", 0.0, 1.0), seg("b", 1.09, 2.0), seg("c", 2.32, 3.0)];
        let words = [
            WordTime {
                text: "a".into(),
                start: 0.1,
                end: 0.5,
            },
            WordTime {
                text: "c".into(),
                start: 2.4,
                end: 2.9,
            },
        ];
        let parts = split(&audio, &segments, Some(&words), 2).expect("two paragraphs");
        assert_eq!(parts[0].segments.len(), 2);
        assert!((parts[0].seconds - 2.0).abs() < 1e-9);
        assert!((parts[1].seconds - 0.68).abs() < 1e-9);
        let w = parts[1].words.as_ref().unwrap();
        assert_eq!(w.len(), 1);
        assert!((w[0].1 - 0.08).abs() < 1e-9, "{:?}", w[0]);
        assert!(split(&audio, &segments, None, 3).is_none());
    }

    #[test]
    fn base64_matches_the_standard_alphabet() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }
}
