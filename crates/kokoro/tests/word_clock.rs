//! The word clock, checked against the audio rather than against itself.
//!
//! A clock can be self-consistent and wrong: an earlier version used a constant for the
//! decoder's upsampling that was out by two, and every word was in the right order, in the
//! right proportion, and in the wrong place. Only the audio can say.

use tts_core::{Engine, EngineConfig, Gaps, Sampling, SynthesisRequest};

/// Where sound starts and stops, by short-window energy.
fn sound(samples: &[f32], rate: u32) -> Option<(f64, f64)> {
    let window = rate as usize / 20;
    let rms: Vec<f64> = samples
        .chunks(window)
        .map(|c| (c.iter().map(|s| (s * s) as f64).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    let peak = rms.iter().copied().fold(0.0, f64::max);
    let loud: Vec<usize> = (0..rms.len()).filter(|i| rms[*i] > peak * 0.02).collect();
    let step = window as f64 / rate as f64;
    Some((
        *loud.first()? as f64 * step,
        (*loud.last()? + 1) as f64 * step,
    ))
}

#[test]
fn the_clock_lands_where_the_sound_is() {
    // Relative to the crate, because that is where cargo runs a test from.
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../references/kokoro/weights");
    if !root.join("kokoro.safetensors").exists() {
        eprintln!("skipped: no kokoro checkpoint at {}", root.display());
        return;
    }
    let engine = kokoro::engine::KokoroEngine::load(&EngineConfig::new(&root)).expect("load");
    let out = engine
        .synthesize(&SynthesisRequest {
            text: "The quick brown fox jumps over the lazy dog.".into(),
            voice: None,
            sampling: Sampling::default(),
            max_chars: 400,
            max_new_tokens: usize::MAX,
            gaps: Gaps::default(),
            progress: None,
            interrupt: None,
        })
        .expect("synthesize");

    let words = out
        .words
        .expect("kokoro predicts durations, so it has a word clock");
    assert_eq!(
        words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."],
        "the clock does not name the words that were said"
    );

    let seconds = out.audio.samples.len() as f64 / out.audio.sample_rate as f64;
    let (from, to) = sound(&out.audio.samples, out.audio.sample_rate).expect("audio");

    for pair in words.windows(2) {
        assert!(
            pair[0].start < pair[0].end,
            "a word takes no time: {pair:?}"
        );
        assert!(
            pair[0].end <= pair[1].start + 1e-9,
            "the clock runs backwards: {pair:?}"
        );
    }

    let first = words.first().expect("a first word").start;
    let last = words.last().expect("a last word").end;
    assert!(
        last <= seconds + 1e-6,
        "the clock runs past the audio: {last} > {seconds}"
    );

    // One 50 ms window of slack at the front: a word starts at its first phoneme, and a stop
    // consonant is near-silent before it releases.
    assert!(
        (first - from).abs() < 0.1,
        "the first word starts at {first:.3}s and the sound at {from:.3}s"
    );
    // The last word may run into the trailing silence it ends with; it must not stop early.
    assert!(
        last >= to - 0.1,
        "the clock ends at {last:.3}s and the sound at {to:.3}s"
    );
}
