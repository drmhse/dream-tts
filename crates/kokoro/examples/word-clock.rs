//! Synthesise a sentence and print when kokoro says each word.

use tts_core::{Engine, EngineConfig, Gaps, Sampling, SynthesisRequest};

fn main() -> anyhow::Result<()> {
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".into());
    let root = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "references/kokoro/weights".into());

    let engine = kokoro::engine::KokoroEngine::load(&EngineConfig::new(&root))?;
    let out = engine.synthesize(&SynthesisRequest {
        text: text.clone(),
        voice: None,
        sampling: Sampling::default(),
        max_chars: 400,
        max_new_tokens: usize::MAX,
        gaps: Gaps::default(),
        progress: None,
        interrupt: None,
    })?;

    let seconds = out.audio.samples.len() as f64 / out.audio.sample_rate as f64;
    println!(
        "{:.3}s of audio, {} samples\n",
        seconds,
        out.audio.samples.len()
    );
    // Where the sound actually is, so the clock can be checked against the audio rather than
    // against itself.
    let window = out.audio.sample_rate as usize / 20; // 50 ms
    let mut loud: Vec<(f64, f64)> = Vec::new();
    for (i, chunk) in out.audio.samples.chunks(window).enumerate() {
        let rms = (chunk.iter().map(|s| (s * s) as f64).sum::<f64>() / chunk.len() as f64).sqrt();
        loud.push((i as f64 * 0.05, rms));
    }
    let peak = loud.iter().map(|(_, r)| *r).fold(0.0, f64::max);
    let speech: Vec<f64> = loud
        .iter()
        .filter(|(_, r)| *r > peak * 0.02)
        .map(|(t, _)| *t)
        .collect();
    if let (Some(first), Some(last)) = (speech.first(), speech.last()) {
        println!("sound runs {first:.3}s to {last:.3}s (peak RMS {peak:.4})\n");
    }

    match &out.words {
        Some(words) => {
            for w in words {
                println!("  {:>7.3} → {:>7.3}  {}", w.start, w.end, w.text);
            }
            let last = words.last().map(|w| w.end).unwrap_or(0.0);
            println!("\nlast word ends at {last:.3}s, audio is {seconds:.3}s");
            let covered: f64 = words.iter().map(|w| w.end - w.start).sum();
            println!("words cover {:.1}% of the audio", 100.0 * covered / seconds);
        }
        None => println!("no word clock"),
    }
    Ok(())
}
