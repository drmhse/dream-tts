//! What the guesser makes of each word on stdin: `word<TAB>ARPAbet<TAB>misaki`.
use std::io::BufRead;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "references/kokoro/weights/frontend".into());
    let guesser = tts_phoneme::guess::Guesser::load(dir.as_ref())?
        .ok_or_else(|| anyhow::anyhow!("no g2p.safetensors in {dir}"))?;
    for word in std::io::stdin().lock().lines() {
        let word = word?;
        let arpabet = guesser.arpabet(&word.to_ascii_lowercase()).join(" ");
        println!(
            "{word}\t{arpabet}\t{}",
            guesser.phonemes(&word).unwrap_or_default()
        );
    }
    Ok(())
}
