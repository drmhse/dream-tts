//! The opt-in approximation for words misaki drops. Skipped, and said so, without the exported
//! frontend.
use std::path::Path;
use tts_phoneme::g2p::G2P;

fn g2p() -> Option<G2P> {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../references/kokoro/weights/frontend");
    if !dir.is_dir() {
        eprintln!("skipped: no frontend at {}", dir.display());
        return None;
    }
    Some(G2P::load(&dir, false).expect("frontend"))
}

#[test]
fn initials_are_spelled_and_compounds_split_only_when_asked() {
    let Some(g2p) = g2p() else { return };
    let (_, unknown) = g2p.phonemize_report("A png on mysite.");
    assert!(unknown.contains(&"png".to_string()) && unknown.contains(&"mysite".to_string()));

    let g2p = g2p.with_fallback(true);
    let (phonemes, unknown) = g2p.phonemize_report("A png on mysite.");
    assert!(unknown.is_empty(), "{unknown:?}");
    assert!(phonemes.contains("pˈi ˈɛn ʤˈi"), "{phonemes}");
    assert!(phonemes.contains("sˈIt"), "{phonemes}");
}
