//! The opt-in approximation for words misaki drops. Skipped, and said so, without the exported
//! frontend.
use std::path::Path;
use tts_phoneme::g2p::G2P;

fn g2p() -> Option<G2P> {
    load(false)
}

fn load(british: bool) -> Option<G2P> {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../references/kokoro/weights/frontend");
    if !dir.is_dir() {
        eprintln!("skipped: no frontend at {}", dir.display());
        return None;
    }
    Some(G2P::load(&dir, british).expect("frontend"))
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

#[test]
fn words_in_other_letters_are_said_as_english_spells_them() {
    let Some(g2p) = g2p() else { return };
    let text = "Sun Tzŭ, Ts’ao Kung and Yao-ch’en; her manœuvre, Cæsar and Dinah’ll.";
    // misaki drops every one of them, and the strict path still does.
    let (_, unknown) = g2p.phonemize_report(text);
    assert!(unknown.len() >= 5, "{unknown:?}");

    let (phonemes, unknown) = g2p.with_fallback(true).phonemize_report(text);
    assert!(unknown.is_empty(), "{unknown:?}");
    assert!(
        phonemes.contains("tsˈu"),
        "Tzŭ, its tz the ts it stands for: {phonemes}"
    );
    assert!(
        phonemes.contains("tsˈW"),
        "Ts’ao whole, not Ts’a + o: {phonemes}"
    );
    assert!(
        phonemes.contains("ʧˈɛn"),
        "ch’en whole between hyphens: {phonemes}"
    );
    assert!(
        phonemes.contains("mənˈuvəɹ"),
        "the lexicon's manoeuvre: {phonemes}"
    );
    assert!(
        phonemes.contains("sˈizəɹ"),
        "the lexicon's Caesar: {phonemes}"
    );
    assert!(
        phonemes.contains("dˈInəəl"),
        "Dinah with the lexicon's ’ll: {phonemes}"
    );
}

#[test]
fn names_are_guessed_before_they_are_split_into_words() {
    let Some(g2p) = g2p() else { return };
    let (phonemes, _) = g2p
        .with_fallback(true)
        .phonemize_report("Dinah, Wickham and O’Brien; the filename.");
    assert!(
        phonemes.contains("dˈInə"),
        "Dinah, not din + ah: {phonemes}"
    );
    assert!(
        !phonemes.contains("wˈɪkhˌæm"),
        "Wickham, not wick + ham: {phonemes}"
    );
    assert!(
        phonemes.contains("ˌObɹˈin"),
        "O’Brien, O + Brien: {phonemes}"
    );
    assert!(
        phonemes.contains("fˈIlnˌAm"),
        "a lowercase compound still splits: {phonemes}"
    );
}

#[test]
fn a_british_voice_says_what_its_lexicon_lacks() {
    let Some(g2p) = load(true) else { return };
    let (_, unknown) = g2p
        .with_fallback(true)
        .phonemize_report("Mr. Darcy at Pemberley; Mowgli, Kaa and the slithy toves.");
    assert!(unknown.is_empty(), "{unknown:?}");
}
