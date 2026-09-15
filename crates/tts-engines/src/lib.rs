//! The registry: the one place that knows which engines exist.
//!
//! Kept separate from `tts-core` so the trait has no dependency on its
//! implementations, and separate from the CLI so a service can reuse selection without
//! inheriting an argument parser. A caller does:
//!
//! ```no_run
//! use tts_core::{EngineConfig, SynthesisRequest, Voice};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Enumerate first: unavailable engines are listed too, with a reason.
//! for caps in tts_engines::catalogue() {
//!     println!("{:<10} {}", caps.id, if caps.available { "ready" } else { "unavailable" });
//! }
//!
//! let id = tts_engines::default_id();
//! let config = EngineConfig::new(tts_engines::default_root(id));
//! let engine = tts_engines::load(id, &config)?;
//!
//! // A caller that did not choose an engine is told what it got, and why that matters.
//! if let Some(caveat) = tts_engines::default_caveat() {
//!     eprintln!("note: {caveat}");
//! }
//!
//! // Voice assets load on the host; the engine pulls them onto its own device.
//! let voice = Voice::load(tts_engines::default_voice(id))?;
//! let request = SynthesisRequest::new("Hello from Rust.").with_voice(voice);
//!
//! // `validate` rejects a mismatched asset up front rather than at the first tensor.
//! engine.validate(&request)?;
//! let out = engine.synthesize(&request)?;
//!
//! tts_core::wav::write("hello.wav", &out.audio)?;
//! let secs = out.audio.seconds();
//! println!("{secs:.2} s at RTF {:.3}", out.stats.rtf(secs));
//! for (stage, s, rtf, share) in out.stats.breakdown(secs) {
//!     println!("  {stage:<8} {s:.1} s ({share:.0}%) RTF {rtf:.3}");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Unavailable engines are listed rather than hidden, so a client can discover an
//! identifier before it works and get an error that says why instead of silently
//! getting a different voice from a different model.

use anyhow::Result;
use tts_core::{Capabilities, Engine, EngineConfig};

/// Every engine known to this build, in preference order.
///
/// Order is preference, and [`default_id`] takes the first *available* entry — so an
/// unfinished engine may sit anywhere without becoming the default.
///
/// `qwen3tts` leads because it is the fastest by 2x on book-length text and the only
/// engine whose setup needs nothing but `curl`. It also speaks ten languages and the
/// list is closed, which used to be handled by keeping it *last* — a guard made of list
/// position, invisible to any caller. That fact now lives in
/// [`Capabilities::languages`] and is reported by [`default_caveat`], so a caller that
/// did not choose an engine is told what it got instead of being quietly given a
/// different one.
///
/// `kokoro` follows it rather than leads despite being the fastest here: it cannot clone,
/// so defaulting to it would answer a request for a cloned voice with a stranger's.
pub fn catalogue() -> Vec<Capabilities> {
    vec![
        qwen3tts::capabilities(),
        kokoro::engine::capabilities(),
        audio8::engine::capabilities(),
        cosyvoice::capabilities(),
    ]
}

/// Ids a caller may pass to [`load`].
pub fn ids() -> Vec<&'static str> {
    catalogue().into_iter().map(|c| c.id).collect()
}

/// The first engine that can actually synthesize — what a client gets when it does not
/// care which.
pub fn default_id() -> &'static str {
    catalogue()
        .into_iter()
        .find(|c| c.available)
        .map(|c| c.id)
        .unwrap_or(audio8::engine::ID)
}

pub fn load(id: &str, config: &EngineConfig) -> Result<Box<dyn Engine>> {
    match id {
        audio8::engine::ID => Ok(Box::new(audio8::engine::Audio8Engine::load(config)?)),
        cosyvoice::ID => Ok(Box::new(cosyvoice::CosyVoiceEngine::load(config)?)),
        qwen3tts::ID => Ok(Box::new(qwen3tts::Qwen3TtsEngine::load(config)?)),
        kokoro::engine::ID => Ok(Box::new(kokoro::engine::KokoroEngine::load(config)?)),
        other => anyhow::bail!("unknown engine `{other}`; available: {}", ids().join(", ")),
    }
}

/// What a caller that did not name an engine must be told: the default engine restricts
/// language, and only that caller knows what text it is about to send.
///
/// Deliberately not a rejection. Identifying a language from arbitrary text is a guess,
/// and a guess that refuses valid English would be worse than the warning. Callers emit
/// this only when the engine was defaulted, never when it was named.
pub fn default_caveat() -> Option<String> {
    let id = default_id();
    let languages = catalogue().into_iter().find(|c| c.id == id)?.languages?;
    Some(format!(
        "engine `{id}` was selected by default and speaks only {}. Text in any other language has no faithful path through it — pass `--engine` to choose another.",
        languages.join(", ")
    ))
}

/// The voice asset shipped for `id`. Conventions rather than configuration, like
/// [`default_root`] — kept here so the CLI, the service and `scripts/bootstrap.sh` do
/// not each carry their own copy of the mapping.
pub fn default_voice(id: &str) -> &'static str {
    match id {
        audio8::engine::ID => "voices/cosy-default",
        cosyvoice::ID => "voices/cosy-default-cosyvoice",
        qwen3tts::ID => "voices/cosy-default-qwen3tts",
        // kokoro is Cloning::None: its voices are inside the checkpoint, so there is no
        // asset to name. `every_id_has_conventions` skips it for that reason.
        kokoro::engine::ID => "",
        _ => "",
    }
}

/// The voices an engine carries inside its own checkpoint, for one that cannot clone.
///
/// Read from the asset rather than listed here: which voicepacks were installed is a fact
/// about the disk, and a caller asking "what can I pass to `--set voice=`" wants the ones
/// that are actually there. `Ok(vec![])` when the engine has none or the asset is absent.
pub fn builtin_voices(id: &str, root: &std::path::Path) -> Result<Vec<String>> {
    match id {
        kokoro::engine::ID => {
            let path = root.join("voices.safetensors");
            if !path.exists() {
                return Ok(Vec::new());
            }
            kokoro::model::Voices::names_in(&path)
        }
        _ => Ok(Vec::new()),
    }
}

/// Default model root per engine, relative to the repo. Conventions rather than
/// configuration, overridable through [`EngineConfig::overrides`].
pub fn default_root(id: &str) -> &'static str {
    match id {
        audio8::engine::ID => "references/audio8/weights",
        cosyvoice::ID => "references/cosyvoice/weights",
        qwen3tts::ID => "references/qwen3tts/weights",
        kokoro::engine::ID => "references/kokoro/weights",
        _ => ".",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_qwen3tts() {
        assert_eq!(default_id(), qwen3tts::ID);
    }

    /// Every id must resolve to a model root, and to a shipped voice exactly when it
    /// clones from one. A new engine that forgets either fails here rather than at a
    /// caller's first request.
    #[test]
    fn every_id_has_conventions() {
        for caps in catalogue() {
            let clones = caps.cloning != tts_core::Cloning::None;
            assert_eq!(
                clones,
                !default_voice(caps.id).is_empty(),
                "{} has a shipped voice iff it clones from one",
                caps.id
            );
            assert_ne!(
                default_root(caps.id),
                ".",
                "{} has no default model root",
                caps.id
            );
        }
    }

    /// The guard that replaced `qwen3tts`-is-last. If the default ever stops being a
    /// closed-list engine this returns `None` legitimately — but while it is one, a
    /// caller must be able to find that out.
    #[test]
    fn closed_list_default_is_reported() {
        let caps = catalogue();
        let default = caps.iter().find(|c| c.id == default_id()).unwrap();
        assert_eq!(default.languages.is_some(), default_caveat().is_some());
        let caveat = default_caveat().expect("qwen3tts speaks a closed list");
        assert!(caveat.contains("english") && caveat.contains(default_id()));
    }
}
