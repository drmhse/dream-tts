//! Markdown to narration text.
//!
//! Feeding raw markdown to a TTS engine reads the syntax aloud: "plus plus plus", "hash
//! hash". Worse, some of what survives is actively destructive — a `***` that reached the
//! voice made one chapter say "asterisk, asterisk, asterisk" and then degenerate into a
//! repetition loop for the rest of the passage.
//!
//! This is a port of `scripts/md-to-narration.py`, which stays in the tree as the
//! long-form record of *why* each rule exists and as the oracle the `narrate-diff` binary
//! checks this against. Every rule here was arrived at by listening to a failure; the
//! comments record the constraint, not the narrative.

pub mod abbrev;
pub mod align;
pub mod blocks;
pub mod code;
pub mod inline;
pub mod lint;
pub mod math;
pub mod numbers;
pub mod page;
pub mod source;
pub mod tables;

mod re;

pub use blocks::{convert, Options};
pub use code::speak_code;
pub use inline::clean_inline;
pub use math::speak_math;
pub use numbers::speak_numbers;
pub use page::page_text;
