//! Kokoro-82M: a non-autoregressive StyleTTS2 derivative.
//!
//! Unlike the other three engines here there is no token-by-token loop — duration is
//! predicted for every phoneme at once, the encoding is stretched to match, and one pass
//! through an iSTFTNet decoder produces the waveform.

pub mod albert;
pub mod blocks;
pub mod cfg;
pub mod decoder;
pub mod engine;
pub mod lstm;
pub mod model;
pub mod predictor;
pub mod source;
pub mod text_encoder;

pub const ID: &str = "kokoro";
