//! Kokoro-82M: a non-autoregressive StyleTTS2 derivative.
//!
//! Unlike the other three engines here there is no token-by-token loop — duration is
//! predicted for every phoneme at once, the encoding is stretched to match, and one pass
//! through an iSTFTNet decoder produces the waveform.
//!
//! Also home to `nano` (`standalone`, `nano_engine`): a 3.27M-parameter model distilled from Kokoro that
//! shares its front end and voice tables and none of its networks.

pub mod albert;
pub mod blocks;
pub mod cfg;
pub mod decoder;
pub mod engine;
pub mod lstm;
pub mod model;
pub mod nano;
pub mod nano_engine;
pub mod predictor;
pub mod source;
pub mod standalone;
pub mod text_encoder;

pub const ID: &str = "kokoro";
