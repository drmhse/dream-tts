//! The English frontend: text to the phoneme string Kokoro's model consumes.
//!
//! Three ports, gated separately, because a single end-to-end comparison cannot say which
//! of them moved: spaCy's tokenizer, spaCy's POS tagger, and misaki's lexicon and rules.

pub mod g2p;
pub mod lexicon;
pub mod murmur;
pub mod num2words;
pub mod tagger;
pub mod tokenize;
pub mod vocab;

pub use tagger::Tagger;
pub use tokenize::{Token, Tokenizer};
pub use vocab::Vocab;
