//! From a document's blocks to what an engine is asked to say: sentences, grouped into chunks
//! that keep their prosody, each carrying the canonical words it covers.

pub mod chunk;
pub mod sentence;
pub mod text;

pub use chunk::Chunk;
