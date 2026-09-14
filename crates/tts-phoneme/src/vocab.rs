//! Strings to keys, the way spaCy's StringStore and NORM chain do it.

use crate::murmur::murmur64a;
use std::collections::HashMap;

pub struct Vocab {
    /// Symbol names resolve to small integer ids, not to hashes. Every empty string and
    /// shapes like "X" land here; without the table those tokens embed from the wrong row
    /// and nothing reports it.
    pub symbols: HashMap<String, u64>,
    /// Currency symbols and smart quotes folded onto one representative, keyed by orth.
    pub base_norms: HashMap<String, String>,
    /// Keyed by the hash of the orth, not by the word — a table keyed by string compiles,
    /// runs, and never hits.
    pub lexeme_norm: HashMap<u64, String>,
}

impl Vocab {
    pub fn string_id(&self, s: &str) -> u64 {
        match self.symbols.get(s) {
            Some(id) => *id,
            None => murmur64a(s.as_bytes(), 1),
        }
    }

    pub fn norm(&self, text: &str) -> String {
        if let Some(n) = self.base_norms.get(text) {
            return n.clone();
        }
        if let Some(n) = self.lexeme_norm.get(&murmur64a(text.as_bytes(), 1)) {
            return n.clone();
        }
        text.to_lowercase()
    }
}
