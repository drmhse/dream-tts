//! The two hashes spaCy's tagger needs. Both are transcribed from the implementations
//! actually used, not from the published MurmurHash reference: `thinc_hash` is named
//! x86_128 upstream and is neither — it has no blocks, no tail, and 64-bit constants.

/// MurmurHash64A — spaCy's `hash_string`, which produces a StringStore key.
pub fn murmur64a(data: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4a793_5bd1e995;
    const R: u32 = 47;
    let mut h = seed ^ (data.len() as u64).wrapping_mul(M);
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let mut k = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }
    let tail = &data[chunks * 8..];
    if !tail.is_empty() {
        let mut buf = [0u8; 8];
        buf[..tail.len()].copy_from_slice(tail);
        h ^= u64::from_le_bytes(buf);
        h = h.wrapping_mul(M);
    }
    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

/// thinc's `ops.hash`: one 64-bit key to the four buckets a HashEmbed table sums.
pub fn thinc_hash(key: u64, seed: u32) -> [u32; 4] {
    fn fmix64(mut h: u64) -> u64 {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7_ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe_1a85ec53);
        h ^ (h >> 33)
    }
    let mut h1 = key.wrapping_mul(0x87c37b91_114253d5);
    h1 = h1.rotate_left(31);
    h1 = h1.wrapping_mul(0x4cf5ad43_2745937f);
    h1 ^= seed as u64;
    h1 ^= 8;
    let mut h2 = (seed as u64) ^ 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [h1 as u32, (h1 >> 32) as u32, h2 as u32, (h2 >> 32) as u32]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values taken from the pinned spaCy/thinc build; see references/kokoro.
    #[test]
    fn matches_upstream() {
        assert_eq!(murmur64a(b"the", 1), 7425985699627899538);
        assert_eq!(murmur64a("naïve".as_bytes(), 1), 16224804062388677344);
        assert_eq!(
            thinc_hash(7425985699627899538, 8),
            [2332737498, 1315472909, 1351622599, 1973209364]
        );
    }
}
