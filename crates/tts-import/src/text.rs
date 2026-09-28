//! Bytes to text, when a file does not say how it was encoded.
//!
//! UTF-8 is taken as it is. A byte-order mark names UTF-16 or UTF-8. Otherwise, text that is not
//! valid UTF-8 is read as Windows-1252, which is what an old `.txt` from a Windows machine almost
//! always is — and unlike a lossy UTF-8 read, it keeps `café` rather than `caf\u{fffd}`.

pub fn decode(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return utf16(rest, u16::from_le_bytes);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return utf16(rest, u16::from_be_bytes);
    }
    // UTF-16 without a mark: ASCII text has a zero in every other byte. Checked before UTF-8,
    // which a zero byte does not make invalid.
    let sample = &bytes[..bytes.len().min(4096)];
    let zeros_odd = sample
        .iter()
        .skip(1)
        .step_by(2)
        .filter(|b| **b == 0)
        .count();
    let zeros_even = sample.iter().step_by(2).filter(|b| **b == 0).count();
    let half = sample.len() / 2;
    if half > 8 && zeros_odd * 10 > half * 6 {
        return utf16(bytes, u16::from_le_bytes);
    }
    if half > 8 && zeros_even * 10 > half * 6 {
        return utf16(bytes, u16::from_be_bytes);
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        return text.to_string();
    }
    bytes.iter().map(|b| cp1252(*b)).collect()
}

fn utf16(bytes: &[u8], unit: fn([u8; 2]) -> u16) -> String {
    let units: Vec<u16> = bytes.chunks_exact(2).map(|p| unit([p[0], p[1]])).collect();
    String::from_utf16_lossy(&units)
}

/// One Windows-1252 byte. Latin-1 everywhere except 0x80–0x9F, where Windows put the curly
/// quotes, dashes and the euro sign.
pub fn cp1252(byte: u8) -> char {
    const HIGH: [char; 32] = [
        '\u{20ac}', '\u{fffd}', '\u{201a}', '\u{0192}', '\u{201e}', '\u{2026}', '\u{2020}',
        '\u{2021}', '\u{02c6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{fffd}',
        '\u{017d}', '\u{fffd}', '\u{fffd}', '\u{2018}', '\u{2019}', '\u{201c}', '\u{201d}',
        '\u{2022}', '\u{2013}', '\u{2014}', '\u{02dc}', '\u{2122}', '\u{0161}', '\u{203a}',
        '\u{0153}', '\u{fffd}', '\u{017e}', '\u{0178}',
    ];
    match byte {
        0x80..=0x9F => HIGH[(byte - 0x80) as usize],
        _ => byte as char,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_encoding_a_text_file_arrives_in() {
        assert_eq!(decode("café".as_bytes()), "café");
        assert_eq!(decode(b"\xEF\xBB\xBFhi"), "hi");
        assert_eq!(
            decode(b"caf\xe9 \x93quoted\x94"),
            "café \u{201c}quoted\u{201d}"
        );
        let le: Vec<u8> = "plain ascii text here"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        assert_eq!(decode(&le), "plain ascii text here");
        let mut marked = vec![0xFF, 0xFE];
        marked.extend(&le);
        assert_eq!(decode(&marked), "plain ascii text here");
    }
}
