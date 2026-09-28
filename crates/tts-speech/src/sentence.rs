//! Splitting narration into sentences.
//!
//! Narration text has already been through the sibling's rules, so it is prose: no markdown, no
//! table pipes, headings ended with a full stop. What is left is the ordinary problem of
//! telling a sentence end from an abbreviation, and the ordinary answer — a full stop followed
//! by a space and a capital — is right often enough that the cost of being wrong is one
//! highlight lasting two sentences instead of one.

/// Abbreviations that end in a full stop and do not end a sentence. Short, because narration
/// has already expanded most of them: this catches what survives.
const NOT_AN_END: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e", "fig", "no",
    "approx", "al",
];

/// Sentence boundaries, as byte offsets just past each terminator.
pub fn split(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;

    for (i, c) in text.char_indices() {
        if !matches!(c, '.' | '!' | '?') {
            continue;
        }
        let after = i + c.len_utf8();
        // A terminator at the very end closes the last sentence below.
        let Some(next) = text[after..].chars().next() else {
            continue;
        };
        if !next.is_whitespace() {
            continue;
        }
        // "Chapter 1. The" is a sentence end; "Dr. Who" is not.
        if c == '.' && is_abbreviation(&text[start..i]) {
            continue;
        }
        // A decimal point never has a space after it, so it is already excluded — what is left
        // is the opening of the next sentence, which is a capital or a digit in ordinary prose.
        let opens = text[after..]
            .chars()
            .find(|c| !c.is_whitespace())
            .is_some_and(|c| c.is_uppercase() || c.is_numeric() || c == '"' || c == '\u{201c}');
        if !opens {
            continue;
        }
        let piece = text[start..after].trim();
        if !piece.is_empty() {
            out.push(piece);
        }
        start = after;
        let _ = bytes;
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn is_abbreviation(before: &str) -> bool {
    let word: String = before
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric() || *c == '.')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let word = word.trim_end_matches('.').to_lowercase();
    // A single letter is an initial: "J. R. R. Tolkien".
    word.chars().count() == 1 || NOT_AN_END.contains(&word.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_sentences_are_separated() {
        assert_eq!(
            split("One thing happened. Then another. And a third."),
            ["One thing happened.", "Then another.", "And a third."]
        );
    }

    #[test]
    fn questions_and_exclamations_end_a_sentence() {
        assert_eq!(split("Why not? Because."), ["Why not?", "Because."]);
    }

    /// The case the whole module exists for: a decimal is not a sentence end, and narration is
    /// full of them.
    #[test]
    fn a_decimal_point_does_not_end_a_sentence() {
        assert_eq!(
            split("It runs at 0.148 RTF on this machine."),
            ["It runs at 0.148 RTF on this machine."]
        );
    }

    #[test]
    fn an_abbreviation_does_not_end_a_sentence() {
        assert_eq!(split("Dr. Who arrived."), ["Dr. Who arrived."]);
        assert_eq!(
            split("Frames, codes, etc. Then the decoder."),
            ["Frames, codes, etc. Then the decoder."]
        );
        assert_eq!(
            split("J. R. R. Tolkien wrote it."),
            ["J. R. R. Tolkien wrote it."]
        );
    }

    /// A lower-case opening is a continuation, not a new sentence — which is what keeps a
    /// version number or a file extension from splitting a line.
    #[test]
    fn a_lower_case_opening_is_a_continuation() {
        assert_eq!(
            split("See chapter 2. then continue"),
            ["See chapter 2. then continue"]
        );
    }

    #[test]
    fn a_number_or_a_quote_can_open_a_sentence() {
        assert_eq!(
            split("It ended. 1999 was the year."),
            ["It ended.", "1999 was the year."]
        );
        assert_eq!(
            split(r#"He spoke. "Not yet.""#),
            ["He spoke.", r#""Not yet.""#]
        );
    }

    #[test]
    fn text_without_a_terminator_is_one_sentence() {
        assert_eq!(split("No full stop here"), ["No full stop here"]);
        assert_eq!(split(""), Vec::<&str>::new());
        assert_eq!(split("   "), Vec::<&str>::new());
    }

    /// Every sentence must reassemble to the original, or the narration would lose words.
    #[test]
    fn the_pieces_reassemble_to_the_whole() {
        for text in [
            "One. Two. Three.",
            "Dr. Who arrived. It was 0.148 RTF. Then 1999.",
            "A trailing fragment without an end",
        ] {
            let joined = split(text).join(" ");
            let want: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
            assert_eq!(joined, want, "{text:?}");
        }
    }
}
