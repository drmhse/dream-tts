//! Narration as an engine can carry it: what the markdown rules leave that would still break a
//! sentence's flow when spoken.

/// Arithmetic said as words. A spaced `*` is a multiplication in prose, and left in the
/// markdown two of them pair up as emphasis and both vanish: "300 million * 50%" was spoken
/// "300 million 50 percent".
pub fn operators(source: &str) -> String {
    source
        .replace(" * ", " times ")
        .replace(" \u{d7} ", " times ")
        .replace(" / ", " divided by ")
        .replace(" = ", " equals ")
}

/// Text an engine can carry a sentence through.
///
/// A bullet is where a list item ends, and an item, a heading or a label with no closing mark
/// ends on a rising voice that runs into whatever is said next — so each gets the full stop the
/// page implied. Space before punctuation, which PDF extraction leaves behind, goes too.
pub fn flowing(narration: &str) -> String {
    const BULLETS: [char; 6] = [
        '\u{2022}', '\u{25e6}', '\u{25aa}', '\u{25cf}', '\u{2023}', '\u{2043}',
    ];
    let mut out = String::with_capacity(narration.len() + 8);
    for token in narration.split_whitespace() {
        if token.chars().all(|c| BULLETS.contains(&c)) {
            close(&mut out);
            continue;
        }
        if token.starts_with([',', '.', ';', ':', '!', '?']) && !out.is_empty() {
            out.push_str(token);
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(token);
    }
    // A lone quote at the end opens the next paragraph: extraction left it on this one's line.
    while let Some(last) = out
        .split(' ')
        .next_back()
        .filter(|t| t.chars().all(is_quote))
    {
        let keep = out.len() - last.len();
        out.truncate(keep);
        out.truncate(out.trim_end().len());
    }
    close(&mut out);
    out
}

fn is_quote(c: char) -> bool {
    matches!(
        c,
        '"' | '\u{201c}' | '\u{201d}' | '\'' | '\u{2018}' | '\u{2019}'
    )
}

fn close(text: &mut String) {
    let bare = text
        .trim_end_matches(['"', '\u{201d}', '\u{2019}', '\'', ')', ']'])
        .trim_end();
    if !bare.is_empty() && !bare.ends_with(['.', '!', '?', ':', ';', ',', '\u{2026}']) {
        text.insert(bare.len(), '.');
    }
}

/// A whole document's narration, for a pipeline that speaks it paragraph by paragraph: the
/// arithmetic said, the markdown rules applied, and each paragraph ended as `flowing` ends one.
pub fn narration(markdown: &str, options: &tts_narrate::Options) -> String {
    tts_narrate::convert(&operators(markdown), options)
        .lines()
        .map(|line| {
            if line.trim().is_empty() {
                String::new()
            } else {
                flowing(line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_read_from_a_pdf_ends_each_item() {
        assert_eq!(
            flowing("Assumptions: \u{2022} 300 million users \u{2022} Data is kept \u{2022}"),
            "Assumptions: 300 million users. Data is kept."
        );
        assert_eq!(
            flowing("Jimmy?\" , the teacher said"),
            "Jimmy?\", the teacher said."
        );
        assert_eq!(
            flowing("Why did the tiger roar?\u{201d}"),
            "Why did the tiger roar?\u{201d}"
        );
        assert_eq!(flowing("A 4-step process"), "A 4-step process.");
        assert_eq!(
            flowing("the teacher responded. \""),
            "the teacher responded."
        );
        assert_eq!(flowing("\"Very good Jimmy. \u{201c}"), "\"Very good Jimmy.");
    }

    #[test]
    fn a_document_keeps_its_paragraphs_and_ends_each() {
        let md = "# Title\n\nFirst line of a paragraph\n\n- one item\n- two item\n\nQPS = 2 * 3\n";
        let out = narration(md, &tts_narrate::Options::default());
        let paragraphs: Vec<&str> = out.split("\n").filter(|l| !l.trim().is_empty()).collect();
        assert!(paragraphs.len() >= 4, "{out:?}");
        assert!(
            paragraphs
                .iter()
                .all(|p| p.ends_with(['.', '!', '?', ':', ';', ','])),
            "{out:?}"
        );
        assert!(out.contains("times"), "{out:?}");
    }

    #[test]
    fn arithmetic_is_said_rather_than_dropped() {
        assert_eq!(
            operators("150 million * 2 tweets / 24 hour"),
            "150 million times 2 tweets divided by 24 hour"
        );
        assert_eq!(operators("QPS = ~3500"), "QPS equals ~3500");
        assert_eq!(operators("read/write *emphasis*"), "read/write *emphasis*");
    }
}
