//! A book's title page, as an edition marks it up: the credits under the title are headings
//! there, so "By Rudyard Kipling" was read out as a chapter and listed in the contents.

/// The credit lines set as headings under the book's own title — "by Mary Shelley", "A NEW
/// TRANSLATION BY" and the name after it, "or, the Modern Prometheus" — as the lines of text
/// they are. Only headings straight after the title, before any prose, are looked at.
pub(crate) fn credits(markdown: &str, title: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    let mut stage = Stage::Before;
    let mut lead_in = false;
    for line in markdown.split_inclusive('\n') {
        let text = line.trim();
        let heading = text.starts_with('#').then(|| plain(text));
        stage = match (stage, heading) {
            (Stage::Before, Some(h)) if same_title(&h, title) => Stage::Title,
            (Stage::Before, Some(_)) => Stage::Done,
            (Stage::Title, Some(h)) if lead_in || is_credit(&h) => {
                lead_in = leads_in(&h);
                out.push_str(line.trim_start().trim_start_matches('#').trim_start());
                continue;
            }
            (Stage::Title, Some(_)) => Stage::Done,
            (Stage::Title, None)
                if text.is_empty()
                    || text == "---"
                    || text.starts_with("![")
                    || text.starts_with("[![") =>
            {
                Stage::Title
            }
            (Stage::Title, None) => Stage::Done,
            (stage, _) => stage,
        };
        out.push_str(line);
    }
    out
}

#[derive(Clone, Copy)]
enum Stage {
    Before,
    Title,
    Done,
}

fn is_credit(heading: &str) -> bool {
    let h = heading.to_lowercase();
    h.starts_with("by ")
        || h.starts_with("or,")
        || h.starts_with("or ")
        || h.contains(" by ")
        || leads_in(&h)
}

/// "A NEW TRANSLATION BY": the name is the heading after it.
fn leads_in(heading: &str) -> bool {
    let h = heading.to_lowercase();
    h == "by" || h.ends_with(" by")
}

fn same_title(heading: &str, title: &str) -> bool {
    let (h, t) = (words(heading), words(title));
    !h.is_empty() && (h == t || t.starts_with(&format!("{h} ")) || h.starts_with(&format!("{t} ")))
}

/// Letters and digits only, lowercase, one space between words.
fn words(text: &str) -> String {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn plain(heading: &str) -> String {
    heading
        .trim_start_matches('#')
        .chars()
        .filter(|c| !matches!(c, '*' | '_'))
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::credits;

    #[test]
    fn credits_under_the_title_become_text() {
        let md = "# AESOP'S FABLES\n\n#### A NEW TRANSLATION BY\n\n## V. S. VERNON JONES\n\n\
                  #### AND ILLUSTRATIONS BY\n\n### ARTHUR RACKHAM\n\n![Title](t.png)\n\n\
                  ## INTRODUCTION\n\nAesop embodies an epigram.\n";
        assert_eq!(
            credits(md, "Aesop's Fables"),
            "# AESOP'S FABLES\n\nA NEW TRANSLATION BY\n\nV. S. VERNON JONES\n\n\
             AND ILLUSTRATIONS BY\n\nARTHUR RACKHAM\n\n![Title](t.png)\n\n\
             ## INTRODUCTION\n\nAesop embodies an epigram.\n"
        );
    }

    #[test]
    fn a_subtitle_and_a_byline_become_text() {
        let md = "# Frankenstein;\n\n### or, the Modern Prometheus\n\n## by Mary Shelley\n\n## CONTENTS\n";
        assert_eq!(
            credits(md, "Frankenstein; Or, The Modern Prometheus"),
            "# Frankenstein;\n\nor, the Modern Prometheus\n\nby Mary Shelley\n\n## CONTENTS\n"
        );
    }

    #[test]
    fn a_chapter_named_like_a_credit_is_left_alone() {
        let md = "# The Book\n\nIt begins.\n\n## By the Sea\n\nWaves.\n";
        assert_eq!(credits(md, "The Book"), md);
    }

    #[test]
    fn a_book_whose_first_heading_is_not_its_title_is_left_alone() {
        let md = "# Cover\n\n## By Someone\n";
        assert_eq!(credits(md, "The Book"), md);
    }
}
