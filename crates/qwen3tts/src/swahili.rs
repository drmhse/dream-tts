//! Spelling Swahili for a talker whose ten languages never open a word on a nasal cluster.
//! Token input only: narration text and alignment keep the real spelling.

/// A word opening `mb mv nd ng nj nz` (not `ng'`, which is one sound).
fn nasal_onset(w: &[char]) -> bool {
    match w {
        [m, b, ..] if m.eq_ignore_ascii_case(&'m') && matches!(b, 'b' | 'v') => true,
        [n, c, rest @ ..] if n.eq_ignore_ascii_case(&'n') && matches!(c, 'd' | 'g' | 'j' | 'z') => {
            !(*c == 'g' && rest.first().is_some_and(|a| matches!(a, '\'' | '’')))
        }
        _ => false,
    }
}

fn opens_on_nasal(w: &[char]) -> bool {
    nasal_onset(w) || (w.len() > 2 && w[0].eq_ignore_ascii_case(&'n') && w[1] == 'g' && matches!(w[2], '\'' | '’'))
}

/// Nothing but spaces, quotes or brackets since the segment start or a `.!?`.
fn opens_sentence(before: &[char]) -> bool {
    before
        .iter()
        .rev()
        .find(|c| !c.is_whitespace() && !matches!(c, '"' | '“' | '‘' | '(' | '\''))
        .is_none_or(|c| matches!(c, '.' | '!' | '?'))
}

/// "M-buzi" where a sentence opens: unmarked, the talker dropped sentence-initial nasals
/// (13/33 kept against 27/33, unprimed transcription). Only there — mid-sentence the plain
/// spelling is spoken right and the hyphen adds a vowel ("em-binu"). A segment opening on a
/// nasal also gets `lead`: the first word after the reference clip lost its onset whatever the
/// spelling ("Ng'ombe" 0/3, 3/3 behind "... ").
pub fn respell(segment: &str, lead: &str) -> String {
    let chars: Vec<char> = segment.chars().collect();
    let mut out = String::with_capacity(segment.len() + 8);
    let first = chars.iter().position(|c| c.is_alphabetic()).unwrap_or(0);
    if opens_on_nasal(&chars[first..]) {
        out.push_str(lead);
    }
    for (i, &c) in chars.iter().enumerate() {
        out.push(c);
        let word_start = i == 0 || !chars[i - 1].is_alphanumeric() && chars[i - 1] != '\'';
        if word_start && opens_sentence(&chars[..i]) && nasal_onset(&chars[i..]) {
            out.push('-');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::respell;

    #[test]
    fn hyphenates_prenasal_onsets() {
        assert_eq!(respell("Mbuzi na ndege. Ndege ni ndogo.", "... "), "... M-buzi na ndege. N-dege ni ndogo.");
        assert_eq!(respell("Ng'ombe, nguo! Njia? Mvua.", "... "), "... Ng'ombe, nguo! N-jia? M-vua.");
        assert_eq!(respell("Kwa kweli mbinu hii.", "... "), "Kwa kweli mbinu hii.");
        assert_eq!(respell("Omba, mtu, nyumba, simba.", "... "), "Omba, mtu, nyumba, simba.");
        assert_eq!(respell("“Mbwa”", "— "), "— “M-bwa”");
    }
}
