//! Apply one narration function to JSON cases on stdin, for differential testing.
//!
//! The mirror of `scripts/narrate-oracle.py`. `scripts/check-narrate.sh` runs both over the
//! same corpus and requires byte-identical output — which is the only real evidence that a
//! port of a thousand regexes matches the original.
//!
//!     echo '["7B parameters"]' | narrate-diff clean_inline

use anyhow::{bail, Result};
use std::io::Read;

fn main() -> Result<()> {
    let name = std::env::args().nth(1).unwrap_or_default();
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let out: Vec<String> = match name.as_str() {
        "speak_code" => map(&input, tts_narrate::speak_code)?,
        "speak_math" => map(&input, tts_narrate::speak_math)?,
        "speak_numbers" => map(&input, tts_narrate::speak_numbers)?,
        "clean_inline" => map(&input, tts_narrate::clean_inline)?,
        "convert" => map(&input, |s| tts_narrate::convert(s, &Default::default()))?,
        "page_text" => map(&input, tts_narrate::page_text)?,
        "align" => {
            // `[[spoken, page], …]` -> the mapping for each pair, as JSON.
            let pairs: Vec<(Vec<String>, Vec<String>)> = serde_json::from_str(&input)?;
            let mapped: Vec<Vec<i64>> = pairs
                .iter()
                .map(|(a, b)| tts_narrate::align::align_tokens(a, b))
                .collect();
            println!("{}", serde_json::to_string(&mapped)?);
            return Ok(());
        }
        other => bail!("unknown function `{other}`"),
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

fn map<F: Fn(&str) -> String>(input: &str, f: F) -> Result<Vec<String>> {
    let cases: Vec<String> = serde_json::from_str(input)?;
    Ok(cases.iter().map(|c| f(c)).collect())
}
