//! Terminal presentation: colour when it is wanted, a progress bar when there is a
//! terminal to draw it on, and neither when the output is going somewhere else.
//!
//! The rule this module exists to enforce is that **decoration is never part of the
//! data**. A pipe, a file, a CI log and `NO_COLOR=1` all get the same plain text, so
//! `dream-tts storage | grep` and `dream-tts config > issue.txt` behave. Escape codes
//! leaking into a redirect is the classic way a pretty CLI becomes an unusable one.
//!
//! Progress goes to stderr and the results go to stdout, so a redirect of either one still
//! makes sense on its own.

use std::io::{IsTerminal, Write};
use std::sync::OnceLock;
use std::time::Instant;

fn colour_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // Precedence follows the de-facto standard: an explicit force beats an explicit
        // disable beats terminal detection.
        if std::env::var_os("CLICOLOR_FORCE").is_some_and(|v| v != "0") {
            return true;
        }
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        if std::env::var("TERM").is_ok_and(|t| t == "dumb") {
            return false;
        }
        if std::env::var("CLICOLOR").is_ok_and(|v| v == "0") {
            return false;
        }
        std::io::stdout().is_terminal()
    })
}

fn paint(code: &str, s: &str) -> String {
    if colour_enabled() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn bold(s: &str) -> String {
    paint("1", s)
}
pub fn dim(s: &str) -> String {
    paint("2", s)
}
pub fn red(s: &str) -> String {
    paint("31", s)
}
pub fn green(s: &str) -> String {
    paint("32", s)
}
pub fn yellow(s: &str) -> String {
    paint("33", s)
}
pub fn cyan(s: &str) -> String {
    paint("36", s)
}

/// Pad to `width` on the *visible* text, then style it.
///
/// `format!("{:<22}", bold(s))` counts the escape bytes toward the width, so a coloured
/// column comes out eight characters narrower than the same column uncoloured — every table
/// here was misaligned in a terminal and correct only when piped. Padding first is the fix.
///
/// Truncates with an ellipsis rather than overflowing, so one long value cannot shift a
/// whole table's columns.
pub fn cell(text: &str, width: usize, style: fn(&str) -> String) -> String {
    let len = text.chars().count();
    if len > width {
        let cut: String = text.chars().take(width.saturating_sub(1)).collect();
        return style(&format!("{cut}…"));
    }
    format!("{}{}", style(text), " ".repeat(width - len))
}

/// A section title. Blank line before, so consecutive sections separate themselves.
pub fn heading(title: &str) {
    println!("\n{}", bold(title));
}

/// Width of the label column shared by [`field`] and [`field_note`].
const LABEL: usize = 11;

/// `label   value`, aligned to a fixed column so a block of them reads as a table.
pub fn field(label: &str, value: impl AsRef<str>) {
    println!("{} {}", cell(label, LABEL, dim), value.as_ref());
}

/// `label   value   [note]`, where the note is subordinate.
pub fn field_note(label: &str, value: impl AsRef<str>, note: &str) {
    println!(
        "{} {}  {}",
        cell(label, LABEL, dim),
        value.as_ref(),
        dim(&format!("[{note}]"))
    );
}

/// Seconds as `2m 40s` — the form the README quotes wall times in. Anything under a minute
/// keeps one decimal, because that is the resolution people compare renders at.
pub fn duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "—".into();
    }
    if seconds < 60.0 {
        return format!("{seconds:.1}s");
    }
    let total = seconds.round() as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else {
        format!("{m}m {s:02}s")
    }
}

/// A single-line progress bar, redrawn in place.
///
/// Only ever drawn when stderr is a terminal. Piped or redirected, each *stage* prints one
/// line when it finishes and nothing in between — a log with 400 carriage returns in it is
/// worse than no progress at all.
pub struct Bar {
    interactive: bool,
    stage: String,
    /// When the current stage's first event arrived — the basis for its ETA, and nothing
    /// else. See [`Bar::print_stage_line`] for why it is not used for reporting time.
    stage_started: Instant,
    total: usize,
    done: usize,
    dirty: bool,
    /// Set by [`Bar::advance_labelled`]: an estimate for something larger than this stage.
    external_eta: Option<f64>,
}

/// Bar cells. Fixed rather than terminal-width-derived: the line also carries counts and an
/// ETA, and a bar that resizes between redraws reads as flicker.
const WIDTH: usize = 24;

impl Bar {
    pub fn new() -> Self {
        Self {
            // A terminal, not a colour terminal: `NO_COLOR` asks for no escape *colours*,
            // not for a 400-line log where a redrawn line would do.
            interactive: std::io::stderr().is_terminal(),
            stage: String::new(),
            stage_started: Instant::now(),
            total: 0,
            done: 0,
            dirty: false,
            external_eta: None,
        }
    }

    pub fn planned(&mut self, segments: usize) {
        if self.interactive {
            let _ = write!(
                std::io::stderr(),
                "\r{} {} segment(s)",
                dim("planning"),
                segments
            );
            let _ = std::io::stderr().flush();
            self.dirty = true;
        }
    }

    pub fn advance(&mut self, stage: &str, done: usize, total: usize) {
        if stage != self.stage {
            // Each finished stage leaves one permanent line behind, interactive or not.
            // That is the record of where the time went; the redrawn bar is only the live
            // view of the stage in flight.
            if !self.stage.is_empty() {
                self.print_stage_line();
            }
            self.stage = stage.to_string();
            self.stage_started = Instant::now();
        }
        self.done = done;
        self.total = total;
        if self.interactive {
            self.draw();
        }
    }

    /// Position only, never a duration.
    ///
    /// A stage whose work happens inside one call reports a single event *after* it — the
    /// qwen3tts codec decodes the whole utterance at once. Timing from that event measures
    /// nothing and would print `0.0s` for 3.5 seconds of work. The synchronised stage
    /// breakdown printed at the end is the only honest source for time, so progress does
    /// not compete with it. See docs/reference.md#how-to-measure-without-fooling-yourself.
    /// Advance with a caller-supplied label and estimate.
    ///
    /// The job runner knows the ETA for the whole book, which is the number a listener
    /// actually wants; the stage's own rate only describes the chapter in flight.
    pub fn advance_labelled(&mut self, label: &str, done: usize, total: usize, eta: Option<f64>) {
        self.external_eta = eta;
        self.advance(label, done, total);
    }

    fn print_stage_line(&self) {
        if self.interactive {
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
        }
        let _ = writeln!(
            std::io::stderr(),
            "  {:<8} {:>4}/{}",
            self.stage,
            self.done,
            self.total
        );
    }

    fn draw(&mut self) {
        let frac = if self.total == 0 {
            0.0
        } else {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0)
        };
        let filled = (frac * WIDTH as f64).round() as usize;
        let bar: String = "━".repeat(filled) + &"─".repeat(WIDTH - filled);

        // ETA from this stage's own observed rate, which is legitimate while the stage is
        // still running. Extrapolating across stages would be a guess: an engine's stages
        // differ in cost per segment by an order of magnitude.
        let elapsed = self.stage_started.elapsed().as_secs_f64();
        let eta = match self.external_eta {
            Some(seconds) => format!("  {} {}", dim("eta"), duration(seconds)),
            None if self.done == 0 || frac >= 1.0 => String::new(),
            None => {
                let remaining = elapsed / self.done as f64 * (self.total - self.done) as f64;
                format!("  {} {}", dim("eta"), duration(remaining))
            }
        };

        let _ = write!(
            std::io::stderr(),
            "\r\x1b[2K  {:<8} {} {:>3}%  {}/{}{}",
            cyan(&self.stage),
            bar,
            (frac * 100.0).round() as u32,
            self.done,
            self.total,
            eta
        );
        let _ = std::io::stderr().flush();
        self.dirty = true;
    }

    /// Clear the bar so whatever prints next starts on a clean line.
    pub fn finish(&mut self) {
        if self.interactive && self.dirty {
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
            let _ = std::io::stderr().flush();
        } else if !self.interactive && !self.stage.is_empty() {
            self.print_stage_line();
        }
        self.dirty = false;
        self.stage.clear();
    }
}

impl Default for Bar {
    fn default() -> Self {
        Self::new()
    }
}

/// Render an error and its causes as a readable block rather than one colon-joined line.
///
/// anyhow's `{:#}` produces `loading engine qwen3tts: opening weights: No such file`, which
/// buries the actionable part at the end. The chain reads better as a list.
pub fn report_error(err: &anyhow::Error) {
    eprintln!("{} {}", red("error:"), err);
    let mut source = err.source();
    while let Some(cause) = source {
        eprintln!("  {} {cause}", dim("caused by:"));
        source = cause.source();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property that keeps a table aligned once colour is switched on.
    #[test]
    fn cells_pad_on_the_visible_text() {
        // Not a terminal here, so the style is a passthrough and the widths are literal.
        assert_eq!(cell("ab", 5, bold), "ab   ");
        assert_eq!(cell("abcdefg", 5, bold), "abcd…");
        assert_eq!(cell("abcde", 5, bold), "abcde");
        // Styling must not change the padding: bold is a passthrough here, and in a
        // terminal the escape bytes sit outside the padded text rather than inside it.
        assert_eq!(cell("ab", 5, bold).chars().count(), 5);
    }

    #[test]
    fn durations_read_the_way_the_docs_quote_them() {
        assert_eq!(duration(9.44), "9.4s");
        assert_eq!(duration(160.0), "2m 40s");
        assert_eq!(duration(3600.0 * 4.0 + 120.0), "4h 02m");
        assert_eq!(duration(f64::NAN), "—");
    }

    /// The tests run without a terminal, so every helper must be a no-op passthrough.
    /// This is the property that keeps escape codes out of redirected output.
    #[test]
    fn no_escape_codes_when_not_a_terminal() {
        for s in [bold("x"), dim("x"), red("x"), green("x"), cyan("x")] {
            assert_eq!(s, "x", "colour leaked into non-terminal output");
        }
    }
}
