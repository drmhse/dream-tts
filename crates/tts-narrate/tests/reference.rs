//! The reference suite, ported case for case from `scripts/test_md_to_narration.py`.
//!
//! Every case here is one thing a listener heard wrong. `scripts/check-narrate.sh` proves
//! this crate matches the Python over thousands of generated cases, but that needs Python;
//! these are the cases that must hold with nothing else installed, and they are the ones
//! with a story behind them.

use tts_narrate::{clean_inline, convert, Options};

fn c(s: &str) -> String {
    clean_inline(s)
}

fn doc(s: &str) -> String {
    convert(s, &Options::default())
}

// ----------------------------------------------------------------------------- numbers

#[test]
fn decimals_and_versions_survive() {
    assert_eq!(
        c("Latency rose to 12.4 seconds."),
        "Latency rose to 12.4 seconds."
    );
    assert_eq!(c("Shipped in v1.2.3."), "Shipped in v1.2.3.");
}

#[test]
fn ranges_are_read_as_ranges() {
    assert_eq!(
        c("Temperatures of 20–25 degrees"),
        "Temperatures of 20 to 25 degrees"
    );
    assert_eq!(c("6-12 months of demand"), "6 to 12 months of demand");
    assert_eq!(c("over FY2024–25"), "over FY2024 to 2025");
}

#[test]
fn scientific_notation() {
    assert_eq!(c("fell to 1.5e-3"), "fell to 1.5 times ten to the minus 3");
    assert_eq!(c("about 10^-3"), "about ten to the minus 3");
    assert_eq!(c("2.1 × 10^22 FLOPs"), "2.1 times ten to the 22 FLOPs");
}

#[test]
fn intervals_are_not_lists() {
    assert_eq!(c("95% CI [1.2, 3.4]"), "95 percent CI 1.2 to 3.4");
}

#[test]
fn comparisons_are_spoken() {
    assert_eq!(c("p < 0.05"), "p less than 0.05");
    assert_eq!(c("N = 42 per arm"), "N equals 42 per arm");
    assert_eq!(
        c("Supply = Demand + Net Exports"),
        "Supply equals Demand plus Net Exports"
    );
}

#[test]
fn an_equation_wrapped_across_lines_keeps_its_operator() {
    assert_eq!(
        c("Generation = Consumption +"),
        "Generation equals Consumption plus"
    );
}

#[test]
fn iso_dates() {
    assert_eq!(c("from 2021-03-04 to then"), "from March 4, 2021 to then");
}

#[test]
fn magnitudes_and_percentage_points() {
    assert_eq!(c("7B parameters"), "7 billion parameters");
    assert_eq!(c("a gain of 17.6 pp"), "a gain of 17.6 percentage points");
}

// ------------------------------------------------------------------------------- units

#[test]
fn rates_are_rates() {
    assert_eq!(
        c("Throughput was 3.2 GB/s"),
        "Throughput was 3.2 gigabytes per second"
    );
    assert_eq!(c("60 %/yr"), "60 percent per year");
}

#[test]
fn an_alternation_is_not_a_rate() {
    assert_eq!(
        c("every bus/node on the grid"),
        "every bus or node on the grid"
    );
}

#[test]
fn number_agreement() {
    assert_eq!(
        c("consume 300 MW of power"),
        "consume 300 megawatts of power"
    );
    assert_eq!(
        c("plug in 1 MW for an hour"),
        "plug in 1 megawatt for an hour"
    );
    assert_eq!(
        c("a 4.4 GW natural gas plant"),
        "a 4.4 gigawatt natural gas plant"
    );
}

#[test]
fn a_gloss_defines_the_abbreviation_it_names() {
    assert_eq!(
        c("measured in megawatts (MW) for most units"),
        "measured in megawatts (MW) for most units"
    );
}

#[test]
fn a_bare_power_unit_is_still_spoken() {
    assert_eq!(c("50 dollars per MWh"), "50 dollars per megawatt hour");
}

// ---------------------------------------------------------------------------- currency

#[test]
fn a_magnitude_suffix_keeps_its_fraction() {
    assert_eq!(
        c("Costs were ~$1.5M over the year"),
        "Costs were about 1.5 million dollars over the year"
    );
}

#[test]
fn a_price_range_carries_the_unit_at_both_ends() {
    assert_eq!(
        c("anywhere from $10-150 per MWh"),
        "anywhere from 10 dollars to 150 dollars per megawatt hour"
    );
}

#[test]
fn a_floor_is_not_an_addition() {
    assert_eq!(
        c("like $100M+ machines"),
        "like 100 million dollars or higher machines"
    );
}

#[test]
fn cents_and_attributive_use_still_work() {
    assert_eq!(
        c("credit $12.50 to revenue"),
        "credit 12 dollars 50 cents to revenue"
    );
    assert_eq!(c("a $12 platform fee"), "a 12 dollar platform fee");
}

// ----------------------------------------------------------------------- abbreviations

#[test]
fn abbreviations_expand() {
    assert_eq!(
        c("energy (e.g. power), agriculture"),
        "energy (for example power), agriculture"
    );
    assert_eq!(
        c("the DA market vs. the RT market"),
        "the DA market versus the RT market"
    );
    assert_eq!(c("cf. Fig. 2 and Table 1"), "compare Figure 2 and Table 1");
    assert_eq!(
        c("vol. 33, no. 2, pp. 2175–2183"),
        "volume 33, number 2, pages 2175 to 2183"
    );
}

#[test]
fn an_abbreviation_can_close_a_line() {
    assert_eq!(c("the Western vs."), "the Western versus");
}

#[test]
fn initials_collapse_only_in_runs() {
    assert_eq!(
        c("Dr. J. R. R. Tolkien, Ph.D."),
        "Doctor J R R Tolkien, PhD"
    );
    // A lone capital before a full stop ends a sentence; stripping it welds two together.
    assert_eq!(
        c("There's 10 MWh of demand at A. The model solves it."),
        "There's 10 megawatt hours of demand at A. The model solves it."
    );
}

// ------------------------------------------------------------------------------- maths

#[test]
fn an_inline_span_is_verbalised() {
    assert_eq!(
        c(r"where $\alpha = 0.9$ and $\beta \in [0, 1]$"),
        "where alpha equals 0.9 and beta in the range 0 to 1"
    );
}

#[test]
fn operators_and_accents() {
    assert_eq!(
        c(r"$\hat{\theta} = \arg\max_\theta \sum_i \log p(x_i)$"),
        "theta hat equals arg max over theta the sum over i of log p(x i)"
    );
}

#[test]
fn display_maths_is_shown_not_spoken() {
    assert_eq!(
        doc("Before.\n\n$$\n\\sum_{i=1}^{N} x_i\n$$\n\nAfter.\n"),
        "Before.\n\nAfter.\n"
    );
}

#[test]
fn glyphs_outside_a_span() {
    assert_eq!(
        c("d ≈ 0.42, Δ ≤ 5% and r ~ 0.3"),
        "d about 0.42, delta at most 5 percent and r about 0.3"
    );
    assert_eq!(c("The α-β trade-off"), "The alpha beta trade off");
}

// --------------------------------------------------------------------------- citations

#[test]
fn a_semicolon_inside_a_citation_is_not_a_full_stop() {
    assert_eq!(
        c("(Smith et al., 2021; Zhou & Lee, 2019) reports"),
        "(Smith and colleagues, 2021, Zhou and Lee, 2019) reports"
    );
}

#[test]
fn a_semicolon_between_clauses_still_splits() {
    assert_eq!(
        c("queue age; webhook retries"),
        "queue age. Webhook retries"
    );
}

#[test]
fn a_url_is_read_as_its_host() {
    assert_eq!(
        c("Data at https://doi.org/10.1234/abcd.5678 (accessed)"),
        "Data at doi.org (accessed)"
    );
}

/// A bare DOI made the AR loop babble on a real paper's abstract: the engine reported
/// "segment 11 runs long — 261 frames for 173 chars" and the audio there was nonsense.
#[test]
fn a_bare_doi_is_not_read_as_its_digits() {
    assert_eq!(
        c("PVLDB, 19(2): 224 - 237, 2025. doi:10.14778/3773749.3773760 PVLDB Artifact"),
        "PVLDB, 19(2): 224 - 237, 2025. DOI PVLDB Artifact"
    );
    assert_eq!(c("DOI:10.1234/abcd.5678 next"), "DOI next");
    // Not a DOI: an ordinary word starting "doi" must survive.
    assert_eq!(c("doing 10.5 of them"), "doing 10.5 of them");
}

#[test]
fn footnotes_are_narrated_where_they_are_defined() {
    assert_eq!(
        doc("A claim.[^1]\n\n[^1]: The replication package.\n"),
        "A claim.\n\nFootnote 1. The replication package.\n"
    );
}

// ------------------------------------------------------------------------- regressions

#[test]
fn prose_is_untouched() {
    assert_eq!(
        c("An ordinary sentence, with a comma."),
        "An ordinary sentence, with a comma."
    );
}

#[test]
fn identifiers_and_code_spans() {
    assert_eq!(
        c("`PaymentCaptured(provider_transaction_id)` fires"),
        "Payment captured, provider transaction id fires"
    );
    assert_eq!(c("an NP-15 price in CAISO"), "an NP-15 price in CAISO");
}

#[test]
fn emphasis_and_dashes() {
    assert_eq!(
        c("**bold** and *italic* — a pause"),
        "bold and italic, a pause"
    );
}

// ------------------------------------------------- block rules the Python covers only
// ------------------------------------------------- through `convert`

#[test]
fn a_table_becomes_one_sentence_per_row() {
    let out = doc("| Partner | Provides |\n|---|---|\n| Product | field synthesis |\n");
    assert_eq!(out, "Product. Provides: field synthesis.\n");
}

#[test]
fn a_heading_gets_its_own_paragraph() {
    // The paragraph gap is 320 ms against a sentence's 90 ms, and that gap is the section
    // break a listener hears.
    assert_eq!(doc("# Title\n\nProse.\n"), "Title.\n\nProse.\n");
}

#[test]
fn front_matter_is_dropped() {
    assert_eq!(doc("+++\ntitle = \"x\"\n+++\n\nBody.\n"), "Body.\n");
    assert_eq!(doc("---\ntitle: x\n---\n\nBody.\n"), "Body.\n");
}

#[test]
fn a_lowercase_opening_is_capitalised() {
    // "founder intervention recorded" was spoken "Sharpen intervention recorded",
    // reproducibly across two seeds.
    // No full stop: only headings, bullets, footnotes and table cells gain one. A plain
    // paragraph keeps the author's punctuation, or lack of it.
    assert_eq!(
        doc("founder_intervention recorded\n"),
        "Founder intervention recorded\n"
    );
}

#[test]
fn code_blocks_are_dropped_unless_asked_for() {
    assert_eq!(doc("```sh\nrm -rf /\n```\n\nProse.\n"), "Prose.\n");
    let kept = convert(
        "```sh\nls\n```\n\nProse.\n",
        &Options {
            keep_code: true,
            keep_captions: true,
        },
    );
    assert!(kept.contains("ls"), "{kept:?}");
}

#[test]
fn a_figure_contributes_its_caption() {
    assert_eq!(
        doc("{{< chapter-figure caption=\"The two paths\" >}}\n\nProse.\n"),
        "The two paths.\n\nProse.\n"
    );
    let dropped = convert(
        "{{< chapter-figure caption=\"The two paths\" >}}\n\nProse.\n",
        &Options {
            keep_code: false,
            keep_captions: false,
        },
    );
    assert_eq!(dropped, "Prose.\n");
}
