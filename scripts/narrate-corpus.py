#!/usr/bin/env python3
"""Build the corpus `check-narrate.sh` compares the two implementations over.

Three sources, because each catches a different class of mistake:

  * **Every construct, one per case.** Hand-written from the rules themselves, so a rule that
    was mistranslated fails on its own line and says which one.
  * **The real chapter.** `prep-handbook/` is prose a person wrote, which contains the
    combinations nobody would think to write down.
  * **Randomised recombination.** Fragments joined at random, seeded so a failure reproduces.
    Most of the traps recorded in the Python are *interactions* — the currency rule consuming
    the number the magnitude rule wanted — and only recombination reaches those.

Written as JSON so both sides read exactly the same bytes.
"""
import json
import random
import sys
from pathlib import Path

# One per rule, in the order the rules run. Anything with a recorded failure in the Python's
# comments is here, because those are the cases known to have shipped wrong once.
LINES = [
    # prose that must not change
    "An ordinary sentence, with a comma.",
    "Latency rose to 12.4 seconds.", "Shipped in v1.2.3.", "an NP-15 price in CAISO",
    "SaaS and NoSQL and GraphQL survive intact.",
    # code spans
    "`PaymentCaptured(provider_transaction_id)` fires",
    "`seats SET remaining = remaining - 1` runs", "`WHERE id = ?` binds",
    "`OBSERVED_AT_IP`, `USED_DEVICE`, `REFERRED` are edges",
    "`product:{id}` is a key template", "`remaining > 0` is a precondition",
    "`idx ASC` and `other DESC`", "`B-tree` and `us-east-2a` and `customer-84`",
    "`SLO` and `API` and `ID` keep their capitals",
    # maths
    r"where $\alpha = 0.9$ and $\beta \in [0, 1]$",
    r"$\hat{\theta} = \arg\max_\theta \sum_i \log p(x_i)$",
    r"the loss $\mathcal{L}$ is $\frac{1}{N}\sum_{i=1}^{N} \ell_i$",
    r"$\sqrt{x^2 + y^2}$ and $x^3$ and $2^{10}$ and $10^{-3}$",
    r"$a \leq b \geq c \neq d \approx e \pm f$",
    r"$\prod_{k} \int_0^1 \bigcup_i \bigcap_j$",
    r"$\bar{x}$ and $\tilde{y}$ and $\vec{v}$ and $\hat p$",
    r"$\Delta \Sigma \Omega \Phi$ and $\mu s$",
    "d ≈ 0.42, Δ ≤ 5% and r ~ 0.3", "The α-β trade-off",
    "2.1 × 10^22 FLOPs", "a ± b and c ÷ d and e ≡ f and g ∝ h",
    "∑ over ∏ with ∫ and √ and ∂ and ∇", "x ∈ S but y ∉ S, ∀z ∃w",
    "A → B ⇒ C ↔ D ← E", "5‰ and 3″ and 2′ and a dagger†",
    # numbers
    "Temperatures of 20–25 degrees", "6-12 months of demand", "over FY2024–25",
    "fell to 1.5e-3", "about 10^-3", "95% CI [1.2, 3.4]",
    "p < 0.05", "N = 42 per arm", "Supply = Demand + Net Exports",
    "Generation = Consumption +", "from 2021-03-04 to then",
    "7B parameters", "a gain of 17.6 pp", "x86-64 and CosyVoice3-0.5B survive",
    "Qwen3.8 9B on a 16 GB M4", "v2.0.1 and p99 and T5 survive",
    "65% - 75% quality", "§5 and ¶3", "temperatures of 20 °C and 68 °F and a 90° turn",
    # units
    "Throughput was 3.2 GB/s", "60 %/yr", "every bus/node on the grid",
    "consume 300 MW of power", "plug in 1 MW for an hour",
    "a 4.4 GW natural gas plant", "measured in megawatts (MW) for most units",
    "50 dollars per MWh", "16 GB of RAM and 3 L of water and 500 ms of latency",
    "a 2.5 GHz core at 45 kHz and 12 Hz", "1 vCPU and 2 kWh and 3 TWh",
    # currency
    "Costs were ~$1.5M over the year", "anywhere from $10-150 per MWh",
    "like $100M+ machines", "credit $12.50 to revenue", "a $12 platform fee",
    "$4.76 million dollars of it", "credit $12 to platform fee revenue",
    "a $1,250,000 line item", "$0.42 per call",
    # abbreviations
    "energy (e.g. power), agriculture", "the DA market vs. the RT market",
    "cf. Fig. 2 and Table 1", "vol. 33, no. 2, pp. 2175–2183", "the Western vs.",
    "Dr. J. R. R. Tolkien, Ph.D.",
    "There's 10 MWh of demand at A. The model solves it.",
    "i.e. this, a.k.a. that, w.r.t. the other", "Smith et al. reported etc.",
    "resp. and approx. and ca. 1990", "Eq. 3, Sec. 2, Ch. 4, App. B, Ref. [7], Alg. 1",
    "M.Sc. and B.Sc. and U.S.A. and Prof. Mr. Mrs. Ms. St. Peter",
    "on Jan. 3 and Mar. 4 and Sept. 5 and Dec. 6",
    # prose punctuation and markup
    "**bold** and *italic* — a pause", "***bold italic*** survives",
    "_emphasis_ and agency_account and source_export",
    "queue age; webhook retries",
    "(Smith et al., 2021; Zhou & Lee, 2019) reports",
    'is vague; "complete through the previous UTC day" is not',
    "Data at https://doi.org/10.1234/abcd.5678 (accessed)",
    "see <https://example.org/path> for more",
    "PVLDB, 19(2): 224 - 237, 2025. doi:10.14778/3773749.3773760 PVLDB Artifact",
    "DOI:10.1234/abcd.5678 and doi: 10.1000/182 in a reference list",
    "a [link](https://example.com) and an ![image](x.png)",
    "a claim.[^1] and another[^note]",
    "It helps [specific customer] move from [painful current state]",
    "- [ ] unchecked and - [x] checked",
    "a <name> placeholder and a <br> tag and <em>markup</em>",
    "ATT&CK and AUTHZ and AUTHN and Zhou & Lee",
    "The API MUST authorize every read, and it MUST NOT trust the client.",
    "SBOM and PKCE and OIDC and SAML keep their letters",
    "we chose it because ___.", "Level 2+ and worker@8f31c2",
    "agency_account -> client -> source_export", "a => b and c → d",
    "I/O and pub/sub and read/write and and/or",
    "warehouse/lakehouse and a spaced / slash", "versioned amendment /",
    "B+tree in prose", "~~struck~~ and around ~5000 dollars and ~ 3",
    "Acc.: 88.9 in a table header", "timezone and signups and Timezone",
    "paid-but-unfulfilled orders and a trade-off",
    "an ellipsis… and “smart quotes” and ‘single’",
    "&amp; and &nbsp; and &#160; and &copy;",
]

# Whole documents, for `convert` and `page_text`: paragraph assembly, headings, tables,
# footnotes, front matter, shortcodes and display maths are block rules and none of them are
# reachable from a single line.
DOCUMENTS = [
    "Before.\n\n$$\n\\sum_{i=1}^{N} x_i\n$$\n\nAfter.\n",
    "A claim.[^1]\n\n[^1]: The replication package.\n",
    "+++\ntitle = \"x\"\n+++\n\nBody text.\n",
    "---\ntitle: x\n---\n\nBody text.\n",
    "# Heading One\n\nProse.\n\n## Heading Two\n\nMore prose.\n",
    "| Partner | Provides | Requires |\n|---|---|---|\n| Product | field synthesis | roadmap |\n"
    "| Sales | pipeline | quota |\n",
    "| Single |\n|---|\n",
    "| Header only |\n",
    "> A blockquote line.\n> A second line.\n",
    "- one item\n- two item\n\n1. first\n2. second\n",
    "---\n\nAfter a rule.\n",
    "***\n\nAfter an asterisk rule.\n",
    "{{< chapter-figure caption=\"A figure caption\" >}}\n\nProse after.\n",
    "{{% notice %}}\n\nProse after a shortcode with no caption.\n",
    "<!-- ILLUSTRATION-PLACEHOLDER\n  several lines of art direction\n-->\n\nProse.\n",
    "```sh\ncode block dropped\n```\n\nProse after code.\n",
    "\\begin{equation}\n  x = y\n\\end{equation}\n\nProse.\n",
    "founder_intervention recorded\n\nsecond paragraph\n",
    "A line\nwrapped across\nthree source lines.\n",
    "Text with a trailing operator = a +\nsecond line.\n",
]


def main() -> int:
    out = Path(sys.argv[1])
    lines = list(LINES)

    # The real chapter, as documents (whole) and as lines (each non-empty line).
    documents = list(DOCUMENTS)
    for path in sorted(Path("prep-handbook").glob("*.md")) if Path("prep-handbook").is_dir() else []:
        text = path.read_text()
        documents.append(text)
        lines.extend(ln for ln in text.splitlines() if ln.strip())

    # Randomised recombination. Seeded: a failure has to be reproducible, and an unseeded
    # generator turns a differential test into a flaky one.
    rng = random.Random(20260903)
    fragments = [ln for ln in LINES if len(ln) < 60]
    base = list(lines)
    for _ in range(4000):
        n = rng.randint(2, 5)
        lines.append(" ".join(rng.choice(fragments) for _ in range(n)))
    # Joined without a space too: adjacency with no separator is what exposed the currency
    # rule swallowing a maths span, and a space-joined corpus never reaches it.
    for _ in range(1000):
        lines.append("".join(rng.choice(fragments) for _ in range(rng.randint(2, 3))))
    # Line-level recombination of the *real* chapter's lines, which are longer and messier
    # than anything hand-written here.
    for _ in range(1000):
        lines.append(" ".join(rng.choice(base) for _ in range(rng.randint(2, 3))))
    for _ in range(400):
        n = rng.randint(2, 6)
        documents.append("\n\n".join(rng.choice(lines) for _ in range(n)) + "\n")
    # Documents built from block constructs, so table/heading/footnote interactions are hit.
    for _ in range(200):
        n = rng.randint(2, 4)
        documents.append("\n".join(rng.choice(DOCUMENTS) for _ in range(n)))

    (out / "lines.json").write_text(json.dumps(lines))
    (out / "documents.json").write_text(json.dumps(documents))
    print(f"corpus: {len(lines)} lines, {len(documents)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
