"""The report's results tables as Markdown, for the README, built from the same rows as the report page so the
two always agree. `scripts/make_report.py` writes them between the README's results markers."""

from __future__ import annotations

from collections.abc import Sequence

from tiered_taxonomy.report import (
    AVG_HEAD,
    HUMSET_GROUPS,
    HUMSET_V2,
    RARE_BANDS,
    Inputs,
    Row,
    cost_text,
    humbert_rerun,
    method_rows,
    tier_one_average,
)
from tiered_taxonomy.report_page import benchmark_rows

START = "<!-- results:start -->"
END = "<!-- results:end -->"
# Where the interactive report is published.
REPORT_URL = "https://www.astrobagel.com/apps/taxonomy-tagger-results/"
RARE_HEADS = {"rare": "Rarest quarter recall", "unseen": "Never-seen recall"}


def table(header: Sequence[str], rows: Sequence[Sequence[str]], numeric_from: int) -> str:
    """Return a Markdown table; columns from `numeric_from` on are right-aligned."""
    align = ["---" if i < numeric_from else "---:" for i in range(len(header))]
    lines = [header, align, *rows]
    return "\n".join("| " + " | ".join(cell.replace("|", "\\|") for cell in line) + " |" for line in lines)


def _rarity(row: Row, band: str) -> str:
    rarity = row.scores["rarity"]
    return f"{rarity[band]['recall']:.2f}" if band in rarity else "none"


def method_table(rows: Sequence[Row], unit: str) -> str:
    """Return one test set's results table: every method, in the report's order (best by exact labels
    first), with the columns the report shows. The macro F1 columns appear only when the rows carry the
    HumBERT paper's measure."""
    tiers = sorted(rows[0].scores["tiers"], key=int)
    paper = all(row.paper is not None for row in rows)
    header = [
        "Method",
        "Kind",
        *([AVG_HEAD] if paper else []),
        "Exact labels",
        *(f"Tier {tier}" for tier in tiers),
        *(["Macro F1"] if paper else []),
        *(RARE_HEADS[band] for band in RARE_BANDS),
        f"Labels per {unit}",
        "Cost",
    ]
    body = []
    for row in rows:
        scores = row.paper or {}
        body.append(
            [
                row.name,
                row.kind,
                *([f"{tier_one_average(scores):.2f}"] if paper else []),
                f"{row.f1:.2f}",
                *(f"{row.tier(tier):.2f}" for tier in tiers),
                *([f"{scores['Mean']:.2f}"] if paper else []),
                *(_rarity(row, band) for band in RARE_BANDS),
                f"{row.scores['labels_per_item']:.1f}",
                cost_text(row)[0],
            ]
        )
    return table(header, body, numeric_from=2)


def benchmark_tables(inputs: Inputs) -> str:
    """Return the comparison with the published benchmarks: mean R-Precision on all 5,000 test laws, and
    macro F1 by label group on all of HumSet 2.0.0's test excerpts."""
    laws, _ = benchmark_rows(inputs)
    law_rows = [[name, sub, f"{score:.3f}"] for name, sub, score, _ in laws]
    rerun = humbert_rerun(inputs.results)
    printed, paper, combined = rerun["printed"], rerun["paper"], rerun["combined_v2"]
    intervals = rerun["combined_v2_intervals"]
    group_rows = [
        [
            group,
            f"{printed[group]:.2f}",
            f"{paper[group]:.3f}",
            f"{combined[group]:.3f}",
            f"{intervals[group][0]:.3f} to {intervals[group][1]:.3f}",
        ]
        for group in HUMSET_GROUPS
    ]
    group_rows.append(
        [
            "Five-group mean",
            f"{printed['Mean']:.3f}",
            f"{paper['Mean']:.3f}",
            f"{combined['Mean']:.3f}",
            f"{intervals['Mean'][0]:.3f} to {intervals['Mean'][1]:.3f}",
        ]
    )
    return (
        "**EU laws: mean R-Precision on all 5,000 test laws.** For each law, take as many of the top-ranked "
        "labels as it really has, and count the share that are correct.\n\n"
        + table(["Method", "Version", "Mean R-Precision"], law_rows, numeric_from=2)
        + "\n\n**HumSet 2.0.0: macro F1 by label group on all 12,896 test excerpts**, the measure the "
        "HumBERT paper reports. Its published figures have two decimals; their mean is worked out from "
        "them.\n\n"
        + table(
            ["Label group", "HumBERT, published", "HumBERT recipe, rerun", "Combined method", "95% interval"],
            group_rows,
            numeric_from=1,
        )
    )


def results_section(inputs: Inputs) -> str:
    """Return the README's results section, without its markers."""
    laws, excerpts = method_rows(inputs, "eurlex"), method_rows(inputs, HUMSET_V2)
    return "\n\n".join(
        [
            "Every number below comes from `outputs/results`, through the same code as the "
            f"[interactive report]({REPORT_URL}), which also has a card "
            "for each method. Scores run from 0 to 1, and higher is better. *Exact labels* gives credit only "
            "for the very label a person assigned; *Tier N* gives credit for landing in the right branch at "
            "that level. Recall columns are the share found of the correct labels that training used least, "
            "and that it never used. This section is written by `scripts/make_report.py`; do not edit it by "
            "hand.",
            "### Against the published benchmarks",
            benchmark_tables(inputs),
            "### EU laws (MultiEURLEX): 600 test laws, EuroVoc's top three tiers",
            method_table(laws, "law"),
            "### Humanitarian excerpts (HumSet 2.0.0): all 12,896 test excerpts",
            method_table(excerpts, "excerpt"),
        ]
    )


def with_results(readme: str, section: str) -> str:
    """Return the README with the text between its results markers replaced. Raises unless each marker
    appears exactly once, start before end."""
    if readme.count(START) != 1 or readme.count(END) != 1 or readme.index(START) > readme.index(END):
        raise ValueError(f"README needs exactly one {START} followed by one {END}")
    before, rest = readme.split(START)
    _, after = rest.split(END)
    return f"{before}{START}\n\n{section}\n\n{END}{after}"
