"""Tests for the README's results section, built from the same rows as the report page."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiered_taxonomy import report as rp
from tiered_taxonomy import report_markdown as md

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs() -> rp.Inputs:
    return rp.load_inputs(REPO)


def test_a_method_table_has_a_row_per_method_and_the_reports_columns(inputs: rp.Inputs) -> None:
    laws = rp.method_rows(inputs, "eurlex")
    lines = md.method_table(laws, "law").splitlines()
    assert len(lines) == 2 + len(laws)
    assert lines[0].startswith("| Method | Kind | Exact labels | Tier 1 | Tier 2 | Tier 3 |")
    assert "Macro F1" not in lines[0]
    first = lines[2]
    assert laws[0].name in first and f"| {laws[0].f1:.2f} |" in first


def test_the_humset_table_adds_both_macro_f1_columns_and_marks_absent_bands(inputs: rp.Inputs) -> None:
    rows = rp.method_rows(inputs, rp.HUMSET_V2)
    text = md.method_table(rows, "excerpt")
    header = text.splitlines()[0]
    assert header.index(rp.AVG_HEAD) < header.index("Exact labels") < header.index("| Macro F1 |")
    combined = next(row for row in rows if row.kind == "Combined")
    assert f"| {rp.tier_one_average(combined.paper or {}):.2f} | {combined.f1:.2f} |" in text
    assert "| none |" in text


def test_the_section_holds_the_benchmarks_and_both_tables(inputs: rp.Inputs) -> None:
    section = md.results_section(inputs)
    assert section.count("\n| --- |") == 4
    assert "| BERT, published |" in section and "| Five-group mean | 0.648 |" in section


def test_with_results_replaces_only_between_the_markers_and_needs_them_once() -> None:
    readme = f"top\n{md.START}\nold\n{md.END}\nbottom\n"
    assert md.with_results(readme, "new") == f"top\n{md.START}\n\nnew\n\n{md.END}\nbottom\n"
    for broken in ("no markers", f"{md.END} {md.START}", f"{md.START}{md.START}{md.END}"):
        with pytest.raises(ValueError, match="exactly one"):
            md.with_results(broken, "new")
    assert md.table(["a", "b"], [["x|y", "1"]], numeric_from=1) == "| a | b |\n| --- | ---: |\n| x\\|y | 1 |"


def test_the_committed_readme_holds_the_current_section(inputs: rp.Inputs) -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert md.with_results(readme, md.results_section(inputs)) == readme
