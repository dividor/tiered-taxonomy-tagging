"""Tests that the charts are written, and that a method keeps its colour."""

from __future__ import annotations

from pathlib import Path

from tiered_taxonomy import figures


def test_method_colours_are_fixed_and_unknown_methods_are_neutral() -> None:
    assert figures.method_colour("trained") == figures.METHOD_COLOURS["trained"]
    assert figures.method_colour("something new") == figures.NEUTRAL
    assert figures.method_name("something_new") == "something new"
    assert figures.method_name("trained") == "Trained classifier"


def test_spread_labels_keeps_them_apart_and_in_order() -> None:
    placed = figures._spread_labels([0.5, 0.51, 0.1], 0.05)
    assert placed[2] == 0.1
    assert placed[1] - placed[0] >= 0.05


def test_each_chart_writes_a_file(tmp_path: Path) -> None:
    tiers = {"Set": {"trained": {"1": 0.7, "2": 0.6}, "pipeline": {"1": 0.5, "2": 0.4}}}
    paths = [
        figures.tier_profile(tiers, "T", "S", tmp_path / "a.png"),
        figures.method_bars({"Set": {"trained": 0.5, "pipeline": 0.2}}, "F1", "T", "S", tmp_path / "b.png"),
        figures.recall_curves({"names": {"10": 0.1, "100": 0.3}}, "Candidates", "T", "S", tmp_path / "c.png"),
        figures.rarity_bars(
            {"trained": {"common": 0.5, "rare": 0.1, "unseen": 0.0}}, "T", "S", tmp_path / "d.png"
        ),
    ]
    assert all(path.stat().st_size > 0 for path in paths)


def test_category_bars_writes_a_file(tmp_path: Path) -> None:
    series = {
        "humbert_printed": {"Sectors": 0.78, "Pillars": 0.7},
        "classifier": {"Sectors": 0.7, "Pillars": 0.5},
    }
    path = figures.category_bars(series, "Macro F1", "Title", "Subtitle", tmp_path / "bars.png")
    assert path.stat().st_size > 0
