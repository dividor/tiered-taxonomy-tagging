"""Tests for scoring on the measures published papers use, on invented data."""

from __future__ import annotations

import numpy as np
import pytest

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.taxonomy import from_paths

TREE = from_paths(["Context->Economy", "Impact->Impact On People", "Agriculture"])


def test_categories_divide_pillars_subpillars_and_sectors() -> None:
    paths = [f"{pillar}->x" for pillar in bm.HUMSET_PILLARS_1D]
    categories = bm.humset_categories(from_paths([*paths, "Impact->y", "Agriculture"]))
    assert categories["Sectors"] == {"Agriculture"}
    assert categories["Pillars 2D"] == {"Impact"}
    assert categories["Subpillars 2D"] == {"Impact->y"}
    assert "Context->x" in categories["Subpillars 1D"]
    assert categories["Pillars 1D"] == bm.HUMSET_PILLARS_1D


def test_categories_refuse_a_tree_without_the_known_pillars() -> None:
    with pytest.raises(ValueError, match="unknown pillars"):
        bm.humset_categories(TREE)


def test_with_parents_uses_the_strongest_child_but_keeps_a_parents_own_score() -> None:
    assert bm.with_parents({"A->x": 0.2, "A->y": 0.5})["A"] == 0.5
    assert bm.with_parents({"A->x": 0.9, "A": 0.1})["A"] == 0.1


def test_gold_with_parents_covers_every_tier() -> None:
    item = Item(
        "1", "t", "en", frozenset({"A->x"}), labels_by_tier={1: frozenset({"A"}), 2: frozenset({"A->x"})}
    )
    assert bm.gold_with_parents(item) == {"A", "A->x"}


def test_best_threshold_separates_a_clean_label() -> None:
    values = np.array([0.1] * 80 + [0.9] * 20)
    truth = np.array([False] * 80 + [True] * 20)
    assert 0.1 < bm.best_threshold(values, truth) <= 0.9


def test_thresholds_apply_per_label() -> None:
    gold = [{"a"}] * 20 + [set()] * 80
    rankings = [{"a": 0.9, "b": 0.2}] * 20 + [{"a": 0.1, "b": 0.2}] * 80
    thresholds = bm.label_thresholds(gold, rankings, frozenset({"a"}))
    assert bm.apply_thresholds([{"a": 0.95}, {"a": 0.05}], thresholds) == [{"a"}, set()]


def test_macro_f1_weighs_labels_equally() -> None:
    gold = [{"a"}, {"a"}, {"b"}]
    predicted = [{"a"}, {"a"}, set()]
    assert bm.macro_f1(gold, predicted, frozenset({"a", "b"})) == 0.5


def test_mean_r_precision_counts_the_top_k_and_skips_empty_gold() -> None:
    gold = [{"a", "b"}, {"c"}, set()]
    rankings = [{"a": 0.9, "x": 0.8, "b": 0.1}, {"c": 0.5}, {"z": 1.0}]
    assert bm.mean_r_precision(gold, rankings) == pytest.approx(75.0)


def test_rank_at_tier_rolls_up_to_the_strongest_descendant() -> None:
    assert bm.rank_at_tier({"Context->Economy": 0.4}, TREE, 1) == {"Context": 0.4}


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = np.array([0.0, 1.0] * 200)
    low, high = bm.bootstrap_interval(lambda picks: float(values[picks].mean()), len(values), seed=1)
    assert low < 0.5 < high


def test_law_year_reads_a_celex_id_and_refuses_others() -> None:
    assert bm.law_year("32012R0782") == 2012
    with pytest.raises(ValueError, match="no year"):
        bm.law_year("abc")


def test_split_by_date_puts_older_laws_first() -> None:
    items = [Item(f"3{year}R0001", "t", "en", frozenset()) for year in (2014, 2010, 2012, 2011)]
    early, late = bm.split_by_date(items)
    assert [item.id for item in early] == ["32010R0001", "32011R0001"]
    assert [item.id for item in late] == ["32012R0001", "32014R0001"]


def test_other_averages_match_hand_worked_values() -> None:
    gold = [{"a"}, {"a"}, {"a"}, {"b"}]
    predicted = [{"a"}, {"a"}, set(), {"a"}]
    scores = bm.other_averages(gold, predicted, frozenset({"a", "b"}))
    assert scores["macro"] == pytest.approx((2 * 2 / 6) / 2)
    assert scores["weighted"] == pytest.approx((2 * 2 / 6) * 3 / 4)
    assert scores["micro"] == pytest.approx(2 * 2 / 7)
    assert scores["sample"] == pytest.approx(2 / 4)
    with pytest.raises(ValueError, match="no item has any"):
        bm.other_averages(gold, predicted, frozenset({"c"}))
