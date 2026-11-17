"""Tests for scoring, on small invented label sets."""

from __future__ import annotations

import pytest

from tiered_taxonomy.metrics import (
    bootstrap_interval,
    candidate_recall,
    macro_f1,
    micro_f1,
    paired_difference,
    score,
    score_by_tier,
    top_k,
)
from tiered_taxonomy.taxonomy import from_paths


def test_score_counts_hits_across_items() -> None:
    result = score([{"a", "b"}, {"c"}], [{"a"}, {"c", "d"}])
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.items == 2


def test_empty_predictions_score_zero_rather_than_failing() -> None:
    assert micro_f1([{"a"}], [set()]) == 0.0
    assert score([set()], [set()]).f1 == 0.0
    assert macro_f1([set()], [{"a"}]) == 0.0


def test_macro_f1_averages_over_gold_labels_only() -> None:
    assert macro_f1([{"a"}, {"b"}], [{"a"}, {"z"}]) == pytest.approx(0.5)


def test_score_by_tier_rolls_predictions_up() -> None:
    tree = from_paths(["P->x", "P->y", "Q->z"])
    results = score_by_tier({1: [{"P"}], 2: [{"P->x"}]}, [{"P->y"}], tree)
    assert results[1].f1 == 1.0
    assert results[2].f1 == 0.0


def test_top_k_breaks_ties_by_label() -> None:
    assert top_k({"b": 1.0, "a": 1.0, "c": 0.5}, 1) == {"a"}


def test_candidate_recall_is_the_share_of_gold_found() -> None:
    assert candidate_recall([{"a", "b"}, {"c"}], [{"a"}, {"c", "d"}]) == pytest.approx(2 / 3)


def test_bootstrap_interval_contains_the_score() -> None:
    gold = [{"a"}] * 8 + [{"b"}] * 2
    predicted = [{"a"}] * 8 + [{"z"}] * 2
    low, high = bootstrap_interval(micro_f1, gold, predicted, samples=200)
    assert low <= 0.8 <= high


def test_paired_difference_is_zero_for_identical_methods() -> None:
    gold = [{"a"}, {"b"}]
    assert paired_difference(micro_f1, gold, gold, gold, samples=50) == (0.0, 0.0, 0.0)
