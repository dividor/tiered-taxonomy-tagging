"""Tests for scoring every technique on the same items, on invented data."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy import experiment as ex
from tiered_taxonomy import like_for_like as lf
from tiered_taxonomy import new_technique as nt
from tiered_taxonomy import signals as sg
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.methods import combined as cb
from tiered_taxonomy.taxonomy import from_paths

TREE = from_paths(["P->x", "P->y", "Q->z"])


def item(item_id: str, labels: set[str]) -> Item:
    by_tier = {tier: frozenset(TREE.roll_up(labels, tier)) for tier in (1, 2)}
    return Item(item_id, "a b c", "en", frozenset(labels), item_id, by_tier)


def toy_set(test: list[Item]) -> ex.TestSet:
    return ex.TestSet("toy", TREE, (1, 2), [], [], test, Counter({"P->x": 50, "P->y": 50, "Q->z": 2}))


def test_restrict_keeps_only_the_allowed_labels() -> None:
    assert lf.restrict([{"P": 0.9, "P->x": 0.5}, {"Q": 0.2}], {"P->x"}) == [{"P->x": 0.5}, {}]


def test_score_ranked_chooses_the_cut_on_tuning_items_and_scores_the_sample() -> None:
    tune = [item("t1", {"P->x"}), item("t2", {"Q->z"})]
    tune_rankings = [{"P->x": 0.9, "P->y": 0.4}, {"Q->z": 0.8, "P->y": 0.3}]
    sample = [item("s1", {"P->x"}), item("s2", {"P->y"})]
    sample_rankings = [{"P->x": 0.95, "Q->z": 0.2}, {"P->x": 0.6, "P->y": 0.3}]
    result = lf.score_ranked(
        toy_set(sample), tune, tune_rankings, sample, sample_rankings, [1, 2], [0.0, 0.5]
    )
    assert (result["limit"], result["threshold"]) == (1, 0.0)
    assert result["dev_f1"] == 1.0
    assert result["labels"] == [{"P->x"}, {"P->x"}]
    assert result["scores"]["exact"]["f1"] == pytest.approx(0.5)
    assert result["scores"]["tiers"]["1"]["f1"] == pytest.approx(1.0)


def test_score_ranked_raises_when_rankings_and_items_differ_in_number() -> None:
    sample = [item("s1", {"P->x"})]
    with pytest.raises(ValueError, match="exactly one ranking"):
        lf.score_ranked(toy_set(sample), sample, [], sample, [{}], [1], [0.0])


def test_source_rankings_follow_item_order_and_raise_on_a_gap() -> None:
    items = [item("2", {"P->x"}), item("1", {"Q->z"})]
    assert lf.source_rankings({"1": {"a": 1.0}, "2": {"b": 1.0}}, items, "s") == [{"b": 1.0}, {"a": 1.0}]
    with pytest.raises(ValueError, match="no ranking for 1 of 2"):
        lf.source_rankings({"1": {"a": 1.0}}, items, "s")


def test_combined_rankings_rank_the_cut_items_and_the_sample() -> None:
    rng = np.random.default_rng(3)
    labels = [{"P->x"} if rng.random() < 0.5 else {"Q->z"} for _ in range(340)]
    items = [Item(str(i), "text", "en", frozenset(gold)) for i, gold in enumerate(labels)]
    everything = ["P->x", "P->y", "Q->z"]
    good = {i.id: {label: 0.9 if label in i.labels else 0.1 for label in everything} for i in items}
    sources: dict[str, sg.Rankings] = {"good": good}
    stack = nt.Stack(
        ("good",),
        items[:200],
        items[200:300],
        sources,
        sources,
        cb.LabelGraph(TREE, []),
        50,
        cb.CombinedSettings(max_iterations=20),
    )
    cut, sample = lf.combined_rankings(stack, items[300:])
    assert len(cut) == 100
    assert len(sample) == 40
    for gold, ranking in zip(items[300:], sample, strict=True):
        assert max(ranking, key=ranking.__getitem__) in gold.labels


def test_check_reproduces_raises_on_a_different_score() -> None:
    saved = {"exact": {"f1": 0.5, "precision": 0.5, "recall": 0.5}}
    lf.check_reproduces("m", saved, saved)
    with pytest.raises(ValueError, match="m: rescored exact f1"):
        lf.check_reproduces("m", {"exact": {"f1": 0.4, "precision": 0.5, "recall": 0.5}}, saved)


PILLAR_PATHS = [f"{pillar}->x" for pillar in sorted(bm.HUMSET_PILLARS_1D)]
HUMSET_TREE = from_paths([*PILLAR_PATHS, "Impact->y", "Health"])


def humset_item(item_id: str, labels: set[str]) -> Item:
    by_tier = {tier: frozenset(HUMSET_TREE.roll_up(labels, tier)) for tier in (1, 2)}
    return Item(item_id, "a b c", "en", frozenset(labels), item_id, by_tier)


def test_paper_scores_of_labels_add_parents_and_average_each_group() -> None:
    sample = [humset_item("1", {"Context->x", "Health"}), humset_item("2", {"Impact->y"})]
    predicted = [{"Context->x"}, {"Impact->y", "Health"}]
    scores = lf.paper_scores_of_labels(sample, predicted, HUMSET_TREE)
    # Health: one hit is missed and one is wrong, so F1 0. Impact and Impact->y: both right.
    assert scores["Sectors"] == 0.0
    assert scores["Pillars 2D"] == 1.0 and scores["Subpillars 2D"] == 1.0
    # Context, implied by Context->x, is right; the six other pillars have no items, so score zero.
    assert scores["Pillars 1D"] == pytest.approx(1 / 7)
    assert scores["Subpillars 1D"] == pytest.approx(1 / 7)
    assert scores["Mean"] == pytest.approx((0 + 1 / 7 + 1 / 7 + 1 + 1) / 5)


def test_paper_scores_choose_each_labels_threshold_on_the_tuning_items() -> None:
    tune = [humset_item("t1", {"Health"}), humset_item("t2", {"Impact->y"})]
    tune_rankings = [{"Health": 0.9, "Impact->y": 0.2}, {"Health": 0.3, "Impact->y": 0.8}]
    sample = [humset_item("s1", {"Health"}), humset_item("s2", {"Impact->y"})]
    sample_rankings = [{"Health": 0.95, "Impact->y": 0.1}, {"Health": 0.2, "Impact->y": 0.85}]
    scores = lf.paper_scores(tune, tune_rankings, sample, sample_rankings, HUMSET_TREE)
    assert scores["Sectors"] == 1.0
    assert scores["Subpillars 2D"] == 1.0 and scores["Pillars 2D"] == 1.0
    assert scores["Pillars 1D"] == 0.0


def test_paper_scores_raise_when_items_and_outputs_differ_in_number() -> None:
    sample = [humset_item("1", {"Health"})]
    with pytest.raises(ValueError, match="exactly one"):
        lf.paper_scores_of_labels(sample, [], HUMSET_TREE)
    with pytest.raises(ValueError, match="exactly one"):
        lf.paper_scores(sample, [], sample, [{}], HUMSET_TREE)
