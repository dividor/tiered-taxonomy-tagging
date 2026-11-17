"""Tests for rolling chunks up and for the trained classifier. Nothing is sent or paid for."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from method_fakes import Scripted
from tiered_taxonomy.methods.base import Prediction
from tiered_taxonomy.methods.llm_methods import (
    rate_labels,
)
from tiered_taxonomy.methods.rollup import (
    any_chunk,
    certainty_weighted,
    evenly_spaced,
    log_odds,
    share_of_chunks,
)
from tiered_taxonomy.methods.trained import Cut, TrainedClassifier, choose_cut, choose_top_k
from tiered_taxonomy.taxonomy import from_paths


def test_a_cut_and_a_label_count_drop_a_parent_kept_with_its_own_child() -> None:
    ranking = {"P": 0.9, "P->x": 0.8, "Q": 0.7}
    assert Cut(3, 0.5).apply(ranking) == {"P->x", "Q"}
    assert Cut(1, 0.5).apply(ranking) == {"P"}
    best, value = choose_top_k([{"P->x", "Q"}], [ranking], [2, 3])
    assert (best, value) == (3, 1.0)


def test_rollup_rules_differ_as_intended() -> None:
    chunks = [
        Prediction(labels={"common": 0.9, "rare": 0.9}),
        Prediction(labels={"common": 0.8, "doubtful": 0.2}),
        Prediction(labels={"common": 0.7}),
        Prediction(),
    ]
    assert any_chunk(chunks) == {"common", "rare", "doubtful"}
    assert share_of_chunks(0.5)(chunks) == {"common"}
    assert certainty_weighted(0.5)(chunks) == {"common", "rare"}
    assert any_chunk([]) == set()


def test_log_odds_stays_finite() -> None:
    assert log_odds(0.5) == 0
    assert np.isfinite(log_odds(0.0))
    assert np.isfinite(log_odds(1.0))


def test_evenly_spaced_keeps_order_and_limit() -> None:
    assert evenly_spaced(["a", "b"], 5) == ["a", "b"]
    assert evenly_spaced(list("abcdef"), 3) == ["a", "c", "e"]


def test_trained_classifier_learns_a_simple_pattern() -> None:
    texts = ["water pump well"] * 6 + ["market price trade"] * 6 + ["water pump market"] * 2
    gold = [{"wash"}] * 6 + [{"trade"}] * 6 + [{"wash", "trade"}] * 2
    model = TrainedClassifier(max_features=100, threads=1)
    model.fit(texts, gold)
    ranked = model.rank(["water pump well", "market price trade"], top_k=2)
    assert max(ranked[0], key=ranked[0].__getitem__) == "wash"
    assert max(ranked[1], key=ranked[1].__getitem__) == "trade"
    best_k, value = choose_top_k([{"wash"}, {"trade"}], ranked, [1, 2])
    assert best_k == 1
    assert value == 1.0


def test_rank_before_training_raises() -> None:
    with pytest.raises(RuntimeError, match="not been trained"):
        TrainedClassifier().rank(["x"])


def test_a_cut_applies_both_its_limit_and_threshold() -> None:
    ranking = {"a": 0.9, "b": 0.5, "c": 0.4}
    assert Cut(2, 0.0).apply(ranking) == {"a", "b"}
    assert Cut(3, 0.45).apply(ranking) == {"a", "b"}


def test_choose_cut_uses_a_threshold_when_a_limit_alone_cannot_win() -> None:
    gold = [{"a"}, {"a", "b"}]
    ranked = [{"a": 0.9, "b": 0.1}, {"a": 0.9, "b": 0.8}]
    cut, value = choose_cut(gold, ranked, [1, 2], [0.0, 0.5])
    assert cut == Cut(2, 0.5)
    assert value == 1.0


def test_rate_labels_reads_ratings_and_makes_no_call_with_nothing_to_rate() -> None:
    tree = from_paths(["Health->Water"])
    client: Any = Scripted('{"ratings": [{"id": "Health->Water", "score": 5}]}')
    rated = rate_labels("text", tree, ["Health->Water"], client)
    assert rated.labels == {"Health->Water": 0.5}
    assert rated.calls == 1
    assert rate_labels("text", tree, [], client).calls == 0
