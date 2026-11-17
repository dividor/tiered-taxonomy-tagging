"""Tests for the measures of why scores look low, on invented data."""

from __future__ import annotations

import numpy as np
import pytest

from tiered_taxonomy import diagnosis as dg
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.taxonomy import from_paths

TREE = from_paths(["A->x", "A->y", "B->z"])


def test_relation_finds_parents_siblings_and_strangers() -> None:
    assert dg.relation("A->x", {"A"}, TREE) == "parent or child"
    assert dg.relation("A->x", {"A->y"}, TREE) == "sibling"
    assert dg.relation("A->x", {"B->z"}, TREE) == "unrelated"


def test_error_shares_add_up_for_each_side() -> None:
    shares = dg.error_shares([{"A->x", "B->z"}], [{"A->y", "B->z", "A"}], TREE)
    assert shares["wrong"] == {"sibling": 0.5, "parent or child": 0.5}
    assert shares["missed"] == {"parent or child": 1.0}


def test_item_shares() -> None:
    assert dg.item_shares([{"a"}, {"a", "b"}, {"c"}], [{"a"}, {"a"}, {"d"}]) == {
        "all right": pytest.approx(1 / 3),
        "at least one right": pytest.approx(2 / 3),
    }


def test_repeated_text_agreement_ignores_case_and_spacing() -> None:
    items = [
        Item("1", "Same  text", "en", frozenset({"a", "b"})),
        Item("2", "same text", "en", frozenset({"a"})),
        Item("3", "other", "en", frozenset({"c"})),
    ]
    scores = dg.repeated_text_agreement(items)
    assert scores["pairs"] == 1.0
    assert scores["f1 between them"] == pytest.approx(2 / 3)
    assert scores["fully agreeing"] == 0.0
    with pytest.raises(ValueError, match="no identical"):
        dg.repeated_text_agreement(items[2:])


def test_nearest_text_agreement_groups_by_similarity() -> None:
    queries = np.array([[1.0, 0.0], [0.6, 0.8]])
    documents = np.array([[1.0, 0.0]])
    scores = dg.nearest_text_agreement(
        queries, [{"a"}, {"b"}], documents, [{"a"}], [(0.95, 1.01), (0.5, 0.95)]
    )
    assert scores["0.95 to 1.01"]["f1 between them"] == 1.0
    assert scores["0.50 to 0.95"]["f1 between them"] == 0.0
