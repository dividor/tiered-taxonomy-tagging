"""Tests for retrieval, the funnel and choosing from candidates. Nothing is sent or paid for."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from method_fakes import Scripted, store, tree
from tiered_taxonomy.methods.llm_methods import (
    FunnelSettings,
    funnel_candidates,
    funnel_predict,
    hybrid_candidates,
    retrieval_only,
    select_from_all,
    select_from_candidates,
)
from tiered_taxonomy.prompts import describe_label
from tiered_taxonomy.retrieval import (
    LabelIndex,
    label_text,
    merge_views,
    normalise,
    prune_by_tier,
)
from tiered_taxonomy.taxonomy import Node, Taxonomy, with_definitions


def test_merge_views_rewards_labels_found_by_several_views() -> None:
    merged = merge_views([{"a": 0.5, "b": 0.5}, {"a": 0.4}])
    assert merged["a"] == pytest.approx(0.5 * (1 + 0.1 * np.log(2)))
    assert merged["b"] == 0.5


def test_prune_by_tier_drops_the_weakest_branch_and_keeps_shallow_labels() -> None:
    ranked = {"Context->Economy": 0.9, "Health->Water": 0.1, "Context": 0.2}
    pruned = prune_by_tier(ranked, tree(), tier=2, keep_share=0.5)
    assert "Context->Economy" in pruned
    assert "Health->Water" not in pruned
    assert "Context" in pruned


def test_embedding_store_only_embeds_new_texts(tmp_path: Path) -> None:
    embeddings, model = store(tmp_path)
    first = embeddings.embed(["economy", "health", "economy"])
    assert model.embedded == 2
    assert first.shape == (3, 4)
    assert np.allclose(first[0], first[2])
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    embeddings.embed(["economy"])
    assert model.embedded == 2


def test_normalise_leaves_zero_rows_alone() -> None:
    assert normalise(np.zeros((1, 2))).tolist() == [[0.0, 0.0]]


def test_label_index_finds_the_nearest_label(tmp_path: Path) -> None:
    embeddings, _ = store(tmp_path)
    index = LabelIndex(tree(), embeddings)
    assert next(iter(index.search(embeddings.embed(["water water"])[0], 1))) == "Health->Water"
    assert len(retrieval_only(embeddings.embed(["water"]), index, 2)[0]) == 2


def test_label_text_includes_a_definition_when_there_is_one() -> None:
    carded = with_definitions(tree(), {"Health": "Care."})
    assert label_text(carded, "Health") == "Health. Care."
    assert label_text(tree(), "Health") == "Health"
    assert label_text(tree(), "Health->Water") == "Water (Health)"
    assert describe_label(carded, "Health") == "Health | Care."
    assert describe_label(tree(), "Health->Water") == "Health->Water"
    numbered = Taxonomy({"10": Node("10", "Fruit")})
    assert describe_label(numbered, "10") == "10 | Fruit"


def test_funnel_only_offers_candidates_it_retrieved(tmp_path: Path) -> None:
    embeddings, _ = store(tmp_path)
    index = LabelIndex(tree(), embeddings)
    settings = FunnelSettings(retrieve_per_view=2, keep_candidates=2, prune_keep_share=1.0)
    candidates = funnel_candidates("water water", index, embeddings, settings)
    assert len(candidates) <= 2
    client: Any = Scripted(
        '{"labels": [{"id": "Health->Water", "confidence": 0.8, "quote": "water"}, {"id": "Nope"}]}'
    )
    result = funnel_predict("water water", index, embeddings, client, settings)
    assert "Health->Water" in candidates
    assert result.labels == {"Health->Water": 0.8}
    assert result.invalid == 1


def test_funnel_with_a_title_view_searches_twice(tmp_path: Path) -> None:
    embeddings, model = store(tmp_path)
    index = LabelIndex(tree(), embeddings)
    before = model.embedded
    funnel_candidates("water and more water", index, embeddings, FunnelSettings(title_words=1))
    assert model.embedded == before + 2


def test_hybrid_candidates_join_both_sources() -> None:
    assert hybrid_candidates({"a": 2.0, "b": 1.0}, {"c": 1.0, "a": 0.5}, 1) == ["a", "c"]


def test_select_from_candidates_only_accepts_what_was_offered() -> None:
    client: Any = Scripted('{"labels": [{"id": "Health"}, {"id": "Context"}]}')
    result = select_from_candidates("text", tree(), ["Health"], client)
    assert set(result.labels) == {"Health"}
    assert result.invalid == 1


def test_select_from_all_lists_every_label() -> None:
    client: Any = Scripted('{"labels": [{"id": "Health"}]}')
    labels = sorted(tree().nodes)
    assert set(select_from_all("text", tree(), labels, client).labels) == {"Health"}
    assert all(label in client.asked[0] for label in labels)
