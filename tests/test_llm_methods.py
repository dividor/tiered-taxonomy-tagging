"""Tests for the LLM pipeline, evidence checks and label cards. Nothing is sent or paid for."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from method_fakes import Scripted, reply, store, tree
from tiered_taxonomy.methods.base import Prediction
from tiered_taxonomy.methods.llm_methods import (
    evidence_check,
    keyword_candidates,
    make_label_cards,
    parse_cards,
    parse_checks,
    parse_keywords,
    parse_ratings,
    pipeline_predict,
    quote_filter,
)
from tiered_taxonomy.retrieval import (
    LabelIndex,
)
from tiered_taxonomy.taxonomy import from_paths


def test_parse_keywords_and_ratings_handle_malformed_replies() -> None:
    assert parse_keywords(reply('{"keywords": ["a", " "]}')) == ["a"]
    assert parse_keywords(reply("nope")) == []
    allowed = {"a": 1.0, "b": 1.0}
    assert parse_ratings(
        reply('{"ratings": [{"id": "a", "score": 7}, {"id": "z", "score": 9}]}'), allowed
    ) == {
        "a": 0.7,
        "b": 0.0,
    }
    assert parse_ratings(reply("nope"), allowed) == {"a": 0.0, "b": 0.0}


def test_keyword_candidates_scale_to_one(tmp_path: Path) -> None:
    embeddings, _ = store(tmp_path)
    index = LabelIndex(tree(), embeddings)
    candidates = keyword_candidates(["water", "health water"], index, embeddings, 2)
    assert max(candidates.values()) == pytest.approx(1.0)
    assert keyword_candidates([], index, embeddings, 2) == {}


def test_pipeline_combines_ensemble_and_relevance(tmp_path: Path) -> None:
    embeddings, _ = store(tmp_path)
    index = LabelIndex(tree(), embeddings)
    generator: Any = Scripted('{"keywords": ["water"]}')
    rater: Any = Scripted('{"ratings": [{"id": "Health->Water", "score": 10}]}')
    result = pipeline_predict("water", index, embeddings, [generator], rater, 1, (0.3, 0.7), 0.5)
    assert result.labels == {"Health->Water": pytest.approx(1.0)}
    assert result.calls == 2


def test_pipeline_with_no_keywords_gives_no_labels_and_skips_the_rater(tmp_path: Path) -> None:
    embeddings, _ = store(tmp_path)
    index = LabelIndex(tree(), embeddings)
    generator: Any = Scripted("nonsense")
    rater: Any = Scripted()
    result = pipeline_predict("water", index, embeddings, [generator], rater, 1, (0.3, 0.7), 0.5)
    assert result.labels == {}
    assert result.malformed == 1
    assert rater.asked == []


def test_parse_checks_needs_both_a_yes_and_a_real_quote() -> None:
    content = (
        '{"checks": [{"id": "a", "supported": true, "quote": "real words"}, '
        '{"id": "b", "supported": true, "quote": "made up"}, '
        '{"id": "c", "supported": false, "quote": "real words"}]}'
    )
    assert parse_checks(reply(content), "some real words here", ["a", "b", "c", "d"]) == {
        "a": True,
        "b": False,
        "c": False,
        "d": False,
    }
    assert parse_checks(reply("nope"), "text", ["a"]) == {"a": False}


def test_evidence_check_keeps_supported_labels_and_counts_the_call() -> None:
    start = Prediction(labels={"Health": 0.9, "Context": 0.6}, calls=1)
    client: Any = Scripted('{"checks": [{"id": "Health", "supported": true, "quote": "clinic"}]}')
    checked = evidence_check("the clinic", start, tree(), client)
    assert set(checked.labels) == {"Health"}
    assert checked.calls == 2
    assert evidence_check("text", Prediction(calls=1), tree(), client).calls == 1


def test_quote_filter_drops_labels_with_invented_quotes() -> None:
    start = Prediction(labels={"a": 1.0, "b": 1.0}, quotes={"a": "the clinic", "b": "not there"})
    assert set(quote_filter("at the clinic", start).labels) == {"a"}


def test_label_cards_keep_only_asked_labels_and_count_what_is_missing() -> None:
    assert parse_cards(
        reply('{"cards": [{"id": "a", "definition": " D. "}, {"id": "z", "definition": "X"}]}'), ["a"]
    ) == {"a": "D."}
    assert parse_cards(reply("nope"), ["a"]) == {}
    client: Any = Scripted('{"cards": [{"id": "Health", "definition": "Care."}]}')
    cards, missing = make_label_cards(from_paths(["Health", "Water"]), client, batch_size=5, workers=1)
    assert cards == {"Health": "Care."}
    assert missing == 1


def test_label_cards_retry_failed_labels_in_smaller_batches() -> None:
    client: Any = Scripted(
        "nonsense",
        '{"cards": [{"id": "A", "definition": "a"}]}',
        '{"cards": [{"id": "B", "definition": "b"}]}',
    )
    cards, missing = make_label_cards(from_paths(["A", "B"]), client, batch_size=4, workers=1)
    assert cards == {"A": "a", "B": "b"}
    assert missing == 0
    assert len(client.asked) == 3
