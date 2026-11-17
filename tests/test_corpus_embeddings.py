"""Tests for embedding whole splits and transferring labels from similar documents."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tiered_taxonomy.corpus_embeddings import encode_split, nearest_label_scores
from tiered_taxonomy.methods.embedding_classifier import (
    ClassifierSettings,
    EmbeddingClassifier,
    combine_rankings,
)


class CountingEncoder:
    """Embeds a text by its length, and counts the texts it was asked about."""

    def __init__(self) -> None:
        self.seen = 0

    def encode(
        self, texts: list[str], batch_size: int, normalize_embeddings: bool, show_progress_bar: bool
    ) -> np.ndarray:
        self.seen += len(texts)
        return np.array([[float(len(text)), 1.0] for text in texts])


def test_encode_split_saves_and_does_not_repeat_work(tmp_path: Path) -> None:
    encoder = CountingEncoder()
    path = tmp_path / "train.npy"
    first = encode_split(encoder, ["a", "bb"], path, 8)
    assert first.shape == (2, 2)
    assert encoder.seen == 2
    assert np.array_equal(encode_split(encoder, ["a", "bb"], path, 8), first)
    assert encoder.seen == 2


def test_encode_split_refuses_a_file_that_does_not_match(tmp_path: Path) -> None:
    path = tmp_path / "train.npy"
    encode_split(CountingEncoder(), ["a"], path, 8)
    with pytest.raises(ValueError, match="rows"):
        encode_split(CountingEncoder(), ["a", "b"], path, 8)


def test_labels_come_from_the_most_similar_documents() -> None:
    documents = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = [frozenset({"water"}), frozenset({"water", "health"}), frozenset({"trade"})]
    scores = nearest_label_scores(np.array([[1.0, 0.0]]), documents, labels, neighbours=2)[0]
    assert scores["water"] == pytest.approx(1.0)
    assert 0 < scores["health"] < 1
    assert "trade" not in scores


def test_embedding_classifier_learns_a_simple_pattern() -> None:
    vectors = np.array([[1.0, 0.0]] * 20 + [[0.0, 1.0]] * 20)
    gold = [{"water"}] * 20 + [{"trade"}] * 20
    model = EmbeddingClassifier(ClassifierSettings(epochs=60, batch_size=8, learning_rate=0.05))
    losses = model.fit(vectors, gold)
    assert losses[-1] < losses[0]
    ranked = model.rank(np.array([[1.0, 0.0], [0.0, 1.0]]), top_k=1)
    assert list(ranked[0]) == ["water"]
    assert list(ranked[1]) == ["trade"]
    assert 0 < ranked[0]["water"] <= 1


def test_ranking_before_training_raises() -> None:
    with pytest.raises(RuntimeError, match="not been trained"):
        EmbeddingClassifier().rank(np.zeros((1, 2)))


def test_combine_rankings_weighs_each_method() -> None:
    combined = combine_rankings([{"a": 1.0}, {"a": 0.5, "b": 1.0}], [1.0, 0.5])
    assert combined == {"a": 1.25, "b": 0.5}
