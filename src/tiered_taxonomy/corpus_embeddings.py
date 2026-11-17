"""Embed whole training splits locally, since sending this many long texts through hosted inference is too
slow."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

SAVE_EVERY = 4000


class Encoder(Protocol):
    """Anything that turns texts into a matrix of normalised vectors, such as a SentenceTransformer."""

    def encode(
        self, texts: list[str], batch_size: int, normalize_embeddings: bool, show_progress_bar: bool
    ) -> np.ndarray:
        """Return one row per text."""
        ...


def encode_split(encoder: Encoder, texts: Sequence[str], path: Path, batch_size: int) -> np.ndarray:
    """Embed texts, saving progress in parts so that an interrupted run carries on where it stopped."""
    parts = path.parent / f"{path.stem}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved: np.ndarray = np.load(path)
        if len(saved) != len(texts):
            raise ValueError(f"{path} has {len(saved)} rows but there are {len(texts)} texts")
        return saved
    for start in range(0, len(texts), SAVE_EVERY):
        part = parts / f"{start:08d}.npy"
        if not part.exists():
            vectors = encoder.encode(
                list(texts[start : start + SAVE_EVERY]),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            np.save(part, vectors.astype(np.float32))
    joined = np.concatenate([np.load(part) for part in sorted(parts.glob("*.npy"))])
    np.save(path, joined)
    return joined


def nearest_label_scores(
    queries: np.ndarray, documents: np.ndarray, labels: Sequence[frozenset[str]], neighbours: int
) -> list[dict[str, float]]:
    """Give each query labels from its most similar labelled documents, weighted by how similar they are.

    This is label transfer: a document is likely to share labels with documents that resemble it. Scores are
    scaled so
    that a label on every neighbour scores 1.
    """
    results: list[dict[str, float]] = []
    for start in range(0, len(queries), 256):
        similarity = queries[start : start + 256] @ documents.T
        top = np.argpartition(-similarity, neighbours - 1, axis=1)[:, :neighbours]
        for row, picks in zip(similarity, top, strict=True):
            weights = np.clip(row[picks], 0, None)
            total = float(weights.sum()) or 1.0
            scores: dict[str, float] = {}
            for pick, weight in zip(picks, weights, strict=True):
                for label in labels[int(pick)]:
                    scores[label] = scores.get(label, 0.0) + float(weight) / total
            results.append(scores)
    return results
