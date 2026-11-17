"""Embeddings, saved so they are computed once, and nearest-label search over a taxonomy."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.embeddings import Embeddings

from tiered_taxonomy.taxonomy import Taxonomy

_SCHEMA = "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)"
EMBED_ATTEMPTS = 6


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length one, so a dot product is a cosine similarity."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(lengths == 0, 1, lengths)


class EmbeddingStore:
    """Embeds texts through a LangChain model, keeping vectors in a file in the data folder."""

    def __init__(
        self, model: Embeddings, model_name: str, path: Path, batch_size: int = 32, workers: int = 4
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._name = model_name
        self._batch_size = batch_size
        self._workers = workers
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute(_SCHEMA)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._name}\n{text}".encode()).hexdigest()

    def _load(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):
            part = keys[start : start + 500]
            with self._lock:
                for key in part:
                    row = self._connection.execute(
                        "SELECT vector FROM vectors WHERE key = ?", (key,)
                    ).fetchone()
                    if row is not None:
                        found[key] = np.frombuffer(row[0], dtype=np.float32)
        return found

    def _embed_batch(self, texts: Sequence[str]) -> None:
        vectors = np.asarray(self._embed_with_retries(texts), dtype=np.float32)
        rows = [(self._key(text), vector.tobytes()) for text, vector in zip(texts, vectors, strict=True)]
        with self._lock, self._connection:
            self._connection.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", rows)

    def _embed_with_retries(self, texts: Sequence[str]) -> list[list[float]]:
        for attempt in range(EMBED_ATTEMPTS):
            try:
                return self._model.embed_documents(list(texts))
            except (HfHubHTTPError, ConnectionError, TimeoutError):
                if attempt == EMBED_ATTEMPTS - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("embedding retries ran out")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalised row per text, computing only the texts not seen before."""
        keys = [self._key(text) for text in texts]
        known = self._load(sorted(set(keys)))
        missing = sorted({text for text, key in zip(texts, keys, strict=True) if key not in known})
        batches = [
            missing[start : start + self._batch_size] for start in range(0, len(missing), self._batch_size)
        ]
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            list(pool.map(self._embed_batch, batches))
        if missing:
            known = self._load(sorted(set(keys)))
        return normalise(np.stack([known[key] for key in keys]))


def label_text(taxonomy: Taxonomy, node_id: str) -> str:
    """Return the text a label is searched by: its name and first parent, plus its card's definition if any.

    Name and parent found the most correct labels on the development split, in exploratory work before
    this repository.
    """
    node = taxonomy.nodes[node_id]
    named = f"{node.name} ({taxonomy.nodes[node.parents[0]].name})" if node.parents else node.name
    return f"{named}. {node.definition}" if node.definition else named


class LabelIndex:
    """Nearest-label search over all the nodes of a taxonomy."""

    def __init__(self, taxonomy: Taxonomy, store: EmbeddingStore) -> None:
        self.taxonomy = taxonomy
        self.ids = sorted(taxonomy.nodes)
        self.vectors = store.embed([label_text(taxonomy, node_id) for node_id in self.ids])

    def search(self, query: np.ndarray, k: int) -> dict[str, float]:
        """Return the k nearest labels to one query vector, with their similarities."""
        scores = self.vectors @ query
        top = np.argsort(-scores)[:k]
        return {self.ids[i]: float(scores[i]) for i in top}

    def search_many(self, queries: np.ndarray, k: int) -> list[dict[str, float]]:
        """Return the k nearest labels for each row of a matrix of queries."""
        return [self.search(query, k) for query in queries]


def merge_views(hits_per_view: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Merge search results across views of a text, rewarding labels that several views found.

    A label's score is its best similarity times (1 + 0.1 ln h), where h is the number of views that found
    it. This
    is the merging rule in MATCH (arXiv 2610.25948).
    """
    best: dict[str, float] = {}
    hits: dict[str, int] = {}
    for view in hits_per_view:
        for label, value in view.items():
            best[label] = max(best.get(label, value), value)
            hits[label] = hits.get(label, 0) + 1
    return {label: value * (1 + 0.1 * math.log(hits[label])) for label, value in best.items()}


def prune_by_tier(
    ranked: Mapping[str, float], taxonomy: Taxonomy, tier: int, keep_share: float
) -> dict[str, float]:
    """Drop candidates under the weakest ancestors at a tier, with no model involved.

    Ancestors are ranked by the summed score of the candidates under them, and only the top share are kept.
    Candidates
    above that tier have no ancestor there and are always kept.
    """
    totals: dict[str, float] = {}
    for label, value in ranked.items():
        for ancestor in taxonomy.roll_up([label], tier):
            totals[ancestor] = totals.get(ancestor, 0.0) + value
    if not totals:
        return dict(ranked)
    ordered = sorted(totals, key=lambda key: (-totals[key], key))
    kept = set(ordered[: max(1, math.ceil(len(ordered) * keep_share))])
    pruned = {}
    for label, value in ranked.items():
        ancestors = taxonomy.roll_up([label], tier)
        if not ancestors or ancestors & kept:
            pruned[label] = value
    return pruned
