"""A trained classifier on document embeddings: one logistic output per label, with no taxonomy involved.

This is the dense counterpart to TF-IDF with Omikuji. It sees meaning across wording and languages, but as
with any
classifier it can only predict labels it saw in training.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from tiered_taxonomy.metrics import LabelSets


@dataclass(frozen=True)
class ClassifierSettings:
    """How the classifier trains."""

    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 7
    device: str = "cpu"


class EmbeddingClassifier:
    """One logistic output per label, trained on fixed document embeddings."""

    def __init__(self, settings: ClassifierSettings | None = None) -> None:
        self.settings = settings or ClassifierSettings()
        self.labels: list[str] = []
        self._layer: nn.Linear | None = None

    def fit(self, vectors: np.ndarray, gold: LabelSets) -> list[float]:
        """Train on document vectors and their labels, returning the mean loss of each epoch."""
        settings = self.settings
        torch.manual_seed(settings.seed)
        self.labels = sorted({label for labels in gold for label in labels})
        positions = {label: i for i, label in enumerate(self.labels)}
        rows = [[positions[label] for label in labels] for labels in gold]
        layer = nn.Linear(vectors.shape[1], len(self.labels)).to(settings.device)
        optimiser = torch.optim.AdamW(
            layer.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        loss_function = nn.BCEWithLogitsLoss()
        generator = np.random.default_rng(settings.seed)
        inputs = torch.from_numpy(vectors.astype(np.float32))
        losses = []
        for _ in range(settings.epochs):
            order = generator.permutation(len(rows))
            total = 0.0
            batches = 0
            for start in range(0, len(order), settings.batch_size):
                picks = order[start : start + settings.batch_size]
                target = torch.zeros((len(picks), len(self.labels)))
                for row, pick in enumerate(picks):
                    target[row, rows[int(pick)]] = 1.0
                optimiser.zero_grad()
                output = layer(inputs[torch.from_numpy(picks)].to(settings.device))
                loss = loss_function(output, target.to(settings.device))
                loss.backward()
                optimiser.step()
                total += float(loss.item())
                batches += 1
            losses.append(total / max(1, batches))
        self._layer = layer
        return losses

    def rank(self, vectors: np.ndarray, top_k: int = 100) -> list[dict[str, float]]:
        """Return each document's highest-scoring labels, with scores from 0 to 1."""
        if self._layer is None:
            raise RuntimeError("the classifier has not been trained")
        results: list[dict[str, float]] = []
        limit = min(top_k, len(self.labels))
        with torch.no_grad():
            for start in range(0, len(vectors), 1024):
                batch = torch.from_numpy(vectors[start : start + 1024].astype(np.float32)).to(
                    self.settings.device
                )
                scores = torch.sigmoid(self._layer(batch))
                values, indices = torch.topk(scores, limit, dim=1)
                for row_values, row_indices in zip(values.tolist(), indices.tolist(), strict=True):
                    results.append(dict(zip((self.labels[i] for i in row_indices), row_values, strict=True)))
        return results


def combine_rankings(rankings: Sequence[dict[str, float]], weights: Sequence[float]) -> dict[str, float]:
    """Return a weighted sum of a document's label scores across methods."""
    combined: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for label, value in ranking.items():
            combined[label] = combined.get(label, 0.0) + weight * value
    return combined
