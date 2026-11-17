"""The trained flat classifier: TF-IDF features and Omikuji's label trees, which ignore the taxonomy
entirely."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import omikuji
from sklearn.feature_extraction.text import TfidfVectorizer

from tiered_taxonomy.metrics import LabelSets, micro_f1
from tiered_taxonomy.taxonomy import without_implied_parents


class TrainedClassifier:
    """A flat extreme multi-label classifier trained on labelled texts."""

    def __init__(self, max_features: int = 300_000, threads: int = 8) -> None:
        self._vectoriser = TfidfVectorizer(
            max_features=max_features, sublinear_tf=True, ngram_range=(1, 2), min_df=2
        )
        self._threads = threads
        self._labels: list[str] = []
        self._model: omikuji.Model | None = None

    def fit(self, texts: Sequence[str], labels: LabelSets) -> None:
        """Train on texts and their label sets. Items with no labels are skipped, as Omikuji needs at least
        one."""
        matrix = self._vectoriser.fit_transform(texts).tocsr()
        self._labels = sorted({label for item in labels for label in item})
        positions = {label: i for i, label in enumerate(self._labels)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.txt"
            with path.open("w", encoding="ascii") as handle:
                used = [i for i, item in enumerate(labels) if item]
                handle.write(f"{len(used)} {matrix.shape[1]} {len(self._labels)}\n")
                for i in used:
                    row = matrix.getrow(i)
                    label_part = ",".join(str(positions[label]) for label in sorted(labels[i]))
                    features = " ".join(f"{j}:{v:.5f}" for j, v in zip(row.indices, row.data, strict=True))
                    handle.write(f"{label_part} {features}\n")
            hyper = omikuji.Model.default_hyper_param()
            hyper.n_trees = 3
            self._model = omikuji.Model.train_on_data(str(path), hyper, n_threads=self._threads)

    def rank(self, texts: Sequence[str], top_k: int = 20) -> list[dict[str, float]]:
        """Return each text's highest-scoring labels."""
        if self._model is None:
            raise RuntimeError("the classifier has not been trained")
        matrix = self._vectoriser.transform(texts).tocsr()
        results = []
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i)
            pairs = list(zip(row.indices.tolist(), row.data.tolist(), strict=True))
            scored = self._model.predict(pairs, top_k=top_k) if pairs else []
            results.append({self._labels[position]: float(value) for position, value in scored})
        return results


def choose_top_k(
    gold: LabelSets, ranked: Sequence[dict[str, float]], choices: Sequence[int]
) -> tuple[int, float]:
    """Return the number of labels per item with the best micro F1, and that score. Use on a development
    split."""
    best_k, best_score = choices[0], -1.0
    for k in choices:
        predicted = [
            without_implied_parents(sorted(item, key=lambda key: (-item[key], key))[:k]) for item in ranked
        ]
        value = micro_f1(gold, predicted)
        if value > best_score:
            best_k, best_score = k, value
    return best_k, best_score


@dataclass(frozen=True)
class Cut:
    """How a ranking becomes a set of labels: at most `limit` labels, each scoring at least `threshold`."""

    limit: int
    threshold: float

    def apply(self, ranking: Mapping[str, float]) -> set[str]:
        """Return the labels this cut keeps from one ranking, less any whose own child it also keeps."""
        ordered = sorted(ranking, key=lambda key: (-ranking[key], key))[: self.limit]
        return without_implied_parents(label for label in ordered if ranking[label] >= self.threshold)


def choose_cut(
    gold: LabelSets,
    ranked: Sequence[Mapping[str, float]],
    limits: Sequence[int],
    thresholds: Sequence[float],
) -> tuple[Cut, float]:
    """Return the cut with the best micro F1, and that score. Use on a development split."""
    best = Cut(limits[0], thresholds[0])
    best_score = -1.0
    for limit in limits:
        for threshold in thresholds:
            cut = Cut(limit, threshold)
            value = micro_f1(gold, [cut.apply(ranking) for ranking in ranked])
            if value > best_score:
                best, best_score = cut, value
    return best, best_score
