"""Scores for multi-label predictions, tier by tier, with bootstrap intervals for paired comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from tiered_taxonomy.taxonomy import Taxonomy

LabelSets = Sequence[set[str]]


@dataclass(frozen=True)
class Scores:
    """Micro-averaged precision, recall and F1, macro F1 over labels in the gold set, and the item count."""

    precision: float
    recall: float
    f1: float
    macro_f1: float
    items: int


def _safe_ratio(top: float, bottom: float) -> float:
    return top / bottom if bottom else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_ratio(2 * precision * recall, precision + recall)


def micro_f1(gold: LabelSets, predicted: LabelSets) -> float:
    """Return micro-averaged F1."""
    hits = sum(len(g & p) for g, p in zip(gold, predicted, strict=True))
    return _f1(
        _safe_ratio(hits, sum(len(p) for p in predicted)),
        _safe_ratio(hits, sum(len(g) for g in gold)),
    )


def macro_f1(gold: LabelSets, predicted: LabelSets) -> float:
    """Return F1 averaged over the labels that occur in the gold set, so absent labels do not count."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for g, p in zip(gold, predicted, strict=True):
        for label in g & p:
            counts[label][0] += 1
        for label in p:
            counts[label][1] += 1
        for label in g:
            counts[label][2] += 1
    labels = [counts[label] for label in {label for g in gold for label in g}]
    if not labels:
        return 0.0
    scores = [_f1(_safe_ratio(hit, pred), _safe_ratio(hit, true)) for hit, pred, true in labels]
    return float(np.mean(scores))


def score(gold: LabelSets, predicted: LabelSets) -> Scores:
    """Return all the scores for one set of predictions."""
    hits = sum(len(g & p) for g, p in zip(gold, predicted, strict=True))
    precision = _safe_ratio(hits, sum(len(p) for p in predicted))
    recall = _safe_ratio(hits, sum(len(g) for g in gold))
    return Scores(precision, recall, _f1(precision, recall), macro_f1(gold, predicted), len(gold))


def score_by_tier(
    gold_by_tier: dict[int, LabelSets], predicted: LabelSets, taxonomy: Taxonomy
) -> dict[int, Scores]:
    """Roll predictions up to each tier and score them against the gold labels for that tier."""
    results: dict[int, Scores] = {}
    for tier, gold in gold_by_tier.items():
        rolled = [taxonomy.roll_up(labels, tier) for labels in predicted]
        results[tier] = score(gold, rolled)
    return results


def top_k(ranked: dict[str, float], k: int) -> set[str]:
    """Return the k highest-scoring labels, breaking ties by label so that runs repeat."""
    return {label for label, _ in sorted(ranked.items(), key=lambda pair: (-pair[1], pair[0]))[:k]}


def candidate_recall(gold: LabelSets, candidates: LabelSets) -> float:
    """Return the share of gold labels that appear among the candidates."""
    return _safe_ratio(
        sum(len(g & c) for g, c in zip(gold, candidates, strict=True)), sum(len(g) for g in gold)
    )


def bootstrap_interval(
    metric: Callable[[LabelSets, LabelSets], float],
    gold: LabelSets,
    predicted: LabelSets,
    samples: int = 1000,
    seed: int = 7,
) -> tuple[float, float]:
    """Return a 95% interval for a metric by resampling items with replacement."""
    rng = np.random.default_rng(seed)
    size = len(gold)
    values = []
    for _ in range(samples):
        picks = rng.integers(0, size, size)
        values.append(metric([gold[i] for i in picks], [predicted[i] for i in picks]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_difference(
    metric: Callable[[LabelSets, LabelSets], float],
    gold: LabelSets,
    first: LabelSets,
    second: LabelSets,
    samples: int = 1000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Return first minus second with a 95% interval, resampling the same items for both methods."""
    rng = np.random.default_rng(seed)
    size = len(gold)
    point = metric(gold, first) - metric(gold, second)
    values = []
    for _ in range(samples):
        picks = rng.integers(0, size, size)
        chosen_gold = [gold[i] for i in picks]
        values.append(
            metric(chosen_gold, [first[i] for i in picks]) - metric(chosen_gold, [second[i] for i in picks])
        )
    low, high = np.percentile(values, [2.5, 97.5])
    return float(point), float(low), float(high)
