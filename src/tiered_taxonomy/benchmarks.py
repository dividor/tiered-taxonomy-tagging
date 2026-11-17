"""Scoring on the measures published papers use, so that our results can be set beside theirs.

HumSet's papers report one F1 per category, macro-averaged over its labels, with a threshold per label chosen
on the validation split. MultiEURLEX's paper reports mean R-Precision. These are different questions from the
micro F1 used elsewhere in this repository, and only like can be compared with like.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from tiered_taxonomy.datasets import Item
from tiered_taxonomy.taxonomy import Taxonomy

Ranking = Mapping[str, float]

# The figures to beat, as printed. HumBERT's table does not say how it averages (arXiv 2305.16756, table 5).
HUMBERT_BEST = {
    "Sectors": 0.78,
    "Pillars 1D": 0.70,
    "Subpillars 1D": 0.54,
    "Pillars 2D": 0.71,
    "Subpillars 2D": 0.51,
}
HUMSET_PAPER_XLM_R = {
    "Sectors": 0.73,
    "Pillars 1D": 0.53,
    "Subpillars 1D": 0.38,
    "Pillars 2D": 0.63,
    "Subpillars 2D": 0.40,
}
# Mean R-Precision, English, chronological split (arXiv 2109.00904, tables 4 and 5).
MULTI_EURLEX_BERT = {"all": 48.4, "level_3": 67.7}
HUMSET_PILLARS_1D = frozenset(
    {
        "Casualties",
        "Context",
        "Covid-19",
        "Displacement",
        "Humanitarian Access",
        "Information And Communication",
        "Shock/Event",
    }
)
THRESHOLD_STEPS = 80


def humset_categories(taxonomy: Taxonomy) -> dict[str, frozenset[str]]:
    """Return the labels in each of HumSet's five published categories."""
    roots = {node for node, tier in taxonomy.tiers.items() if tier == 1}
    pillars = {node for node in roots if taxonomy.children.get(node)}
    if not HUMSET_PILLARS_1D <= pillars:
        raise ValueError(f"unknown pillars: {sorted(HUMSET_PILLARS_1D - pillars)}")
    subpillars = {node for node, tier in taxonomy.tiers.items() if tier == 2}
    in_one_d = {node for node in subpillars if node.split("->")[0] in HUMSET_PILLARS_1D}
    return {
        "Sectors": frozenset(roots - pillars),
        "Pillars 1D": HUMSET_PILLARS_1D,
        "Subpillars 1D": frozenset(in_one_d),
        "Pillars 2D": frozenset(pillars - HUMSET_PILLARS_1D),
        "Subpillars 2D": frozenset(subpillars - in_one_d),
    }


def with_parents(ranking: Ranking) -> dict[str, float]:
    """Add each `Parent->Child` label's parent, scored by its strongest child unless it has its own score."""
    expanded = dict(ranking)
    for label, value in ranking.items():
        if "->" in label:
            parent = label.split("->")[0]
            if parent not in ranking:
                expanded[parent] = max(expanded.get(parent, 0.0), value)
    return expanded


def gold_with_parents(item: Item) -> set[str]:
    """Return an item's labels at every tier."""
    return {label for labels in item.labels_by_tier.values() for label in labels}


def best_threshold(values: np.ndarray, truth: np.ndarray) -> float:
    """Return the score threshold with the best F1 for one label, among quantiles of its scores."""
    best, best_f1 = 1.0, -1.0
    for threshold in np.unique(np.quantile(values, np.linspace(0.3, 0.995, THRESHOLD_STEPS))):
        kept = values >= threshold
        total = kept.sum() + truth.sum()
        f1 = 2 * (kept & truth).sum() / total if total else 0.0
        if f1 > best_f1:
            best, best_f1 = float(threshold), float(f1)
    return best


def label_thresholds(
    gold: Sequence[set[str]], rankings: Sequence[Ranking], labels: frozenset[str]
) -> dict[str, float]:
    """Choose one threshold per label on the given items."""
    return {
        label: best_threshold(
            np.array([ranking.get(label, 0.0) for ranking in rankings]),
            np.array([label in truth for truth in gold]),
        )
        for label in labels
    }


def macro_f1(gold: Sequence[set[str]], predicted: Sequence[set[str]], labels: frozenset[str]) -> float:
    """Return F1 averaged equally over the given labels. A label with no true positives scores zero."""
    scores = []
    for label in labels:
        truth = np.array([label in item for item in gold])
        kept = np.array([label in item for item in predicted])
        total = truth.sum() + kept.sum()
        scores.append(2 * (truth & kept).sum() / total if total else 0.0)
    return float(np.mean(scores))


def other_averages(
    gold: Sequence[set[str]], predicted: Sequence[set[str]], labels: frozenset[str]
) -> dict[str, float]:
    """Return F1 averaged in each usual way, to test what a table that does not name its averaging might mean.
    Sample averaging covers items with a gold label among these labels. Labels no item has raise."""
    ordered = sorted(labels)
    truth = np.array([[label in item for label in ordered] for item in gold])
    if not truth.any():
        raise ValueError("no item has any of these labels, so their F1 cannot be averaged")
    kept = np.array([[label in item for label in ordered] for item in predicted])
    hits = (truth & kept).sum(axis=0)
    totals = truth.sum(axis=0) + kept.sum(axis=0)
    per_label = np.divide(2 * hits, totals, out=np.zeros(len(ordered)), where=totals > 0)
    support = truth.sum(axis=0)
    has_gold = truth.any(axis=1)
    per_item = 2 * (truth & kept).sum(axis=1)[has_gold] / (truth.sum(axis=1) + kept.sum(axis=1))[has_gold]
    return {
        "macro": float(per_label.mean()),
        "weighted": float((per_label * support).sum() / support.sum()),
        "micro": float(2 * hits.sum() / totals.sum()),
        "sample": float(per_item.mean()),
    }


def apply_thresholds(rankings: Sequence[Ranking], thresholds: Mapping[str, float]) -> list[set[str]]:
    """Keep each label that reaches its own threshold."""
    return [
        {label for label, limit in thresholds.items() if ranking.get(label, 0.0) >= limit}
        for ranking in rankings
    ]


def r_precisions(gold: Sequence[set[str]], rankings: Sequence[Ranking]) -> np.ndarray:
    """Return each document's precision at k, where k is its number of gold labels. Documents with no gold
    labels are left out."""
    values = []
    for truth, ranking in zip(gold, rankings, strict=True):
        if truth:
            top = sorted(ranking, key=lambda label: (-ranking[label], label))[: len(truth)]
            values.append(len(truth & set(top)) / len(truth))
    return np.array(values)


def mean_r_precision(gold: Sequence[set[str]], rankings: Sequence[Ranking]) -> float:
    """Return mean R-Precision as a percentage, as MultiEURLEX's paper prints it."""
    return 100 * float(r_precisions(gold, rankings).mean())


def rank_at_tier(ranking: Ranking, taxonomy: Taxonomy, tier: int) -> dict[str, float]:
    """Roll a ranking up to one tier, scoring each ancestor by its strongest descendant."""
    rolled: dict[str, float] = {}
    for label, value in ranking.items():
        for ancestor in taxonomy.roll_up([label], tier):
            rolled[ancestor] = max(rolled.get(ancestor, 0.0), value)
    return rolled


def bootstrap_interval(
    score: Callable[[np.ndarray], float], items: int, seed: int, draws: int = 200
) -> tuple[float, float]:
    """Return a 95% percentile interval, resampling items. `score` receives the positions of one resample."""
    rng = np.random.default_rng(seed)
    values = [score(rng.integers(0, items, items)) for _ in range(draws)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def law_year(celex_id: str) -> int:
    """Return the year in a CELEX identifier such as 32012R0782."""
    match = re.match(r"^\d(\d{4})", celex_id)
    if match is None:
        raise ValueError(f"no year in {celex_id!r}")
    return int(match.group(1))


def split_by_date(items: Sequence[Item]) -> tuple[list[Item], list[Item]]:
    """Divide items by date into an earlier half and a later half, ties broken by identifier."""
    ordered = sorted(items, key=lambda item: (law_year(item.id), item.id))
    middle = len(ordered) // 2
    return ordered[:middle], ordered[middle:]
