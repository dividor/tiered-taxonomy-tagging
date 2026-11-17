"""Score the HumBERT reproduction's test predictions per HumSet category, the paper's way and ours.

The reproduction (scripts/hf_jobs/reproduce_humbert.py) names labels as the paper's code does:
`first_level_tags->sectors->Health`, `first_level_tags->pillars_1d->Context`,
`subpillars_1d->Context->Environment`. Its code scores each label's F1 at the threshold chosen on
validation, then averages. For sectors and pillars the average is a plain mean over labels; for subpillars
it is a mean over pillars of each pillar's mean. This module recomputes both that and a plain mean over
every label in the category, as this repository's other HumSet scores use, so that the reproduction can be
checked against the trainer's own table and compared with ours.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

CATEGORIES = ("Sectors", "Pillars 1D", "Subpillars 1D", "Pillars 2D", "Subpillars 2D")
_PREFIXES = {
    "first_level_tags->sectors": "Sectors",
    "first_level_tags->pillars_1d": "Pillars 1D",
    "first_level_tags->pillars_2d": "Pillars 2D",
    "subpillars_1d": "Subpillars 1D",
    "subpillars_2d": "Subpillars 2D",
}
# The rows of the trainer's results table that give each category's score, and its "Avg".
TABLE_ROWS = {
    "Sectors": "mean->first_level_tags->sectors",
    "Pillars 1D": "mean->first_level_tags->pillars_1d",
    "Subpillars 1D": "mean->subpillars_1d",
    "Pillars 2D": "mean->first_level_tags->pillars_2d",
    "Subpillars 2D": "mean->subpillars_2d",
    "Avg": "mean->first_level_tags",
}


def category(tag: str) -> str:
    """Return the published category a label belongs to. Raises for a name the paper's code would not make."""
    for prefix, name in _PREFIXES.items():
        if tag.startswith(prefix + "->"):
            return name
    raise ValueError(f"not a HumSet label name: {tag}")


def label_f1(records: Iterable[Mapping[str, Sequence[str]]], labels: Iterable[str]) -> dict[str, float]:
    """Return each label's F1 over the records' gold and predicted labels; 0 when it has no positives."""
    hits: dict[str, int] = defaultdict(int)
    predicted: dict[str, int] = defaultdict(int)
    gold: dict[str, int] = defaultdict(int)
    for record in records:
        truth, guess = set(record["gold"]), set(record["predicted"])
        for tag in truth & guess:
            hits[tag] += 1
        for tag in guess:
            predicted[tag] += 1
        for tag in truth:
            gold[tag] += 1
    scores = {}
    for tag in labels:
        denominator = predicted[tag] + gold[tag]
        scores[tag] = 2 * hits[tag] / denominator if denominator else 0.0
    return scores


def category_scores(f1: Mapping[str, float], nested: bool) -> dict[str, float]:
    """Return each category's score and their mean. With `nested`, subpillars are averaged within each pillar
    first, as the paper's code does; otherwise every label counts once."""
    groups: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in CATEGORIES}
    for tag, value in f1.items():
        name = category(tag)
        parent = tag.split("->")[1] if nested and name.startswith("Subpillars") else ""
        groups[name][parent].append(value)
    scores = {}
    for name, by_parent in groups.items():
        if not by_parent:
            raise ValueError(f"no labels in {name}")
        scores[name] = float(np.mean([np.mean(values) for values in by_parent.values()]))
    scores["Mean"] = float(np.mean([scores[name] for name in CATEGORIES]))
    return scores


def table_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Return the trainer's own F1 for each category and its "Avg", from its results table. Raises if a row is
    missing."""
    by_tag = {str(row["tag"]): float(str(row["f_score"])) for row in rows}
    missing = [row for row in TABLE_ROWS.values() if row not in by_tag]
    if missing:
        raise ValueError(f"the results table lacks {missing}")
    return {name: by_tag[row] for name, row in TABLE_ROWS.items()}


def our_label(tag: str) -> str:
    """Return this repository's id for one of the paper's label names: a sector or pillar by its name, a
    subpillar as `pillar->subpillar`."""
    category(tag)
    parts = tag.split("->")
    return parts[-1] if tag.startswith("first_level_tags->") else "->".join(parts[1:])


def to_rankings(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """Return the reproduction's saved probabilities as rankings keyed by entry id, in this repository's label
    ids. HumSet 1.0.0 holds a few entries twice, row for row, so a repeat is kept once if its ranking is the
    same; a repeat with a different ranking raises."""
    rankings: dict[str, dict[str, float]] = {}
    for record in records:
        entry = str(record["entry_id"])
        ranking = record["ranking"]
        if not isinstance(ranking, Mapping):
            raise TypeError(f"entry {entry} has no ranking")
        renamed = {our_label(str(tag)): float(value) for tag, value in ranking.items()}
        if entry in rankings and rankings[entry] != renamed:
            raise ValueError(f"entry {entry} appears twice with different rankings")
        rankings[entry] = renamed
    return rankings
