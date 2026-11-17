"""Turning labels on chunks into labels on a whole document.

The rules compared are: any chunk carries the label, the label is on at least a share of chunks, and
certainty-weighted
log-odds, which sums the evidence from each chunk and is the rule TEF found best (arXiv 2609.27165).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence

from tiered_taxonomy.methods.base import Prediction

Rule = Callable[[Sequence[Prediction]], set[str]]
_EPSILON = 1e-3


def any_chunk(chunks: Sequence[Prediction]) -> set[str]:
    """Keep a label if any chunk has it."""
    return {label for chunk in chunks for label in chunk.labels}


def share_of_chunks(share: float) -> Rule:
    """Return a rule keeping a label if it is on at least this share of chunks, and always on at least one."""

    def rule(chunks: Sequence[Prediction]) -> set[str]:
        counts: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            for label in chunk.labels:
                counts[label] += 1
        needed = max(1, math.ceil(len(chunks) * share))
        return {label for label, count in counts.items() if count >= needed}

    return rule


def log_odds(probability: float) -> float:
    """Return the log-odds of a probability, staying finite at 0 and 1."""
    bounded = min(1 - _EPSILON, max(_EPSILON, probability))
    return math.log(bounded / (1 - bounded))


def certainty_weighted(floor: float) -> Rule:
    """Return a rule that sums log-odds over the chunks that have a label, keeping labels with a positive
    total.

    A chunk adds log-odds of its confidence, so confidence below `floor` counts against the label. `floor`
    shifts the
    point where a chunk's evidence turns from negative to positive.
    """
    shift = log_odds(floor)

    def rule(chunks: Sequence[Prediction]) -> set[str]:
        totals: dict[str, float] = defaultdict(float)
        for chunk in chunks:
            for label, confidence in chunk.labels.items():
                totals[label] += log_odds(confidence) - shift
        return {label for label, total in totals.items() if total > 0}

    return rule


def evenly_spaced(items: Sequence[str], limit: int) -> list[str]:
    """Return up to `limit` items spread evenly through a sequence, keeping their order."""
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[int(position * step)] for position in range(limit)]
