"""Measures of why scores on these test sets look low: how strict the scoring is, how near the misses are,
and how consistently the same text was labelled."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from tiered_taxonomy.datasets import Item
from tiered_taxonomy.metrics import micro_f1
from tiered_taxonomy.taxonomy import Taxonomy


def relation(label: str, others: set[str], taxonomy: Taxonomy) -> str:
    """Say how a wrong or missed label relates to the labels on the other side of the comparison."""
    parents = set(taxonomy.nodes[label].parents)
    children = set(taxonomy.children.get(label, []))
    if others & (parents | children):
        return "parent or child"
    if any(other in taxonomy.nodes and parents & set(taxonomy.nodes[other].parents) for other in others):
        return "sibling"
    return "unrelated"


def error_shares(
    gold: Sequence[set[str]], predicted: Sequence[set[str]], taxonomy: Taxonomy
) -> dict[str, dict]:
    """Return the share of wrong and of missed labels that sit next to a right answer in the tree."""
    wrong: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    for truth, guess in zip(gold, predicted, strict=True):
        wrong.update(relation(label, truth, taxonomy) for label in guess - truth)
        missed.update(relation(label, guess, taxonomy) for label in truth - guess)
    return {
        name: {kind: count / sum(counts.values()) for kind, count in counts.items()}
        for name, counts in (("wrong", wrong), ("missed", missed))
    }


def item_shares(gold: Sequence[set[str]], predicted: Sequence[set[str]]) -> dict[str, float]:
    """Return the share of items with every label right, and with at least one right."""
    pairs = list(zip(gold, predicted, strict=True))
    return {
        "all right": float(np.mean([truth == guess for truth, guess in pairs])),
        "at least one right": float(np.mean([bool(truth & guess) for truth, guess in pairs])),
    }


def repeated_text_agreement(items: Iterable[Item]) -> dict[str, float]:
    """Compare the labels given to identical texts, taking the first two items with each text."""
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.labels:
            groups[" ".join(item.text.lower().split())].append(item)
    pairs = [(set(group[0].labels), set(group[1].labels)) for group in groups.values() if len(group) > 1]
    if not pairs:
        raise ValueError("no identical texts to compare")
    first, second = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    return {
        "pairs": float(len(pairs)),
        "f1 between them": micro_f1(first, second),
        "fully agreeing": float(np.mean([one == two for one, two in pairs])),
    }


def nearest_text_agreement(
    queries: np.ndarray,
    query_gold: Sequence[set[str]],
    documents: np.ndarray,
    document_gold: Sequence[set[str]],
    bands: Sequence[tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Compare each query's labels with those of its most similar document, by bands of similarity."""
    similarity = queries @ documents.T
    nearest = similarity.argmax(axis=1)
    closeness = similarity[np.arange(len(queries)), nearest]
    results = {}
    for low, high in bands:
        kept = [
            (query_gold[i], document_gold[int(nearest[i])])
            for i in range(len(queries))
            if low <= closeness[i] < high and query_gold[i] and document_gold[int(nearest[i])]
        ]
        if kept:
            results[f"{low:.2f} to {high:.2f}"] = {
                "pairs": float(len(kept)),
                "f1 between them": micro_f1([pair[0] for pair in kept], [pair[1] for pair in kept]),
            }
    return results
