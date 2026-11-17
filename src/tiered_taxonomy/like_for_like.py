"""Score every technique on the same sample items with the same measures, so that results compare like
with like.

The LLM techniques were scored on fixed samples of test items: 600 laws and 1,500 excerpts. The trained models
and the combined method were scored on the full test sets with the benchmarks' own measures. Here every ranked
method gets a label limit and a score cut-off chosen on development items, as notebook 08 gave the trained
models, and is scored on the same sample with the same exact and tier-by-tier measures as the LLM techniques.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy import datasets as ds
from tiered_taxonomy import experiment as ex
from tiered_taxonomy import new_technique as nt
from tiered_taxonomy import signals as sg
from tiered_taxonomy.methods.trained import choose_cut
from tiered_taxonomy.taxonomy import Taxonomy

Ranking = Mapping[str, float]


def restrict(rankings: Sequence[Ranking], labels: Collection[str]) -> list[dict[str, float]]:
    """Keep only the given labels in each ranking. The HumSet combiner also ranks parent labels, which are
    never gold labels on their own, so it is cut and scored on the labels the gold can hold."""
    allowed = set(labels)
    return [{label: value for label, value in ranking.items() if label in allowed} for ranking in rankings]


def score_ranked(
    test_set: ex.TestSet,
    tune_items: Sequence[ds.Item],
    tune_rankings: Sequence[Ranking],
    sample: Sequence[ds.Item],
    sample_rankings: Sequence[Ranking],
    limits: Sequence[int],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Choose a label limit and cut-off on the tuning items, apply them to the sample and score it."""
    if len(tune_items) != len(tune_rankings) or len(sample) != len(sample_rankings):
        raise ValueError("every item needs exactly one ranking")
    cut, dev_f1 = choose_cut(ex.gold_exact(tune_items), tune_rankings, limits, thresholds)
    labels = [cut.apply(ranking) for ranking in sample_rankings]
    return {
        "limit": cut.limit,
        "threshold": cut.threshold,
        "dev_f1": round(dev_f1, 4),
        "scores": ex.evaluate(test_set, sample, labels),
        "labels": labels,
    }


def source_rankings(rankings: sg.Rankings, items: Sequence[ds.Item], name: str) -> list[Mapping[str, float]]:
    """Return a source's rankings in the items' order. A missing item raises."""
    nt.require_rankings({name: rankings}, items)
    return [rankings[item.id] for item in items]


def combined_rankings(
    stack: nt.Stack, sample: Sequence[ds.Item]
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Train the stack's combiner as the test runs did, and return its rankings of the cut items and the
    sample."""
    nt.require_rankings({name: stack.test_sources[name] for name in stack.names}, sample)
    made = nt.combine(
        stack.fit, stack.cut, stack.dev_sources, stack.names, stack.graph, stack.words, stack.settings
    )
    ranked = sg.apply_combiner(
        made["model"], sample, stack.test_sources, stack.names, stack.graph, stack.words, stack.settings
    )
    return made["tune_rankings"], ranked


def check_reproduces(name: str, recomputed: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    """Raise unless rescoring a method's saved labels gives the scores its notebook saved."""
    for key in ("f1", "precision", "recall"):
        if abs(recomputed["exact"][key] - saved["exact"][key]) > 1e-9:
            raise ValueError(
                f"{name}: rescored exact {key} {recomputed['exact'][key]:.6f} differs from the saved "
                f"{saved['exact'][key]:.6f}"
            )


def _with_mean(scores: Mapping[str, float]) -> dict[str, float]:
    return {**scores, "Mean": float(np.mean(list(scores.values())))}


def paper_scores(
    tune_items: Sequence[ds.Item],
    tune_rankings: Sequence[Ranking],
    sample: Sequence[ds.Item],
    sample_rankings: Sequence[Ranking],
    taxonomy: Taxonomy,
) -> dict[str, float]:
    """Return HumBERT's measure on the sample: macro F1 in each of HumSet's five published groups, with one
    threshold per label chosen on the tuning items, and the groups' mean. As on the full test set, gold and
    rankings cover every tier, and every label in a group counts: one that no sample item has scores zero."""
    if len(tune_items) != len(tune_rankings) or len(sample) != len(sample_rankings):
        raise ValueError("every item needs exactly one ranking")
    scores = nt.humset_score(
        [bm.gold_with_parents(item) for item in tune_items],
        [bm.with_parents(ranking) for ranking in tune_rankings],
        [bm.gold_with_parents(item) for item in sample],
        [bm.with_parents(ranking) for ranking in sample_rankings],
        taxonomy,
    )
    return _with_mean(scores)


def paper_scores_of_labels(
    sample: Sequence[ds.Item], predicted: Sequence[Collection[str]], taxonomy: Taxonomy
) -> dict[str, float]:
    """Return the same measure for a method that returns labels without scores, which leaves no threshold to
    choose: each label it returns counts, and so does that label's parent."""
    if len(sample) != len(predicted):
        raise ValueError("every item needs exactly one set of labels")
    gold = [bm.gold_with_parents(item) for item in sample]
    kept = [set(bm.with_parents(dict.fromkeys(labels, 1.0))) for labels in predicted]
    return _with_mean(
        {
            category: bm.macro_f1(gold, kept, labels)
            for category, labels in bm.humset_categories(taxonomy).items()
        }
    )
