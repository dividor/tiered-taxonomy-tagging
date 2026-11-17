"""The new technique: fine-tuned models, an LLM's suggestions and the local models, weighed by the combiner.

Everything here is chosen on development items, and notebook 10 scores the test sets once, after that choice.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy import datasets as ds
from tiered_taxonomy import experiment as ex
from tiered_taxonomy import gpu_jobs as gj
from tiered_taxonomy import signals as sg
from tiered_taxonomy.methods import combined as cb
from tiered_taxonomy.metrics import micro_f1
from tiered_taxonomy.taxonomy import Taxonomy, without_implied_parents

# A source stays only if removing it costs more than this on development items.
KEEP_MARGIN = {"humset": 0.001, "eurlex": 0.1}
# On MultiEURLEX these were trained on training laws alone, whichever stack uses them.
TRAIN_ONLY_SOURCES = frozenset({"xr_transformer", "ki_fsprompt"})
# These were only trained with the newer laws, so the stack without them leaves them out.
NEWER_ONLY_SOURCES = frozenset({"reranker", "llm", "gliner"})
# How far above the printed score counts as blowing it away: the largest gain found in the papers read.
WELL_AHEAD = {"humset": 0.05, "eurlex": 5.0}


def fetch_fine_tuned(
    root: Path,
    test_set: str,
    run: str,
    source: str,
    taxonomy: Taxonomy,
) -> set[str]:
    """Download one fine-tuning job's rankings, save them as a signal and return the labels dropped for having
    no place in the tree. A missing file raises."""
    dropped: set[str] = set()
    for split in ("dev", "test"):
        kept, outside = sg.drop_outside_tree(
            sg.read_rankings(gj.fetch(root, f"{run}_{split}.json")), taxonomy
        )
        dropped |= outside
        sg.save_signal(root, test_set, source, split, kept)
    return dropped


def fetch_history(root: Path, run: str) -> list[dict[str, Any]]:
    """Download one fine-tuning job's record of development scores. A missing file raises."""
    history: list[dict[str, Any]] = json.loads(
        gj.fetch(root, f"{run}_history.json").read_text(encoding="utf-8")
    )
    return history


def with_all_tiers(items: Sequence[ds.Item]) -> list[ds.Item]:
    """Return items whose labels include every tier, because published HumSet scores cover pillars as well."""
    return [replace(item, labels=frozenset(bm.gold_with_parents(item))) for item in items]


def expand_sources(sources: Mapping[str, sg.Rankings]) -> dict[str, sg.Rankings]:
    """Give every ranking its parent labels too, scored by their strongest child unless already scored."""
    return {
        name: {item_id: bm.with_parents(ranking) for item_id, ranking in rankings.items()}
        for name, rankings in sources.items()
    }


def humset_score(
    tune_gold: Sequence[set[str]],
    tune_rankings: Sequence[Mapping[str, float]],
    gold: Sequence[set[str]],
    rankings: Sequence[Mapping[str, float]],
    taxonomy: Taxonomy,
) -> dict[str, float]:
    """Return macro F1 per published category, with thresholds per label chosen on the tuning items."""
    scores = {}
    for category, labels in bm.humset_categories(taxonomy).items():
        thresholds = bm.label_thresholds(tune_gold, tune_rankings, labels)
        scores[category] = bm.macro_f1(gold, bm.apply_thresholds(rankings, thresholds), labels)
    return scores


def eurlex_score(
    gold: Sequence[set[str]], rankings: Sequence[Mapping[str, float]], taxonomy: Taxonomy
) -> dict[str, float]:
    """Return mean R-Precision on all labels and rolled up to level 3."""
    gold_three = [taxonomy.roll_up(truth, 3) for truth in gold]
    rolled = [bm.rank_at_tier(ranking, taxonomy, 3) for ranking in rankings]
    return {
        "all": bm.mean_r_precision(gold, rankings),
        "level_3": bm.mean_r_precision(gold_three, rolled),
    }


def headline(test_set: str, scores: Mapping[str, float]) -> float:
    """Return the one number that decides whether a source stays."""
    return float(np.mean(list(scores.values()))) if test_set == "humset" else scores["all"]


def combine(
    fit: Sequence[ds.Item],
    tune: Sequence[ds.Item],
    sources: Mapping[str, sg.Rankings],
    names: Sequence[str],
    graph: cb.LabelGraph,
    words: int,
    settings: cb.CombinedSettings,
) -> dict[str, Any]:
    """Train the combiner on one set of development items and return it with its scores on another."""
    trained = sg.combine_and_score(fit, sources, tune, sources, names, graph, words, settings)
    tuned = sg.apply_combiner(trained["model"], tune, sources, names, graph, words, settings)
    return {**trained, "tune_rankings": tuned}


def source_scores(
    test_set: str,
    tune: Sequence[ds.Item],
    fit: Sequence[ds.Item],
    sources: Mapping[str, sg.Rankings],
    taxonomy: Taxonomy,
) -> dict[str, dict[str, float]]:
    """Return every source's own score on development items. On HumSet its thresholds come from the fitting
    items, and scores from the others."""
    results = {}
    gold = [set(item.labels) for item in tune]
    for name, rankings in sources.items():
        tuned = [rankings.get(item.id, {}) for item in tune]
        if test_set == "humset":
            fit_gold = [set(item.labels) for item in fit]
            fit_rankings = [rankings.get(item.id, {}) for item in fit]
            results[name] = humset_score(fit_gold, fit_rankings, gold, tuned, taxonomy)
        else:
            results[name] = eurlex_score(gold, tuned, taxonomy)
    return results


def choose_sources(
    test_set: str,
    all_scores: Mapping[str, Mapping[str, float]],
    full: str = "all sources",
) -> list[str]:
    """Keep each source whose removal costs more than the margin. `all_scores` holds the score with every
    source under `full`, and each other entry is the score without the named source."""
    margin = KEEP_MARGIN[test_set]
    base = headline(test_set, all_scores[full])
    return sorted(
        name
        for name, scores in all_scores.items()
        if name != full and base - headline(test_set, scores) > margin
    )


def development_choice(
    test_set: str,
    fit: Sequence[ds.Item],
    cut: Sequence[ds.Item],
    threshold_items: Sequence[ds.Item],
    score_items: Sequence[ds.Item],
    sources: Mapping[str, sg.Rankings],
    graph: cb.LabelGraph,
    words: int,
    settings: cb.CombinedSettings,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Score each source alone, the combiner with every source and without each one, choose the sources and
    score the combiner with those alone. The combiner trains on `fit` and ranks `cut`; scores use
    `score_items`, a part of `cut`, with any thresholds from `threshold_items`. Raises if none is chosen."""
    names = sorted(sources)

    def with_sources(use: Sequence[str]) -> dict[str, float]:
        made = combine(fit, cut, sources, use, graph, words, settings)
        ranked = {item.id: ranking for item, ranking in zip(cut, made["tune_rankings"], strict=True)}
        return source_scores(test_set, score_items, threshold_items, {"combined": ranked}, taxonomy)[
            "combined"
        ]

    combined = {"all sources": with_sources(names)}
    combined.update({name: with_sources([other for other in names if other != name]) for name in names})
    chosen = choose_sources(test_set, combined)
    if not chosen:
        raise ValueError("no source's removal costs more than the margin, so none was chosen")
    return {
        "single": source_scores(test_set, score_items, threshold_items, sources, taxonomy),
        "combined": combined,
        "chosen": chosen,
        "chosen_scores": with_sources(chosen),
    }


PRIORS = "time-aware priors"


def addition_choice(
    test_set: str,
    fit: Sequence[ds.Item],
    cut: Sequence[ds.Item],
    threshold_items: Sequence[ds.Item],
    score_items: Sequence[ds.Item],
    sources: Mapping[str, sg.Rankings],
    base: Sequence[str],
    graphs: tuple[cb.LabelGraph, cb.LabelGraph | None],
    words: int,
    settings: cb.CombinedSettings,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Score a stack already chosen, then the same stack with each new source added and, if a graph with
    time-aware priors is given, with those priors. An addition is kept if it raises the score by more than
    the margin, and the stack with every kept addition is scored. `graphs` holds the graph without priors
    and the graph with them, or None. New sources are those in `sources` but not in `base`."""
    plain, timed = graphs
    margin = KEEP_MARGIN[test_set]

    def score(names: Sequence[str], graph: cb.LabelGraph) -> dict[str, float]:
        made = combine(fit, cut, sources, sorted(names), graph, words, settings)
        ranked = {item.id: ranking for item, ranking in zip(cut, made["tune_rankings"], strict=True)}
        return source_scores(test_set, score_items, threshold_items, {"combined": ranked}, taxonomy)[
            "combined"
        ]

    variants = {"first attempt": score(base, plain)}
    added = sorted(set(sources) - set(base))
    variants.update({f"+ {name}": score([*base, name], plain) for name in added})
    if timed is not None:
        variants[f"+ {PRIORS}"] = score(base, timed)
    before = headline(test_set, variants["first attempt"])
    kept = sorted(
        name.removeprefix("+ ")
        for name, scores in variants.items()
        if name != "first attempt" and headline(test_set, scores) - before > margin
    )
    names = sorted({*base, *(name for name in kept if name != PRIORS)})
    graph = timed if PRIORS in kept and timed is not None else plain
    return {
        "variants": variants,
        "kept": kept,
        "names": names,
        "priors": PRIORS in kept,
        "graph": graph,
        "chosen_scores": score(names, graph) if kept else variants["first attempt"],
        "single": source_scores(test_set, score_items, threshold_items, sources, taxonomy),
    }


def best_single(test_set: str, choice: Mapping[str, Any]) -> str:
    """Return the source that scored best alone on development items."""
    single = choice["single"]
    return max(sorted(single), key=lambda name: headline(test_set, single[name]))


def humset_passes(scores: Mapping[str, tuple[float, float]]) -> dict[str, bool]:
    """Return whether each category's interval lies above HumBERT's printed score."""
    return {category: scores[category][0] > bm.HUMBERT_BEST[category] for category in bm.HUMBERT_BEST}


def eurlex_passes(scores: Mapping[str, tuple[float, float]]) -> dict[str, bool]:
    """Return whether each mean R-Precision interval lies above BERT's printed score."""
    return {name: scores[name][0] > bm.MULTI_EURLEX_BERT[name] for name in bm.MULTI_EURLEX_BERT}


def macro_interval(
    gold: Sequence[set[str]],
    predicted: Sequence[set[str]],
    labels: frozenset[str],
    seed: int,
    draws: int = 200,
) -> tuple[float, float]:
    """Return a bootstrap interval for macro F1, resampling items."""
    ordered = sorted(labels)
    truth = np.array([[label in item for label in ordered] for item in gold])
    kept = np.array([[label in item for label in ordered] for item in predicted])

    def score(picks: np.ndarray) -> float:
        hits = (truth[picks] & kept[picks]).sum(axis=0)
        totals = truth[picks].sum(axis=0) + kept[picks].sum(axis=0)
        return float(np.mean(np.divide(2 * hits, totals, out=np.zeros(len(ordered)), where=totals > 0)))

    return bm.bootstrap_interval(score, len(gold), seed, draws)


def mrp_interval(
    gold: Sequence[set[str]], rankings: Sequence[Mapping[str, float]], seed: int, draws: int = 200
) -> tuple[float, float]:
    """Return a bootstrap interval for mean R-Precision as a percentage."""
    values = 100 * bm.r_precisions(gold, rankings)
    return bm.bootstrap_interval(lambda picks: float(values[picks].mean()), len(values), seed, draws)


def exact_micro_f1(gold: Sequence[set[str]], predicted: Sequence[set[str]]) -> float:
    """Return micro F1 for comparison with the earlier notebooks, less any label given with its own child."""
    return micro_f1(gold, [without_implied_parents(labels) for labels in predicted])


def eurlex_sources(root: Path, names: Sequence[str], split: str, with_newer: bool) -> dict[str, sg.Rankings]:
    """Load MultiEURLEX sources, from models trained with the newer laws where asked. `classifier_train_only`
    is the fine-tuned encoder trained without them. A missing source raises."""
    ds_name = "eurlex"
    folder = sg.NEWER_LAWS_SUFFIX.format(ds_name) if with_newer else ds_name
    loaded = {}
    for name in names:
        if name == "classifier_train_only":
            loaded[name] = sg.load_signal(root, ds_name, "classifier", split)
        elif name in TRAIN_ONLY_SOURCES:
            loaded[name] = sg.load_signal(root, ds_name, name, split)
        else:
            loaded[name] = sg.load_signal(root, folder, name, split)
    return loaded


def without_newer(names: Sequence[str]) -> list[str]:
    """Return the sources a stack trained on training laws alone can use, in place of those chosen with the
    newer laws."""
    swapped = {"classifier" if name == "classifier_train_only" else name for name in names}
    return sorted(swapped - NEWER_ONLY_SOURCES)


def top_r(gold: Sequence[set[str]], rankings: Sequence[Mapping[str, float]]) -> list[set[str]]:
    """Return each item's top labels, as many as it has gold labels, which is what R-Precision scores."""
    return [
        set(sorted(ranking, key=lambda label: (-ranking[label], label))[: len(truth)])
        for truth, ranking in zip(gold, rankings, strict=True)
    ]


def category_matrices(
    gold: Sequence[set[str]],
    predicted: Mapping[str, Sequence[set[str]]],
    categories: Mapping[str, frozenset[str]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return, per category, items by labels as true and predicted matrices, ready for resampling."""
    matrices = {}
    for category, labels in categories.items():
        ordered = sorted(labels)
        truth = np.array([[label in item for label in ordered] for item in gold])
        kept = np.array([[label in item for label in ordered] for item in predicted[category]])
        matrices[category] = (truth, kept)
    return matrices


def macro_scores(
    matrices: Mapping[str, tuple[np.ndarray, np.ndarray]], picks: np.ndarray
) -> dict[str, float]:
    """Return macro F1 per category, and their mean, on the resampled items."""
    scores = {}
    for category, (truth, kept) in matrices.items():
        hits = (truth[picks] & kept[picks]).sum(axis=0)
        totals = truth[picks].sum(axis=0) + kept[picks].sum(axis=0)
        scores[category] = float(
            np.mean(np.divide(2 * hits, totals, out=np.zeros(len(hits)), where=totals > 0))
        )
    scores["Mean"] = float(np.mean(list(scores.values())))
    return scores


def humset_intervals(
    matrices: Mapping[str, tuple[np.ndarray, np.ndarray]], seed: int, draws: int = 200
) -> dict[str, tuple[float, float]]:
    """Return a bootstrap interval for each category's macro F1 and for their mean, resampling items."""
    items = len(next(iter(matrices.values()))[0])
    rng = np.random.default_rng(seed)
    values = [macro_scores(matrices, rng.integers(0, items, items)) for _ in range(draws)]
    return {
        name: (
            float(np.percentile([v[name] for v in values], 2.5)),
            float(np.percentile([v[name] for v in values], 97.5)),
        )
        for name in values[0]
    }


def humset_gain(
    first: Mapping[str, tuple[np.ndarray, np.ndarray]],
    second: Mapping[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    draws: int = 200,
) -> tuple[float, float, float]:
    """Return the first's mean macro F1 minus the second's, with a 95% interval on the same resampled
    items."""
    items = len(next(iter(first.values()))[0])
    everything = np.arange(items)
    point = macro_scores(first, everything)["Mean"] - macro_scores(second, everything)["Mean"]
    low, high = bm.bootstrap_interval(
        lambda picks: macro_scores(first, picks)["Mean"] - macro_scores(second, picks)["Mean"],
        items,
        seed,
        draws,
    )
    return point, low, high


def mrp_gain(
    gold: Sequence[set[str]],
    first: Sequence[Mapping[str, float]],
    second: Sequence[Mapping[str, float]],
    seed: int,
    draws: int = 200,
) -> tuple[float, float, float]:
    """Return the first's mean R-Precision minus the second's, in points, with a 95% paired interval."""
    gaps = 100 * (bm.r_precisions(gold, first) - bm.r_precisions(gold, second))
    low, high = bm.bootstrap_interval(lambda picks: float(gaps[picks].mean()), len(gaps), seed, draws)
    return float(gaps.mean()), low, high


@dataclass(frozen=True)
class Stack:
    """The chosen sources and what the combiner needs to train on development items and rank test items."""

    names: Sequence[str]
    fit: Sequence[ds.Item]
    cut: Sequence[ds.Item]
    dev_sources: Mapping[str, sg.Rankings]
    test_sources: Mapping[str, sg.Rankings]
    graph: cb.LabelGraph
    words: int
    settings: cb.CombinedSettings


def require_rankings(sources: Mapping[str, sg.Rankings], items: Sequence[ds.Item]) -> None:
    """Raise if any source has no ranking, even an empty one, for any of the items."""
    for name, rankings in sources.items():
        missing = sum(1 for item in items if item.id not in rankings)
        if missing:
            raise ValueError(f"{name} has no ranking for {missing} of {len(items)} items")


def importance(model: Any, stack: Stack, seed: int) -> dict[str, float]:
    """Return how much the trained combiner relies on each signal, on the development items it was cut on."""
    rows, keys = cb.build_rows(
        sg.assemble(stack.cut, stack.dev_sources, stack.names),
        [ds.first_words(item.text, stack.words) for item in stack.cut],
        stack.graph,
        stack.names,
        stack.settings.per_source,
    )
    target = np.array([label in stack.cut[position].labels for position, label in keys])
    return sg.signal_importance(
        model, rows, target, cb.feature_names(stack.names, stack.graph.priors is not None), seed
    )


def _rank_test(stack: Stack, test: Sequence[ds.Item]) -> dict[str, Any]:
    """Train the combiner on the fitting items, and return it with its rankings of the cut and test items."""
    require_rankings({name: stack.test_sources[name] for name in stack.names}, test)
    made = combine(
        stack.fit, stack.cut, stack.dev_sources, stack.names, stack.graph, stack.words, stack.settings
    )
    ranked = sg.apply_combiner(
        made["model"], test, stack.test_sources, stack.names, stack.graph, stack.words, stack.settings
    )
    return {"model": made["model"], "cut": made["tune_rankings"], "test": ranked}


def humset_final(
    stack: Stack, test: Sequence[ds.Item], best_single: str, taxonomy: Taxonomy, seed: int, draws: int = 200
) -> dict[str, Any]:
    """Score the combined stack once on the test items, beside the best single source chosen on development
    items. Each label's threshold comes from the cut items, for the stack and the single source alike. As in
    HumBERT's measure, every ranking covers every tier: a pillar it leaves out takes its strongest subpillar's
    score (`bm.with_parents`), the same rule `like_for_like.paper_scores` applies."""
    require_rankings({best_single: stack.test_sources[best_single]}, test)
    made = _rank_test(stack, test)
    categories = bm.humset_categories(taxonomy)
    cut_gold = [set(item.labels) for item in stack.cut]
    gold = [set(item.labels) for item in test]

    def kept(cut_rankings: Sequence[Mapping[str, float]], rankings: Sequence[Mapping[str, float]]) -> dict:
        cut_full = [bm.with_parents(ranking) for ranking in cut_rankings]
        full = [bm.with_parents(ranking) for ranking in rankings]
        return {
            category: bm.apply_thresholds(full, bm.label_thresholds(cut_gold, cut_full, labels))
            for category, labels in categories.items()
        }

    new = kept(made["cut"], made["test"])
    single = stack.dev_sources[best_single]
    best = kept(
        [single.get(item.id, {}) for item in stack.cut], [stack.test_sources[best_single][i.id] for i in test]
    )
    new_matrices = category_matrices(gold, new, categories)
    best_matrices = category_matrices(gold, best, categories)
    everything = np.arange(len(test))
    intervals = humset_intervals(new_matrices, seed, draws)
    printed_mean = float(np.mean(list(bm.HUMBERT_BEST.values())))
    return {
        "scores": macro_scores(new_matrices, everything),
        "intervals": intervals,
        "best_single": macro_scores(best_matrices, everything),
        "gain_over_best_single": humset_gain(new_matrices, best_matrices, seed, draws),
        "passes": humset_passes(intervals),
        "well_ahead": intervals["Mean"][0] > printed_mean + WELL_AHEAD["humset"],
        "other_averages": {
            category: bm.other_averages(gold, new[category], labels)
            for category, labels in categories.items()
        },
        "importance": importance(made["model"], stack, seed),
        "rankings": made["test"],
    }


def eurlex_final(
    stack: Stack,
    test: Sequence[ds.Item],
    best_single: str,
    taxonomy: Taxonomy,
    counts: Mapping[str, int],
    seed: int,
    draws: int = 200,
) -> dict[str, Any]:
    """Score the combined stack once on the test laws, beside the best single source chosen on development
    laws, with recall by how often each label was seen in what the stack trained on."""
    require_rankings({best_single: stack.test_sources[best_single]}, test)
    made = _rank_test(stack, test)
    gold = {"all": [set(item.labels) for item in test]}
    gold["level_3"] = [taxonomy.roll_up(truth, 3) for truth in gold["all"]]
    new = {"all": made["test"], "level_3": [bm.rank_at_tier(r, taxonomy, 3) for r in made["test"]]}
    single = [stack.test_sources[best_single][item.id] for item in test]
    best = {"all": single, "level_3": [bm.rank_at_tier(r, taxonomy, 3) for r in single]}
    intervals = {level: mrp_interval(gold[level], new[level], seed, draws) for level in gold}
    return {
        "scores": {level: bm.mean_r_precision(gold[level], new[level]) for level in gold},
        "intervals": intervals,
        "best_single": {level: bm.mean_r_precision(gold[level], best[level]) for level in gold},
        "gain_over_best_single": {
            level: mrp_gain(gold[level], new[level], best[level], seed, draws) for level in gold
        },
        "passes": eurlex_passes(intervals),
        "well_ahead": {
            level: intervals[level][0] > bm.MULTI_EURLEX_BERT[level] + WELL_AHEAD["eurlex"] for level in gold
        },
        "rarity": {
            "new_technique": ex.recall_by_rarity(gold["all"], top_r(gold["all"], new["all"]), counts),
            best_single: ex.recall_by_rarity(gold["all"], top_r(gold["all"], single), counts),
        },
        "importance": importance(made["model"], stack, seed),
        "rankings": made["test"],
    }


def _humset_matrices(
    stack: Stack, test: Sequence[ds.Item], taxonomy: Taxonomy
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rank the test items with a stack, keep labels above thresholds chosen on its cut items, and return
    the true and kept matrices per category."""
    made = _rank_test(stack, test)
    categories = bm.humset_categories(taxonomy)
    cut_gold = [set(item.labels) for item in stack.cut]
    kept = {
        category: bm.apply_thresholds(made["test"], bm.label_thresholds(cut_gold, made["cut"], labels))
        for category, labels in categories.items()
    }
    return category_matrices([set(item.labels) for item in test], kept, categories)


def humset_against(
    second: Stack, first: Stack, test: Sequence[ds.Item], taxonomy: Taxonomy, seed: int, draws: int = 200
) -> dict[str, Any]:
    """Return both stacks' macro F1 on the test items and the second's gain in mean macro F1 over the first,
    with a 95% interval on the same resampled items."""
    new, old = _humset_matrices(second, test, taxonomy), _humset_matrices(first, test, taxonomy)
    everything = np.arange(len(test))
    return {
        "second": macro_scores(new, everything),
        "first": macro_scores(old, everything),
        "gain": humset_gain(new, old, seed, draws),
    }


def eurlex_against(
    second: Stack, first: Stack, test: Sequence[ds.Item], taxonomy: Taxonomy, seed: int, draws: int = 200
) -> dict[str, Any]:
    """Return both stacks' mean R-Precision on the test laws, on all labels and at level 3, and the second's
    gain over the first in points, with a 95% paired interval."""
    gold = {"all": [set(item.labels) for item in test]}
    gold["level_3"] = [taxonomy.roll_up(truth, 3) for truth in gold["all"]]
    ranked = {}
    for name, stack in (("second", second), ("first", first)):
        made = _rank_test(stack, test)["test"]
        ranked[name] = {"all": made, "level_3": [bm.rank_at_tier(r, taxonomy, 3) for r in made]}
    return {
        **{
            name: {level: bm.mean_r_precision(gold[level], ranked[name][level]) for level in gold}
            for name in ranked
        },
        "gain": {
            level: mrp_gain(gold[level], ranked["second"][level], ranked["first"][level], seed, draws)
            for level in gold
        },
    }
