"""Shared steps of the notebooks: load a test set, sample it, run a method, score it and save the results."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tiered_taxonomy import datasets as ds
from tiered_taxonomy import eurovoc
from tiered_taxonomy.config import RESULTS
from tiered_taxonomy.llm import Client, run_all
from tiered_taxonomy.methods.base import Prediction, combine
from tiered_taxonomy.metrics import LabelSets, score
from tiered_taxonomy.taxonomy import Taxonomy, without_implied_parents

EURLEX = "eurlex"
HUMSET = "humset"
# HumSet 2.0.0, the version HumBERT's authors' code loads, kept under its own name so that its results never
# mix with version 1.0.0's. It follows every HumSet rule.
HUMSET_V2 = "humset_v2"
HUMSET_VERSIONS = {HUMSET: "1.0.0", HUMSET_V2: "2.0.0"}
# The languages a HumSet sample draws from. Version 2.0.0's test set holds one Portuguese excerpt.
HUMSET_LANGUAGES = ("en", "es", "fr")
# A label counts as rare when it is among the least-used quarter of the labels given in training, so that
# "rare" means the same thing in a long-tailed taxonomy and in a small, evenly used one.
RARE_SHARE = 0.25


@dataclass
class TestSet:
    """A taxonomy with train, development and test items, and the gold labels' frequency in training."""

    name: str
    taxonomy: Taxonomy
    tiers: tuple[int, ...]
    train: list[ds.Item]
    dev: list[ds.Item]
    test: list[ds.Item]
    train_counts: Counter[str]


def _keep_known(items: Sequence[ds.Item], taxonomy: Taxonomy) -> list[ds.Item]:
    """Drop gold labels with no place in the taxonomy, and set each item's gold by tier from that same
    tree."""
    tiers = sorted(set(taxonomy.tiers.values()))
    kept = []
    for item in items:
        labels = frozenset(label for label in item.labels if label in taxonomy.nodes)
        by_tier = {tier: frozenset(taxonomy.roll_up(labels, tier)) for tier in tiers}
        kept.append(ds.Item(item.id, item.text, item.language, labels, item.group, by_tier, item.project))
    return kept


def load_eurlex(root: Path) -> TestSet:
    """Load MultiEURLEX with the EuroVoc tree. Gold labels at every tier come from rolling up through that
    tree."""
    ds.prepare_eurlex(root)
    folder = root / "multi_eurlex"
    names = json.loads((folder / "eurovoc_descriptors.json").read_text(encoding="utf-8"))
    edges = eurovoc.fetch_edges(folder / "eurovoc_edges.json")
    frames = {split: ds.load_eurlex_rows(root, split) for split in ("train", "dev", "test")}
    domains = sorted({label for frame in frames.values() for labels in frame["level_1"] for label in labels})
    taxonomy = eurovoc.build_taxonomy(edges, names, domains)
    items = {split: _keep_known(ds.eurlex_items(frame), taxonomy) for split, frame in frames.items()}
    counts = Counter(label for item in items["train"] for label in item.labels)
    return TestSet(EURLEX, taxonomy, (1, 2, 3), items["train"], items["dev"], items["test"], counts)


def is_humset(name: str) -> bool:
    """Return whether a test set is a version of HumSet, so that HumSet's rules apply to it."""
    return name in HUMSET_VERSIONS


def load_humset(root: Path, name: str = HUMSET) -> TestSet:
    """Load a version of HumSet. Its taxonomy is pillars over subpillars, with sectors as flat labels at the
    top tier."""
    version = HUMSET_VERSIONS[name]
    frames = {split: ds.load_humset_rows(root, split, version) for split in ("train", "validation", "test")}
    taxonomy = ds.humset_taxonomy(pd.concat(frames.values()))
    items = {
        split: _keep_known(ds.humset_items(frame, taxonomy), taxonomy) for split, frame in frames.items()
    }
    counts = Counter(label for item in items["train"] for label in item.labels)
    return TestSet(name, taxonomy, (1, 2), items["train"], items["validation"], items["test"], counts)


def input_text(item: ds.Item, words: int) -> str:
    """Return the opening of an item, which stands in for it in the methods that read one piece of text."""
    return ds.first_words(item.text, words)


def gold_exact(items: Sequence[ds.Item]) -> list[set[str]]:
    """Return items' gold labels, at whatever tier they were assigned."""
    return [set(item.labels) for item in items]


def gold_at_tier(items: Sequence[ds.Item], tier: int) -> list[set[str]]:
    """Return items' gold labels rolled up to a tier."""
    return [set(item.labels_by_tier[tier]) for item in items]


def rare_cutoff(counts: Mapping[str, int]) -> int:
    """Return the most times a label can have been given in training and still be rare: the count at which
    the least-used quarter of the labels given in training is reached. Ties at that count are rare too.
    When no label was given in training, none can be rare, so the cut-off is 0 and every label is unseen."""
    used = sorted(count for count in counts.values() if count > 0)
    return used[math.ceil(len(used) * RARE_SHARE) - 1] if used else 0


def rarity(label: str, counts: Mapping[str, int], rare_at_most: int) -> str:
    """Return how often a label was seen in training: never, rarely or commonly."""
    seen = counts.get(label, 0)
    if seen == 0:
        return "unseen"
    return "rare" if seen <= rare_at_most else "common"


def recall_by_rarity(
    gold: LabelSets, predicted: LabelSets, counts: Mapping[str, int]
) -> dict[str, dict[str, float]]:
    """Return recall of the gold labels by how often each label was seen in training."""
    hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    cutoff = rare_cutoff(counts)
    for gold_labels, predicted_labels in zip(gold, predicted, strict=True):
        for label in gold_labels:
            band = rarity(label, counts, cutoff)
            totals[band] += 1
            hits[band] += int(label in predicted_labels)
    return {
        band: {"recall": round(hits[band] / totals[band], 4), "gold_labels": totals[band]}
        for band in sorted(totals)
    }


def evaluate(test_set: TestSet, items: Sequence[ds.Item], predicted: LabelSets) -> dict[str, Any]:
    """Score predictions exactly, at each tier, by language and by label rarity. A label given with one of its
    own children is dropped first (`taxonomy.without_implied_parents`)."""
    taxonomy = test_set.taxonomy
    known = [
        without_implied_parents(label for label in labels if label in taxonomy.nodes) for labels in predicted
    ]
    results: dict[str, Any] = {"exact": asdict(score(gold_exact(items), known))}
    results["tiers"] = {
        str(tier): asdict(
            score(gold_at_tier(items, tier), [taxonomy.roll_up(labels, tier) for labels in known])
        )
        for tier in test_set.tiers
    }
    results["languages"] = _by_language(test_set, items, known)
    results["rarity"] = recall_by_rarity(gold_exact(items), known, test_set.train_counts)
    results["labels_per_item"] = round(float(np.mean([len(labels) for labels in known])), 2) if known else 0.0
    return results


def _by_language(
    test_set: TestSet, items: Sequence[ds.Item], predicted: LabelSets
) -> dict[str, dict[str, float]]:
    taxonomy = test_set.taxonomy
    tier = test_set.tiers[-1]
    results: dict[str, dict[str, float]] = {}
    for language in sorted({item.language for item in items}):
        picks = [i for i, item in enumerate(items) if item.language == language]
        chosen = [items[i] for i in picks]
        rolled = [taxonomy.roll_up(predicted[i], tier) for i in picks]
        results[language] = {
            "f1_finest_tier": round(score(gold_at_tier(chosen, tier), rolled).f1, 4),
            "items": len(chosen),
        }
    return results


def project_labels(items: Sequence[ds.Item], taxonomy: Taxonomy) -> dict[str, set[str]]:
    """Return the labels each project ever used, including the ancestors of those labels."""
    used: dict[str, set[str]] = defaultdict(set)
    for item in items:
        for label in item.labels:
            used[item.project].add(label)
            used[item.project].update(taxonomy.ancestors(label))
    return dict(used)


def evaluate_within_project(
    test_set: TestSet, items: Sequence[ds.Item], predicted: LabelSets, used: Mapping[str, set[str]]
) -> dict[str, Any]:
    """Score only the labels each item's project ever used, since a missing label there does not mean no."""
    allowed = [used.get(item.project, set()) for item in items]
    known = [{label for label in labels if label in test_set.taxonomy.nodes} for labels in predicted]
    results: dict[str, Any] = {}
    for tier in test_set.tiers:
        rolled = [test_set.taxonomy.roll_up(labels, tier) for labels in known]
        kept = [labels & limit for labels, limit in zip(rolled, allowed, strict=True)]
        results[str(tier)] = asdict(score(gold_at_tier(items, tier), kept))
    return results


@dataclass
class RunRecord:
    """What a method cost to run."""

    method: str
    items: int
    seconds: float
    calls: int
    malformed: int
    invalid: int
    cost_usd: float


def run_method(
    name: str,
    work: Callable[[ds.Item], Prediction],
    items: Sequence[ds.Item],
    clients: Sequence[Client],
    workers: int,
) -> tuple[list[Prediction], RunRecord]:
    """Run a method over items, recording its time, calls, failures and estimated cost."""
    costs_before = [client.cost() for client in clients]
    started = time.monotonic()
    predictions = run_all(work, items, workers)
    seconds = time.monotonic() - started
    calls, malformed, invalid = combine(predictions)
    cost = sum(client.cost() - before for client, before in zip(clients, costs_before, strict=True))
    return predictions, RunRecord(
        name, len(items), round(seconds, 1), calls, malformed, invalid, round(cost, 4)
    )


def label_sets(predictions: Sequence[Prediction]) -> list[set[str]]:
    """Return the labels of each prediction."""
    return [set(prediction.labels) for prediction in predictions]


def save_results(name: str, results: Mapping[str, Any]) -> Path:
    """Save a notebook's results in outputs/results, and return the path."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_results(name: str) -> dict[str, Any]:
    """Read a notebook's saved results. A missing file raises."""
    results: dict[str, Any] = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
    return results


def save_predictions(
    root: Path, test_set: str, method: str, ids: Sequence[str], predicted: LabelSets
) -> Path:
    """Save a method's labels per item in the data folder, so that later notebooks can compare methods."""
    folder = root / "predictions" / test_set
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{method}.json"
    body = {item_id: sorted(labels) for item_id, labels in zip(ids, predicted, strict=True)}
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def load_predictions(root: Path, test_set: str, method: str, ids: Sequence[str]) -> list[set[str]]:
    """Read a method's labels back, in the order of the given ids. A missing item raises."""
    body = json.loads((root / "predictions" / test_set / f"{method}.json").read_text(encoding="utf-8"))
    return [set(body[item_id]) for item_id in ids]


def sample_eurlex(
    test_set: TestSet, normal: int, long_documents: int, threshold: int, seed: int
) -> list[ds.Item]:
    """Return a repeatable sample of test laws: `normal` under the length threshold and `long_documents`
    over it."""
    lengths = np.array([ds.word_count(item.text) for item in test_set.test])
    long_items, normal_items = ds.sample_by_length(lengths, threshold, normal, seed)
    rng = np.random.default_rng(seed)
    kept_long = rng.choice(long_items, size=min(long_documents, len(long_items)), replace=False)
    return [test_set.test[int(i)] for i in sorted([*normal_items, *kept_long.tolist()])]


def split_overlap(train: Sequence[ds.Item], test: Sequence[ds.Item]) -> dict[str, float]:
    """Return the share of test items whose source document also has items in training, and the share whose
    text appears word for word in training."""
    documents = {item.group for item in train}
    texts = {item.text.strip() for item in train}
    return {
        "same_document": round(sum(item.group in documents for item in test) / len(test), 4),
        "same_text": round(sum(item.text.strip() in texts for item in test) / len(test), 4),
    }


def sample_humset(test_set: TestSet, per_language: int, seed: int) -> list[ds.Item]:
    """Return a repeatable sample of test excerpts in English, French and Spanish, balanced by language."""
    kept = [item for item in test_set.test if item.language in HUMSET_LANGUAGES]
    picks = ds.sample_by_language([item.language for item in kept], per_language, seed)
    return [kept[i] for i in picks]


def pilot_subset(items: Sequence[ds.Item], size: int, seed: int) -> list[ds.Item]:
    """Return a repeatable random subset for the pilot run."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(items), size=min(size, len(items)), replace=False)
    return [items[int(i)] for i in sorted(picks.tolist())]
