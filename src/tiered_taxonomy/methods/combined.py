"""The combined method: several signals per candidate label, weighed by a model that learns when to trust
each.

This design is ours and comes from no single paper. Each candidate label for a document gets a row of
signals: trained
classifiers, labels of similar documents, search by label name, exact name matches, and the taxonomy and label
co-occurrence graph. A gradient-boosted model, trained on a split the classifiers never saw, turns those
into one
score. Because MultiEURLEX's splits run in time order, that split contains labels unseen in training, which
is how the
model learns to trust search and the LLM when a classifier has no opinion. An LLM is asked only about labels
the model
is unsure of, or that are rare.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from tiered_taxonomy.taxonomy import Taxonomy

Ranking = Mapping[str, float]


@dataclass(frozen=True)
class CombinedSettings:
    """Limits on how many candidates are considered per document, and how the combiner trains."""

    per_source: int = 30
    max_iterations: int = 300
    learning_rate: float = 0.05
    seed: int = 7


TIME_FEATURES = ("log_recent_count", "recent_trend", "years_since_used")


@dataclass(frozen=True)
class TimedPriors:
    """How each label's use changes over time in training: its count among the most recent documents, that
    share against its overall share, and the years since it was last used. Labels drift on a chronological
    split, so recent use predicts later use better than overall use alone."""

    recent: Mapping[str, int]
    overall: Mapping[str, int]
    recent_total: int
    overall_total: int
    last_used: Mapping[str, int]
    latest_year: int

    @classmethod
    def from_documents(
        cls, documents: Sequence[tuple[int, Iterable[str]]], recent_documents: int
    ) -> TimedPriors:
        """Build priors from (year, labels) pairs in date order, oldest first. Raises if they are out of order
        or if the recent window is not smaller than the whole."""
        years = [year for year, _ in documents]
        if years != sorted(years):
            raise ValueError("documents must be in date order, oldest first")
        if not 0 < recent_documents < len(documents):
            raise ValueError(f"the recent window of {recent_documents} must be smaller than {len(documents)}")
        overall: Counter[str] = Counter()
        recent: Counter[str] = Counter()
        last_used: dict[str, int] = {}
        start = len(documents) - recent_documents
        for position, (year, labels) in enumerate(documents):
            unique = set(labels)
            overall.update(unique)
            if position >= start:
                recent.update(unique)
            for label in unique:
                last_used[label] = year
        return cls(recent, overall, recent_documents, len(documents), last_used, years[-1])

    def features(self, label: str) -> list[float]:
        """Return the label's recent count (logged), its recent against overall share (logged), and the years
        since its last use. The last two are missing for a label never used, which the combiner handles."""
        if self.overall.get(label, 0) == 0:
            return [0.0, math.nan, math.nan]
        recent = self.recent.get(label, 0)
        recent_share = (recent + 0.5) / (self.recent_total + 1)
        overall_share = (self.overall[label] + 0.5) / (self.overall_total + 1)
        return [
            math.log1p(recent),
            math.log(recent_share / overall_share),
            float(self.latest_year - self.last_used[label]),
        ]


class LabelGraph:
    """Links between labels: taxonomy parents and children, and labels that often appear together in
    training. With `priors`, each label's change in use over time is added to its signals."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        documents: Iterable[Iterable[str]],
        top_neighbours: int = 20,
        priors: TimedPriors | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.priors = priors
        pairs: dict[str, Counter[str]] = defaultdict(Counter)
        counts: Counter[str] = Counter()
        for labels in documents:
            unique = sorted(set(labels))
            counts.update(unique)
            for label in unique:
                pairs[label].update(other for other in unique if other != label)
        self.counts = counts
        self.neighbours: dict[str, dict[str, float]] = {
            label: {
                other: together / math.sqrt(counts[label] * counts[other])
                for other, together in partners.most_common(top_neighbours)
            }
            for label, partners in pairs.items()
        }

    def co_occurrence_support(self, label: str, scores: Ranking) -> float:
        """Return how strongly the other candidates' scores point to this label through co-occurrence."""
        return sum(
            weight * scores.get(other, 0.0) for other, weight in self.neighbours.get(label, {}).items()
        )

    def family_support(self, label: str, scores: Ranking) -> tuple[float, float]:
        """Return the strongest candidate score among a label's parents and among its children."""
        if label not in self.taxonomy.nodes:
            return 0.0, 0.0
        parents = self.taxonomy.nodes[label].parents
        children = self.taxonomy.children.get(label, [])
        return (
            max((scores.get(parent, 0.0) for parent in parents), default=0.0),
            max((scores.get(child, 0.0) for child in children), default=0.0),
        )


def _strongest(ranking: Ranking, count: int) -> list[str]:
    return sorted(ranking, key=lambda label: (-ranking[label], label))[:count]


def candidates_for(sources: Mapping[str, Ranking], per_source: int) -> list[str]:
    """Return the union of each source's strongest labels for one document."""
    found: set[str] = set()
    for ranking in sources.values():
        found.update(_strongest(ranking, per_source))
    return sorted(found)


def name_in_text(name: str, text: str) -> float:
    """Return 1 if a label's name appears word for word in a lower-cased text."""
    return float(f" {name.lower()} " in f" {text} ")


def feature_names(source_names: Sequence[str], timed: bool = False) -> list[str]:
    """Return the signal names, in the order `label_features` writes them. `timed` adds the time-aware
    priors."""
    signals = []
    for source in source_names:
        signals.extend([f"{source}_score", f"{source}_found"])
    return [
        *signals,
        "sources_agreeing",
        "name_in_text",
        "log_training_count",
        "tier",
        "co_occurrence",
        "parent_score",
        "child_score",
        *(TIME_FEATURES if timed else ()),
    ]


def pooled_scores(sources: Mapping[str, Ranking]) -> dict[str, float]:
    """Return each label's highest score from any source, for one document."""
    pooled: dict[str, float] = {}
    for ranking in sources.values():
        for label, value in ranking.items():
            pooled[label] = max(pooled.get(label, 0.0), value)
    return pooled


def label_features(
    label: str,
    sources: Mapping[str, Ranking],
    source_names: Sequence[str],
    text: str,
    graph: LabelGraph,
    pooled: Ranking,
) -> list[float]:
    """Return one candidate label's signals for one document."""
    row: list[float] = []
    agreeing = 0
    for source in source_names:
        value = sources[source].get(label)
        row.extend([value if value is not None else 0.0, float(value is not None)])
        agreeing += int(value is not None)
    node = graph.taxonomy.nodes.get(label)
    parent, child = graph.family_support(label, pooled)
    row.extend(
        [
            float(agreeing),
            name_in_text(node.name, text) if node else 0.0,
            math.log1p(graph.counts.get(label, 0)),
            float(graph.taxonomy.tiers.get(label, 0)),
            graph.co_occurrence_support(label, pooled),
            parent,
            child,
        ]
    )
    if graph.priors is not None:
        row.extend(graph.priors.features(label))
    return row


def build_rows(
    documents: Sequence[Mapping[str, Ranking]],
    texts: Sequence[str],
    graph: LabelGraph,
    source_names: Sequence[str],
    per_source: int,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Return signals for every candidate of every document, and which (document, label) each row is."""
    rows: list[list[float]] = []
    keys: list[tuple[int, str]] = []
    for position, (sources, text) in enumerate(zip(documents, texts, strict=True)):
        lowered = " ".join(text.lower().split())
        pooled = pooled_scores(sources)
        for label in candidates_for(sources, per_source):
            rows.append(label_features(label, sources, source_names, lowered, graph, pooled))
            keys.append((position, label))
    width = len(feature_names(source_names, graph.priors is not None))
    return (np.array(rows, dtype=np.float32) if rows else np.zeros((0, width), dtype=np.float32)), keys


def train_combiner(
    rows: np.ndarray, keys: Sequence[tuple[int, str]], gold: Sequence[set[str]], settings: CombinedSettings
) -> HistGradientBoostingClassifier:
    """Train the model that weighs the signals, on documents the signal sources did not train on."""
    target = np.array([label in gold[position] for position, label in keys], dtype=np.int8)
    model = HistGradientBoostingClassifier(
        max_iter=settings.max_iterations, learning_rate=settings.learning_rate, random_state=settings.seed
    )
    model.fit(rows, target)
    return model


def scores_by_document(
    probabilities: np.ndarray, keys: Sequence[tuple[int, str]], count: int
) -> list[dict[str, float]]:
    """Group row-level probabilities back into one ranking per document."""
    results: list[dict[str, float]] = [{} for _ in range(count)]
    for value, (position, label) in zip(probabilities, keys, strict=True):
        results[position][label] = float(value)
    return results


def choose_threshold(
    gold: Sequence[set[str]], rankings: Sequence[Ranking], choices: Sequence[float]
) -> tuple[float, float]:
    """Return the score cut-off with the best micro F1 on the given documents, and that F1."""
    best_threshold, best_score = choices[0], -1.0
    for threshold in choices:
        hits = predicted = 0
        for labels, ranking in zip(gold, rankings, strict=True):
            kept = {label for label, value in ranking.items() if value >= threshold}
            hits += len(labels & kept)
            predicted += len(kept)
        total = sum(len(labels) for labels in gold)
        precision = hits / predicted if predicted else 0.0
        recall = hits / total if total else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold, best_score


def needs_llm(
    ranking: Ranking,
    band: tuple[float, float],
    counts: Mapping[str, int],
    rare_below: int,
    most: int | None = None,
) -> list[str]:
    """Return the labels worth an LLM call: those the model is unsure of, and rare ones with any support. Only
    the `most` strongest are kept, when a limit is given."""
    low, high = band
    wanted = [
        label
        for label, value in ranking.items()
        if low <= value < high or (counts.get(label, 0) < rare_below and value > 0)
    ]
    return sorted(sorted(wanted, key=lambda label: (-ranking[label], label))[:most])


SECOND_STAGE_FEATURES = ("first_stage_score", "asked_llm", "llm_rating", "log_training_count", "tier")


def second_stage_rows(
    rankings: Sequence[Ranking], ratings: Sequence[Ranking], graph: LabelGraph
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Return, for every candidate, the first model's score and what the LLM said of it, if it was asked."""
    rows: list[list[float]] = []
    keys: list[tuple[int, str]] = []
    for position, (ranking, rated) in enumerate(zip(rankings, ratings, strict=True)):
        for label, value in ranking.items():
            rating = rated.get(label)
            rows.append(
                [
                    value,
                    float(rating is not None),
                    rating if rating is not None else 0.0,
                    math.log1p(graph.counts.get(label, 0)),
                    float(graph.taxonomy.tiers.get(label, 0)),
                ]
            )
            keys.append((position, label))
    width = len(SECOND_STAGE_FEATURES)
    return (np.array(rows, dtype=np.float32) if rows else np.zeros((0, width), dtype=np.float32)), keys
