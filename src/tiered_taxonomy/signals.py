"""Each method's scores per label, saved in the data folder so that the combined method can use them all."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy import datasets as ds
from tiered_taxonomy import experiment as ex
from tiered_taxonomy import gpu_jobs as gj
from tiered_taxonomy.corpus_embeddings import nearest_label_scores
from tiered_taxonomy.methods import combined as cb
from tiered_taxonomy.methods import kahler as kh
from tiered_taxonomy.methods.base import Prediction
from tiered_taxonomy.methods.embedding_classifier import ClassifierSettings, EmbeddingClassifier
from tiered_taxonomy.methods.trained import Cut, TrainedClassifier, choose_cut
from tiered_taxonomy.retrieval import LabelIndex
from tiered_taxonomy.taxonomy import Taxonomy

Rankings = dict[str, dict[str, float]]
RANK_DEPTH = 100
# GPU jobs train on raw labels, a few of which have no place in the tree. More than this means the wrong data.
MAX_OUTSIDE_SHARE = 0.01
NEWER_LAWS_SUFFIX = "{}_newer"
NEIGHBOUR_BATCH = 512
TRAINING_NEIGHBOURS = 20


def signal_path(root: Path, test_set: str, source: str, split: str) -> Path:
    """Return where one source's rankings for one split are kept."""
    return root / "signals" / test_set / f"{source}_{split}.json"


def save_signal(root: Path, test_set: str, source: str, split: str, rankings: Rankings) -> Path:
    """Save one source's rankings."""
    path = signal_path(root, test_set, source, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rankings), encoding="utf-8")
    return path


def read_rankings(path: Path) -> Rankings:
    """Read rankings from a file."""
    rankings: Rankings = json.loads(path.read_text(encoding="utf-8"))
    return rankings


def load_signal(root: Path, test_set: str, source: str, split: str) -> Rankings:
    """Read one source's rankings. A missing file raises."""
    rankings: Rankings = json.loads(signal_path(root, test_set, source, split).read_text(encoding="utf-8"))
    return rankings


def by_id(items: Sequence[ds.Item], rankings: Sequence[dict[str, float]]) -> Rankings:
    """Key rankings by item id."""
    return {item.id: ranking for item, ranking in zip(items, rankings, strict=True)}


def fetch_xr_rankings(root: Path, test_set: str, split: str) -> Rankings:
    """Download XR-Transformer's rankings from the jobs dataset, at the commit recorded in the data folder."""
    return read_rankings(gj.fetch(root, f"xr_transformer_{test_set}_{split}.json"))


def drop_outside_tree(rankings: Rankings, taxonomy: Taxonomy) -> tuple[Rankings, set[str]]:
    """Remove labels with no place in the tree, which gold labels never include, and return them for
    reporting. Raises if there are so many that the rankings cannot belong to this test set."""
    seen = {label for ranking in rankings.values() for label in ranking}
    outside = seen - set(taxonomy.nodes)
    if seen and len(outside) / len(seen) > MAX_OUTSIDE_SHARE:
        raise ValueError(f"{len(outside)} of {len(seen)} labels are not in this taxonomy")
    kept = {
        item_id: {label: value for label, value in ranking.items() if label not in outside}
        for item_id, ranking in rankings.items()
    }
    return kept, outside


def load_embeddings(root: Path, test_set: str, split: str) -> np.ndarray:
    """Read the document embeddings for a split, written by scripts/embed_corpus.py."""
    return np.load(root / "corpus_embeddings" / test_set / f"{split}.npy")


def neighbour_rankings(
    queries: np.ndarray, documents: np.ndarray, train: Sequence[ds.Item], neighbours: int
) -> list[dict[str, float]]:
    """Give each query the labels of its most similar training documents."""
    return nearest_label_scores(queries, documents, [item.labels for item in train], neighbours)


def strongest(ranking: Mapping[str, float], count: int = RANK_DEPTH) -> dict[str, float]:
    """Return a ranking's highest-scoring labels, so that saved rankings stay a sensible size."""
    return dict(sorted(ranking.items(), key=lambda pair: (-pair[1], pair[0]))[:count])


def tuning_cut(
    items: Sequence[ds.Item], rankings: Rankings, limits: Sequence[int], thresholds: Sequence[float]
) -> tuple[Cut, float]:
    """Choose how many labels to keep, and from what score, on development items."""
    return choose_cut(ex.gold_exact(items), [rankings[item.id] for item in items], limits, thresholds)


def omikuji_signals(data: ex.TestSet, root: Path, max_features: int, words: int) -> float:
    """Train Omikuji, save its rankings for the development and test splits, and return the seconds taken."""
    started = time.monotonic()
    model = TrainedClassifier(max_features)
    model.fit([ex.input_text(item, words) for item in data.train], ex.gold_exact(data.train))
    for split, items in (("dev", data.dev), ("test", data.test)):
        ranked = model.rank([ex.input_text(item, words) for item in items], top_k=RANK_DEPTH)
        save_signal(root, data.name, "omikuji", split, by_id(items, ranked))
    return round(time.monotonic() - started, 1)


def embedding_signals(
    data: ex.TestSet,
    root: Path,
    tuning: Sequence[ds.Item],
    positions: Mapping[str, int],
    rates: Sequence[float],
    settings: ClassifierSettings,
    limits: Sequence[int],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Train the embedding classifier at each learning rate, keep the best on development items and save its
    rankings."""
    train = load_embeddings(root, data.name, "train")
    dev = load_embeddings(root, data.name, "dev")
    test = load_embeddings(root, data.name, "test")
    started = time.monotonic()
    best: EmbeddingClassifier | None = None
    best_value = -1.0
    tried: dict[str, float] = {}
    for rate in rates:
        model = EmbeddingClassifier(replace(settings, learning_rate=rate))
        model.fit(train, ex.gold_exact(data.train))
        ranked = model.rank(dev[[positions[item.id] for item in tuning]], RANK_DEPTH)
        _, value = choose_cut(ex.gold_exact(tuning), ranked, limits, thresholds)
        tried[str(rate)] = round(value, 4)
        if value > best_value:
            best, best_value = model, value
    if best is None:
        raise ValueError("no learning rates were given")
    for split, vectors, items in (("dev", dev, data.dev), ("test", test, data.test)):
        ranked = best.rank(vectors, RANK_DEPTH)
        save_signal(root, data.name, "embedding_classifier", split, by_id(items, ranked))
    return {"dev_f1_by_learning_rate": tried, "seconds": round(time.monotonic() - started, 1)}


def knn_signals(
    data: ex.TestSet,
    root: Path,
    tuning: Sequence[ds.Item],
    positions: Mapping[str, int],
    choices: Sequence[int],
    limits: Sequence[int],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Choose how many similar training documents to use on development items, and save their labels as
    rankings."""
    train = load_embeddings(root, data.name, "train")
    dev = load_embeddings(root, data.name, "dev")
    test = load_embeddings(root, data.name, "test")
    queries = dev[[positions[item.id] for item in tuning]]
    tried: dict[str, float] = {}
    for neighbours in choices:
        ranked = neighbour_rankings(queries, train, data.train, neighbours)
        tried[str(neighbours)] = round(choose_cut(ex.gold_exact(tuning), ranked, limits, thresholds)[1], 4)
    chosen = int(max(tried, key=lambda key: tried[key]))
    for split, vectors, items in (("dev", dev, data.dev), ("test", test, data.test)):
        ranked = neighbour_rankings(vectors, train, data.train, chosen)
        save_signal(root, data.name, "knn", split, by_id(items, [strongest(entry) for entry in ranked]))
    return {"neighbours": chosen, "dev_f1_by_neighbours": tried}


def label_search_signals(data: ex.TestSet, root: Path, index: LabelIndex) -> None:
    """Save the labels nearest to each document, using the same model's embeddings of both."""
    for split, items in (("dev", data.dev), ("test", data.test)):
        hits = index.search_many(load_embeddings(root, data.name, split), RANK_DEPTH)
        save_signal(root, data.name, "label_search", split, by_id(items, hits))


def xr_signals(data: ex.TestSet, root: Path) -> set[str]:
    """Fetch XR-Transformer's rankings from its job's output, raise if they do not fit this test set, and
    return the labels dropped for having no place in the tree."""
    dropped: set[str] = set()
    for split, items in (("dev", data.dev), ("test", data.test)):
        rankings = fetch_xr_rankings(root, data.name, split)
        missing = {item.id for item in items} - set(rankings)
        if missing:
            raise ValueError(f"XR-Transformer has no ranking for {len(missing)} {split} items of {data.name}")
        kept, outside = drop_outside_tree(rankings, data.taxonomy)
        dropped |= outside
        save_signal(root, data.name, "xr_transformer", split, kept)
    return dropped


def collect_signals(
    data: ex.TestSet,
    root: Path,
    settings: Mapping[str, Any],
    seed: int,
    words: int,
    device: str,
    index: LabelIndex,
) -> dict[str, Any]:
    """Train each model that runs here and save its rankings. Finished steps are not repeated, as they are
    noted in the data folder; delete that folder's `signals` to start again."""
    path = root / "signals" / data.name / "details.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    details: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    tuning = ex.pilot_subset(data.dev, settings["tuning_items"], seed)
    positions = {item.id: position for position, item in enumerate(data.dev)}
    limits, thresholds = settings["cut_limits"], settings["cut_thresholds"]
    embedding = settings["embedding_classifier"]
    classifier = ClassifierSettings(embedding["epochs"], embedding["batch_size"], seed=seed, device=device)

    def label_search() -> dict[str, Any]:
        label_search_signals(data, root, index)
        return {"finished": True}

    steps = {
        "omikuji": lambda: {"seconds": omikuji_signals(data, root, settings["max_features"], words)},
        "embedding_classifier": lambda: embedding_signals(
            data, root, tuning, positions, embedding["learning_rates"], classifier, limits, thresholds
        ),
        "knn": lambda: knn_signals(
            data, root, tuning, positions, settings["neighbour_choices"], limits, thresholds
        ),
        "label_search": label_search,
    }
    for name, step in steps.items():
        if name not in details:
            details[name] = step()
            path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def newer_laws_view(data: ex.TestSet, root: Path) -> ex.TestSet:
    """Return MultiEURLEX as if the earlier half of its development laws, by date, were training laws, and
    save the matching embeddings under the view's name, so that the same code can train on it."""
    early, late = bm.split_by_date(data.dev)
    view = replace(
        data,
        name=NEWER_LAWS_SUFFIX.format(data.name),
        train=[*data.train, *early],
        dev=late,
        train_counts=Counter(label for item in [*data.train, *early] for label in item.labels),
    )
    train = load_embeddings(root, data.name, "train")
    dev = load_embeddings(root, data.name, "dev")
    test = load_embeddings(root, data.name, "test")
    positions = {item.id: position for position, item in enumerate(data.dev)}
    folder = root / "corpus_embeddings" / view.name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "train.npy", np.vstack([train, dev[[positions[item.id] for item in early]]]))
    np.save(folder / "dev.npy", dev[[positions[item.id] for item in late]])
    np.save(folder / "test.npy", test)
    return view


def candidate_lists(
    sources: Mapping[str, Rankings], ids: Sequence[str], per_source: int
) -> dict[str, list[str]]:
    """Return, per item, the union of each source's strongest labels. An item no source ranks gets no list."""
    lists: dict[str, list[str]] = {}
    for item_id in ids:
        found: set[str] = set()
        for rankings in sources.values():
            found.update(strongest(rankings.get(item_id, {}), per_source))
        if found:
            lists[item_id] = sorted(found)
    return lists


def training_candidates(
    data: ex.TestSet, root: Path, index: LabelIndex, items: int, per_source: int, seed: int
) -> dict[str, list[str]]:
    """Return wrong-answer candidates for a sample of training items, from similar other documents and label
    search. An item is never its own neighbour, so these are as hard as the candidates it will meet later."""
    vectors = load_embeddings(root, data.name, "train")
    picks = np.random.default_rng(seed).choice(len(data.train), min(items, len(data.train)), replace=False)
    lists: dict[str, list[str]] = {}
    for start in range(0, len(picks), NEIGHBOUR_BATCH):
        chunk = picks[start : start + NEIGHBOUR_BATCH]
        scores = vectors[chunk] @ vectors.T
        scores[np.arange(len(chunk)), chunk] = -np.inf
        nearest = np.argpartition(-scores, TRAINING_NEIGHBOURS, axis=1)[:, :TRAINING_NEIGHBOURS]
        for position, row in zip(chunk, nearest, strict=True):
            found: Counter[str] = Counter(label for i in row for label in data.train[int(i)].labels)
            searched = index.search(vectors[int(position)], per_source)
            lists[data.train[int(position)].id] = sorted(
                set(most_frequent(found, per_source)) | set(searched)
            )
    return lists


def most_frequent(counts: Counter[str], limit: int) -> list[str]:
    """Return the most frequent labels, breaking ties by label. `Counter.most_common` breaks them by insertion
    order, which follows set order and so changes from one process to the next."""
    return sorted(counts, key=lambda label: (-counts[label], label))[:limit]


def label_names(taxonomy: Taxonomy) -> dict[str, str]:
    """Return each label's readable path through the tree, which is what the reranker reads."""
    return {node: taxonomy.path(node) for node in taxonomy.nodes}


def few_shot_examples(train: Sequence[ds.Item], taxonomy: Taxonomy, size: int, seed: int) -> list[kh.Example]:
    """Return fixed few-shot examples drawn from training items that have labels."""
    usable = [item for item in train if item.labels]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(usable), size=min(size, len(usable)), replace=False)
    return [_example(usable[int(i)], taxonomy) for i in sorted(picks.tolist())]


def _example(item: ds.Item, taxonomy: Taxonomy) -> kh.Example:
    names = sorted(taxonomy.nodes[label].name for label in item.labels if label in taxonomy.nodes)
    return kh.Example(item.text, tuple(names))


def similar_examples(
    query: np.ndarray,
    documents: np.ndarray,
    train: Sequence[ds.Item],
    taxonomy: Taxonomy,
    language: str,
    count: int,
) -> list[kh.Example]:
    """Return examples from the training items most similar to a text, in its language and with labels."""
    positions = kh.similar_positions(query, documents, count * 5)
    usable = [train[i] for i in positions if train[i].labels and train[i].language == language]
    return [_example(item, taxonomy) for item in usable[:count]]


def rankings_from_predictions(predictions: Sequence[Prediction]) -> list[dict[str, float]]:
    """Return each prediction's label scores."""
    return [dict(prediction.labels) for prediction in predictions]


def assemble(
    items: Sequence[ds.Item], sources: Mapping[str, Rankings], source_names: Sequence[str]
) -> list[dict[str, dict[str, float]]]:
    """Return, per item, its ranking from every source. An item a source has no ranking for gets an empty
    one."""
    return [{name: sources[name].get(item.id, {}) for name in source_names} for item in items]


def combine_and_score(
    fit_items: Sequence[ds.Item],
    fit_sources: Mapping[str, Rankings],
    tune_items: Sequence[ds.Item],
    tune_sources: Mapping[str, Rankings],
    source_names: Sequence[str],
    graph: cb.LabelGraph,
    words: int,
    settings: cb.CombinedSettings,
) -> dict[str, Any]:
    """Train the combiner on one set of items and choose its score cut-off on another."""
    fit_rows, fit_keys = cb.build_rows(
        assemble(fit_items, fit_sources, source_names),
        [ds.first_words(item.text, words) for item in fit_items],
        graph,
        source_names,
        settings.per_source,
    )
    model = cb.train_combiner(fit_rows, fit_keys, [set(item.labels) for item in fit_items], settings)
    tuned = apply_combiner(model, tune_items, tune_sources, source_names, graph, words, settings)
    choices = [round(value, 2) for value in np.arange(0.05, 0.9, 0.05)]
    threshold, tuned_f1 = cb.choose_threshold([set(item.labels) for item in tune_items], tuned, choices)
    return {"model": model, "threshold": threshold, "tune_f1": tuned_f1, "rows": len(fit_rows)}


def apply_combiner(
    model: Any,
    items: Sequence[ds.Item],
    sources: Mapping[str, Rankings],
    source_names: Sequence[str],
    graph: cb.LabelGraph,
    words: int,
    settings: cb.CombinedSettings,
) -> list[dict[str, float]]:
    """Score every candidate label of every item with a trained combiner."""
    rows, keys = cb.build_rows(
        assemble(items, sources, source_names),
        [ds.first_words(item.text, words) for item in items],
        graph,
        source_names,
        settings.per_source,
    )
    probabilities = model.predict_proba(rows)[:, 1] if len(rows) else np.zeros(0)
    return cb.scores_by_document(probabilities, keys, len(items))


def load_sources(root: Path, test_set: str, names: Sequence[str], split: str) -> dict[str, Rankings]:
    """Read several sources' rankings for one split. A missing source raises."""
    return {name: load_signal(root, test_set, name, split) for name in names}


def split_for_combining(
    items: Sequence[ds.Item], fit: int, review: int, cut: int, seed: int
) -> tuple[list[ds.Item], list[ds.Item], list[ds.Item]]:
    """Divide development items, with no overlap, into those that train the combiner, those that train its
    second stage, and those that choose the final score cut-off."""
    if fit + review + cut > len(items):
        raise ValueError(f"asked for {fit + review + cut} development items but there are {len(items)}")
    order = np.random.default_rng(seed).permutation(len(items))
    chosen = [items[int(i)] for i in order]
    return chosen[:fit], chosen[fit : fit + review], chosen[fit + review : fit + review + cut]


def second_stage(
    fit_rankings: Sequence[Mapping[str, float]],
    fit_ratings: Sequence[Mapping[str, float]],
    fit_gold: Sequence[set[str]],
    graph: cb.LabelGraph,
    settings: cb.CombinedSettings,
) -> Any:
    """Train the model that revises the combiner's scores in light of the LLM's ratings."""
    rows, keys = cb.second_stage_rows(fit_rankings, fit_ratings, graph)
    return cb.train_combiner(rows, keys, list(fit_gold), settings)


def apply_second_stage(
    model: Any,
    rankings: Sequence[Mapping[str, float]],
    ratings: Sequence[Mapping[str, float]],
    graph: cb.LabelGraph,
) -> list[dict[str, float]]:
    """Return each item's revised label scores."""
    rows, keys = cb.second_stage_rows(rankings, ratings, graph)
    probabilities = model.predict_proba(rows)[:, 1] if len(rows) else np.zeros(0)
    return cb.scores_by_document(probabilities, keys, len(rankings))


def signal_importance(
    model: Any, rows: np.ndarray, target: np.ndarray, names: Sequence[str], seed: int
) -> dict[str, float]:
    """Return how much the combiner's ranking of candidates depends on each signal, by shuffling one at a
    time. It is measured by average precision, as accuracy says little when most candidates are wrong."""
    rng = np.random.default_rng(seed)
    baseline = float(average_precision_score(target, model.predict_proba(rows)[:, 1]))
    importance = {}
    for column, name in enumerate(names):
        shuffled = rows.copy()
        shuffled[:, column] = rng.permutation(shuffled[:, column])
        value = float(average_precision_score(target, model.predict_proba(shuffled)[:, 1]))
        importance[name] = round(baseline - value, 5)
    return importance
