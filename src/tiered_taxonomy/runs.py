"""Set up and run each method on a test set. Notebooks call these and show the results."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tiered_taxonomy import datasets as ds
from tiered_taxonomy import experiment as ex
from tiered_taxonomy.llm import Client, ReplyStore, run_all
from tiered_taxonomy.methods import kahler as kh
from tiered_taxonomy.methods import llm_methods as lm
from tiered_taxonomy.methods import rollup as ru
from tiered_taxonomy.methods import tree_walk as tw
from tiered_taxonomy.methods.base import Prediction
from tiered_taxonomy.methods.combined import needs_llm
from tiered_taxonomy.metrics import candidate_recall, micro_f1, paired_difference, top_k
from tiered_taxonomy.models import make_embeddings, spec_from_config
from tiered_taxonomy.retrieval import EmbeddingStore, LabelIndex
from tiered_taxonomy.signals import by_id, few_shot_examples, load_embeddings, save_signal, similar_examples
from tiered_taxonomy.taxonomy import Taxonomy, with_definitions, without_implied_parents

Work = Callable[[ds.Item], Prediction]
ENTRY_TIER = {ex.EURLEX: 2, ex.HUMSET: 1, ex.HUMSET_V2: 1}


@dataclass
class Bench:
    """Everything a method needs: the test set, models, saved replies and label search."""

    test_set: ex.TestSet
    root: Path
    config: Mapping[str, Any]
    store: EmbeddingStore
    index: LabelIndex
    main: Client
    small: Client
    card_index: LabelIndex | None = None
    words: int = 500
    workers: int = 16
    records: list[ex.RunRecord] = field(default_factory=list)
    pipeline_keep: float | None = None

    @property
    def clients(self) -> list[Client]:
        """Return both chat clients."""
        return [self.main, self.small]

    def funnel_settings(self) -> lm.FunnelSettings:
        """Return the funnel's settings. HumSet's tree is too small to prune or to narrow."""
        funnel = self.config["funnel"]
        if ex.is_humset(self.test_set.name):
            size = len(self.test_set.taxonomy.nodes)
            return lm.FunnelSettings(size, size, 1, 1.0)
        return lm.FunnelSettings(
            funnel["retrieve_per_view"],
            funnel["keep_candidates"],
            funnel["prune_tier"],
            funnel["prune_keep_share"],
        )


def make_bench(test_set: ex.TestSet, root: Path, config: Mapping[str, Any]) -> Bench:
    """Build the models, the saved-reply store and the label search for a test set."""
    embeddings = config["embeddings"]
    model = make_embeddings(embeddings["provider"], embeddings["model"])
    store = EmbeddingStore(model, embeddings["model"], root / "embeddings.sqlite", embeddings["batch_size"])
    replies = ReplyStore(root / "replies.sqlite")
    main = Client(spec_from_config(config["models"]["main"]), replies)
    small = Client(spec_from_config(config["models"]["small"]), replies)
    return Bench(
        test_set,
        root,
        config,
        store,
        LabelIndex(test_set.taxonomy, store),
        main,
        small,
        words=config["text"]["input_words"],
        workers=config["workers"],
    )


def load_cards(bench: Bench) -> tuple[Taxonomy, int]:
    """Write label cards once, keep them in the data folder, and index labels by them.

    Returns the taxonomy with its definitions and how many labels have no card.
    """
    path = bench.root / "cards" / f"{bench.test_set.name}.json"
    taxonomy = bench.test_set.taxonomy
    if path.exists():
        cards = json.loads(path.read_text(encoding="utf-8"))
    else:
        settings = bench.config["label_cards"]
        client = bench.main.limited(settings["max_tokens"][bench.test_set.name])
        cards, _ = lm.make_label_cards(taxonomy, client, settings["batch_size"], bench.workers)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cards), encoding="utf-8")
    carded = with_definitions(taxonomy, cards)
    bench.card_index = LabelIndex(carded, bench.store)
    return carded, len(taxonomy.nodes) - len(cards)


def embed_inputs(bench: Bench, items: Sequence[ds.Item]) -> None:
    """Embed every item ahead of time, so that methods do not embed one text at a time."""
    bench.store.embed(sorted({ex.input_text(item, bench.words) for item in items}))


def retrieved_for(bench: Bench, text: str, index: LabelIndex) -> dict[str, float]:
    """Return the funnel's candidates for a text."""
    return lm.funnel_candidates(text, index, bench.store, bench.funnel_settings())


def _pipeline_work(bench: Bench, keep: float | None = None) -> Work:
    settings = bench.config["pipeline"]
    weights = (settings["ensemble_weight"], settings["relevance_weight"])
    if keep is None:
        keep = settings["keep_score"] if bench.pipeline_keep is None else bench.pipeline_keep
    threshold = keep

    def work(item: ds.Item) -> Prediction:
        text = ex.input_text(item, bench.words)
        return lm.pipeline_predict(
            text,
            bench.index,
            bench.store,
            [bench.main, bench.small],
            bench.main,
            settings["candidates_per_keyword"],
            weights,
            threshold,
        )

    return work


def _funnel_work(bench: Bench, index: LabelIndex) -> Work:
    settings = bench.funnel_settings()

    def work(item: ds.Item) -> Prediction:
        return lm.funnel_predict(ex.input_text(item, bench.words), index, bench.store, bench.main, settings)

    return work


def trained_ranked_path(root: Path, test_set: str) -> Path:
    """Return where the trained classifier's ranked labels for a sample are kept."""
    return root / "predictions" / test_set / "trained_ranked.json"


def save_trained_ranked(
    root: Path, test_set: str, ids: Sequence[str], ranked: Sequence[dict[str, float]]
) -> None:
    """Save the trained classifier's ranked labels, for the hybrid funnel to use as candidates."""
    path = trained_ranked_path(root, test_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(zip(ids, ranked, strict=True))), encoding="utf-8")


@dataclass(frozen=True)
class TrainedBaseline:
    """The flat classifier's choices on the development split, and its rankings of the sample."""

    k: int
    dev_f1: float
    train_seconds: float
    ranked: list[dict[str, float]]


def trained_baseline(
    root: Path,
    test_set: str,
    ids: Sequence[str],
    train: Callable[[], tuple[int, float, float, list[dict[str, float]]]],
) -> TrainedBaseline:
    """Return the flat classifier's choices and rankings, training it with `train` only if they are not saved.

    Omikuji, inside the classifier, starts its clustering from unseeded random centres, so training twice
    gives slightly different rankings, and the hybrid funnel's saved replies were asked with the first. So
    the rankings are kept like any finished step. Saved rankings that do not cover the sample raise."""
    choices_path = root / "predictions" / test_set / "trained_baseline.json"
    ranked_path = trained_ranked_path(root, test_set)
    if choices_path.exists() and ranked_path.exists():
        choices = json.loads(choices_path.read_text(encoding="utf-8"))
        saved = json.loads(ranked_path.read_text(encoding="utf-8"))
        missing = [item_id for item_id in ids if item_id not in saved]
        if missing:
            raise ValueError(
                f"{ranked_path} has no ranking for {len(missing)} sample items: delete it to retrain"
            )
        return TrainedBaseline(
            choices["k"], choices["dev_f1"], choices["train_seconds"], [saved[item_id] for item_id in ids]
        )
    k, dev_f1, seconds, ranked = train()
    save_trained_ranked(root, test_set, ids, ranked)
    choices_path.write_text(
        json.dumps({"k": k, "dev_f1": dev_f1, "train_seconds": seconds}, indent=1), encoding="utf-8"
    )
    return TrainedBaseline(k, dev_f1, seconds, ranked)


def _hybrid_work(bench: Bench, index: LabelIndex, ranked: Mapping[str, Mapping[str, float]]) -> Work:
    each = bench.config["funnel"]["hybrid_candidates_each"]

    def work(item: ds.Item) -> Prediction:
        text = ex.input_text(item, bench.words)
        retrieved = retrieved_for(bench, text, index)
        labels = lm.hybrid_candidates(ranked[item.id], retrieved, each)
        return lm.select_from_candidates(text, index.taxonomy, labels, bench.main)

    return work


def _select_all_work(bench: Bench) -> Work:
    labels = sorted(bench.test_set.taxonomy.nodes)

    def work(item: ds.Item) -> Prediction:
        return lm.select_from_all(
            ex.input_text(item, bench.words), bench.test_set.taxonomy, labels, bench.main
        )

    return work


def _walk_work(bench: Bench, kind: str) -> Work:
    walk_config = bench.config["tree_walk"]
    taxonomy = bench.test_set.taxonomy
    anchored = kind == "anchored"
    settings = tw.WalkSettings(
        max_branches=1 if kind == "strict" else walk_config["max_branches"],
        max_calls=walk_config["max_calls"],
        look_ahead_leaves=walk_config["look_ahead_leaves"] if anchored else 0,
    )

    def work(item: ds.Item) -> Prediction:
        text = ex.input_text(item, bench.words)
        if not anchored:
            return tw.walk(text, taxonomy, bench.main, settings)
        retrieved = retrieved_for(bench, text, bench.index)
        tier = ENTRY_TIER[bench.test_set.name]
        starts = tw.entry_points(taxonomy, retrieved, tier, walk_config["anchor_entries"])
        return tw.walk(text, taxonomy, bench.main, settings, starts, retrieved)

    return work


def method_table(bench: Bench) -> dict[str, Work]:
    """Return every model-using method available for a test set, by name."""
    table: dict[str, Work] = {
        "pipeline": _pipeline_work(bench),
        "funnel_names": _funnel_work(bench, bench.index),
        "walk_strict": _walk_work(bench, "strict"),
        "walk_branching": _walk_work(bench, "branching"),
        "walk_anchored": _walk_work(bench, "anchored"),
    }
    if bench.card_index is not None:
        table["funnel_cards"] = _funnel_work(bench, bench.card_index)
        ranked_path = trained_ranked_path(bench.root, bench.test_set.name)
        if bench.test_set.name == ex.EURLEX and ranked_path.exists():
            ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
            table["funnel_hybrid"] = _hybrid_work(bench, bench.card_index, ranked)
    if ex.is_humset(bench.test_set.name):
        table["select_all"] = _select_all_work(bench)
    return table


def tune_pipeline(bench: Bench, items: Sequence[ds.Item]) -> dict[str, Any]:
    """Choose the pipeline's score cut-off on development items, as the trained model's label count is
    chosen."""
    predictions = run_all(_pipeline_work(bench, keep=0.0), items, bench.workers)
    gold = ex.gold_exact(items)
    scored = {}
    for keep in bench.config["pipeline"]["keep_score_choices"]:
        kept = [
            without_implied_parents(label for label, value in p.labels.items() if value >= keep)
            for p in predictions
        ]
        scored[keep] = micro_f1(gold, kept)
    best = max(sorted(scored), key=lambda keep: scored[keep])
    bench.pipeline_keep = best
    return {"keep_score": best, "dev_f1": round(scored[best], 4), "items": len(items)}


def run_and_score(bench: Bench, name: str, items: Sequence[ds.Item]) -> dict[str, Any]:
    """Run one method, save its labels in the data folder, and return its scores and cost."""
    work = method_table(bench)[name]
    predictions, record = ex.run_method(name, work, items, bench.clients, bench.workers)
    labels = ex.label_sets(predictions)
    ex.save_predictions(bench.root, bench.test_set.name, name, [item.id for item in items], labels)
    bench.records.append(record)
    return {"scores": ex.evaluate(bench.test_set, items, labels), "run": record.__dict__}


def evidence_runs(bench: Bench, base: str, items: Sequence[ds.Item]) -> dict[str, Any]:
    """Compare a funnel's labels as they are, kept only if their own quote is real, and after a second
    check."""
    work = method_table(bench)[base]
    taxonomy = bench.test_set.taxonomy

    def checked(item: ds.Item) -> Prediction:
        return lm.evidence_check(ex.input_text(item, bench.words), work(item), taxonomy, bench.main)

    def quoted(item: ds.Item) -> Prediction:
        return lm.quote_filter(ex.input_text(item, bench.words), work(item))

    results: dict[str, Any] = {}
    for name, step in ((base, work), (f"{base}_quote_filter", quoted), (f"{base}_checked", checked)):
        predictions, record = ex.run_method(name, step, items, bench.clients, bench.workers)
        labels = ex.label_sets(predictions)
        ex.save_predictions(bench.root, bench.test_set.name, name, [item.id for item in items], labels)
        results[name] = {"scores": ex.evaluate(bench.test_set, items, labels), "run": record.__dict__}
    return results


def candidate_grid(
    bench: Bench, items: Sequence[ds.Item], index: LabelIndex, grid: Sequence[int]
) -> dict[str, dict[str, float]]:
    """Measure how many gold labels survive retrieval at each candidate count, with no model involved."""
    texts = [ex.input_text(item, bench.words) for item in items]
    vectors = bench.store.embed(texts)
    ranked = index.search_many(vectors, max(grid))
    gold = ex.gold_exact(items)
    counts = bench.test_set.train_counts
    cutoff = ex.rare_cutoff(counts)
    results: dict[str, dict[str, float]] = {}
    for k in grid:
        candidates = [top_k(item, k) for item in ranked]
        row = {"all": round(candidate_recall(gold, candidates), 4)}
        for band in ("common", "rare", "unseen"):
            banded = [
                {label for label in labels if ex.rarity(label, counts, cutoff) == band} for labels in gold
            ]
            if any(banded):
                row[band] = round(candidate_recall(banded, candidates), 4)
        results[str(k)] = row
    return results


def retrieval_labels(bench: Bench, items: Sequence[ds.Item], k: int) -> list[set[str]]:
    """Return the k nearest labels for each item, with no model involved."""
    vectors = bench.store.embed([ex.input_text(item, bench.words) for item in items])
    return [set(hits) for hits in bench.index.search_many(vectors, k)]


@dataclass
class RecipeTools:
    """What Kähler et al.'s two recipes need: their models, label search and the training documents."""

    generators: list[Client]
    ranker: Client
    search: kh.HybridLabelSearch
    settings: kh.RecipeSettings
    prompts: list[list[kh.Example]]
    train_vectors: np.ndarray
    similar_documents: int
    similar_generators: int

    @property
    def clients(self) -> list[Client]:
        """Return every model the recipes call."""
        return [*self.generators, self.ranker]


def make_recipe_tools(bench: Bench) -> RecipeTools:
    """Build the recipes' models from the configuration, and fix their few-shot prompts."""
    paper = bench.config["paper_models"]
    replies = ReplyStore(bench.root / "replies.sqlite")
    data = bench.test_set
    seed = bench.config["seed"]
    return RecipeTools(
        [Client(spec_from_config(entry), replies) for entry in paper["generators"]],
        Client(spec_from_config(paper["ranker"]), replies),
        kh.HybridLabelSearch(bench.index, bench.store),
        kh.RecipeSettings(),
        [
            few_shot_examples(data.train, data.taxonomy, size, seed + position)
            for position, size in enumerate(paper["few_shot_sizes"])
        ],
        load_embeddings(bench.root, data.name, "train"),
        paper["similar_documents"],
        paper["ki_fsprompt_generators"],
    )


def recipe_work(bench: Bench, tools: RecipeTools, recipe: str, split: str, items: Sequence[ds.Item]) -> Work:
    """Return the work for one item of one of the recipes. Similar documents are found with its stored
    embedding."""
    data = bench.test_set
    positions = {item.id: position for position, item in enumerate(getattr(data, split))}
    vectors = load_embeddings(bench.root, data.name, split)
    if set(item.id for item in items) - set(positions):
        raise ValueError(f"some items are not in the {split} split of {data.name}")

    def ensemble(item: ds.Item) -> Prediction:
        return kh.llm_ensemble(
            ex.input_text(item, bench.words),
            tools.generators,
            tools.prompts,
            tools.search,
            bench.index,
            tools.ranker,
            tools.settings,
        )

    def similar(item: ds.Item) -> Prediction:
        examples = similar_examples(
            vectors[positions[item.id]],
            tools.train_vectors,
            data.train,
            data.taxonomy,
            item.language,
            tools.similar_documents,
        )
        return kh.ki_fsprompt(
            ex.input_text(item, bench.words),
            examples,
            tools.generators[: tools.similar_generators],
            tools.search,
            bench.index,
            tools.ranker,
            tools.settings,
        )

    return {"llm_ensemble": ensemble, "ki_fsprompt": similar}[recipe]


def run_recipe(
    bench: Bench, tools: RecipeTools, recipe: str, split: str, items: Sequence[ds.Item]
) -> tuple[list[dict[str, float]], ex.RunRecord]:
    """Run a recipe over items, returning each item's label scores and what the run cost."""
    work = recipe_work(bench, tools, recipe, split, items)
    predictions, record = ex.run_method(recipe, work, items, tools.clients, bench.workers)
    return [dict(prediction.labels) for prediction in predictions], record


def recipe_signals(
    bench: Bench, tools: RecipeTools, recipe: str, split: str, items: Sequence[ds.Item]
) -> ex.RunRecord:
    """Run a recipe over items and save its label scores for the combiner, returning what the run cost."""
    ranked, record = run_recipe(bench, tools, recipe, split, items)
    save_signal(bench.root, bench.test_set.name, recipe, split, by_id(items, ranked))
    return record


def review_candidates(
    bench: Bench,
    items: Sequence[ds.Item],
    rankings: Sequence[Mapping[str, float]],
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, float]], ex.RunRecord]:
    """Ask the main model about the labels the combiner is unsure of, or that are rare, one call per item."""
    asked = {
        item.id: needs_llm(
            ranking,
            tuple(settings["llm_band"]),
            bench.test_set.train_counts,
            settings["rare_below"],
            settings["most_reviewed"],
        )
        for item, ranking in zip(items, rankings, strict=True)
    }

    def work(item: ds.Item) -> Prediction:
        return lm.rate_labels(
            ex.input_text(item, bench.words), bench.test_set.taxonomy, asked[item.id], bench.main
        )

    predictions, record = ex.run_method("combined_review", work, items, [bench.main], bench.workers)
    return [dict(prediction.labels) for prediction in predictions], record


def rollup_sample_eurlex(test_set: ex.TestSet, size: int, threshold: int, seed: int) -> list[ds.Item]:
    """Return long laws for the rollup test: all of the longest (20,000 words or more), then a random
    remainder."""
    lengths = np.array([ds.word_count(item.text) for item in test_set.test])
    longest = np.flatnonzero(lengths >= 20_000)
    others = np.flatnonzero((lengths >= threshold) & (lengths < 20_000))
    rng = np.random.default_rng(seed)
    extra = rng.choice(others, size=min(max(0, size - len(longest)), len(others)), replace=False)
    return [test_set.test[int(i)] for i in sorted([*longest.tolist()[:size], *extra.tolist()])]


def group_items(items: Sequence[ds.Item], min_entries: int) -> dict[str, list[ds.Item]]:
    """Group a test set's excerpts by source document, keeping documents with enough excerpts."""
    groups: dict[str, list[ds.Item]] = defaultdict(list)
    for item in items:
        groups[item.group].append(item)
    return {key: value for key, value in groups.items() if len(value) >= min_entries}


def merge_group(group_id: str, entries: Sequence[ds.Item], taxonomy: Taxonomy) -> ds.Item:
    """Make one document item from its excerpts. Its gold labels are the union of theirs."""
    labels = frozenset(label for entry in entries for label in entry.labels)
    tiers = sorted(set(taxonomy.tiers.values()))
    by_tier = {tier: frozenset(taxonomy.roll_up(labels, tier)) for tier in tiers}
    text = "\n\n".join(entry.text for entry in entries)
    return ds.Item(group_id, text, entries[0].language, labels, group_id, by_tier)


def rollup_rules(config: Mapping[str, Any]) -> dict[str, ru.Rule]:
    """Return the rollup rules being compared."""
    return {
        "any_chunk": ru.any_chunk,
        "share_of_chunks": ru.share_of_chunks(config["majority_share"]),
        "certainty_weighted": ru.certainty_weighted(config["certainty_floor"]),
    }


def rollup_compare(bench: Bench, documents: Sequence[tuple[ds.Item, list[str]]]) -> dict[str, Any]:
    """Label each document's chunks, then compare rollup rules and reading the opening
    only."""
    index = bench.card_index or bench.index
    settings = bench.funnel_settings()
    chunks = [(i, text) for i, (_, texts) in enumerate(documents) for text in texts]
    bench.store.embed(sorted({text for _, text in chunks}))

    def label_chunk(entry: tuple[int, str]) -> Prediction:
        return lm.funnel_predict(entry[1], index, bench.store, bench.main, settings)

    costs_before = bench.main.cost()
    per_chunk = run_all(label_chunk, chunks, bench.workers)
    by_document: dict[int, list[Prediction]] = defaultdict(list)
    for (i, _), prediction in zip(chunks, per_chunk, strict=True):
        by_document[i].append(prediction)
    items = [item for item, _ in documents]
    results: dict[str, Any] = {
        "chunks": len(chunks),
        "documents": len(documents),
        "cost_usd": round(bench.main.cost() - costs_before, 4),
        "malformed_chunks": sum(prediction.malformed for prediction in per_chunk),
    }
    predicted = {
        name: [without_implied_parents(rule(by_document[i])) for i in range(len(documents))]
        for name, rule in rollup_rules(bench.config["rollup"]).items()
    }
    predicted["first_chunk_only"] = [
        without_implied_parents(by_document[i][0].labels) if by_document[i] else set()
        for i in range(len(documents))
    ]
    for name, labels in predicted.items():
        results[name] = ex.evaluate(bench.test_set, items, labels)
    gold = ex.gold_exact(items)
    base = predicted["certainty_weighted"]
    results["differences"] = {
        f"certainty_weighted minus {other}": [
            round(value, 4) for value in paired_difference(micro_f1, gold, base, predicted[other])
        ]
        for other in ("any_chunk", "share_of_chunks", "first_chunk_only")
    }
    return results
