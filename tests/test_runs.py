"""Tests for setting up and running methods, with invented models so nothing is sent or paid for."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from tiered_taxonomy import experiment as ex
from tiered_taxonomy import runs
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.llm import Client, ReplyStore
from tiered_taxonomy.models import ModelSpec
from tiered_taxonomy.retrieval import EmbeddingStore, LabelIndex
from tiered_taxonomy.signals import load_signal
from tiered_taxonomy.taxonomy import from_paths

TREE = from_paths(["Health->Water", "Health->Clinics", "Context->Economy"])
CONFIG: dict[str, Any] = {
    "funnel": {"retrieve_per_view": 3, "keep_candidates": 3, "prune_tier": 1, "prune_keep_share": 1.0},
    "pipeline": {
        "ensemble_weight": 0.3,
        "relevance_weight": 0.7,
        "candidates_per_keyword": 2,
        "keep_score": 0.5,
        "keep_score_choices": [0.1, 0.9],
    },
    "tree_walk": {"max_branches": 2, "max_calls": 5, "anchor_entries": 2, "look_ahead_leaves": 1},
    "rollup": {"majority_share": 0.5, "certainty_floor": 0.5},
    "label_cards": {"batch_size": 10, "max_tokens": {ex.HUMSET: 4000, ex.EURLEX: 4000}},
}


class CountEmbeddings(Embeddings):
    """Embeds by counting a few words."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        words = ("water", "clinics", "economy")
        return [[float(text.lower().count(word)) + 0.01 for word in words] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class Always:
    """A chat model that always answers the same way."""

    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, _messages: object) -> AIMessage:
        return AIMessage(content=self.content)


def item(item_id: str, labels: set[str], text: str = "water", group: str = "") -> Item:
    by_tier = {tier: frozenset(TREE.roll_up(labels, tier)) for tier in (1, 2)}
    return Item(item_id, text, "en", frozenset(labels), group or item_id, by_tier)


def bench(tmp_path: Path, content: str, name: str = ex.HUMSET) -> runs.Bench:
    test_set = ex.TestSet(name, TREE, (1, 2), [], [], [item("1", {"Health->Water"})], Counter())
    store = EmbeddingStore(CountEmbeddings(), "fake", tmp_path / "v.sqlite")
    replies = ReplyStore(tmp_path / "r.sqlite")
    model: Any = Always(content)
    client = Client(ModelSpec("fake", "m"), replies, model)
    return runs.Bench(
        test_set, tmp_path, CONFIG, store, LabelIndex(TREE, store), client, client, words=50, workers=1
    )


SELECT = '{"labels": [{"id": "Health->Water", "confidence": 0.9, "quote": "water"}]}'


def test_method_table_lists_small_taxonomy_methods_only_for_humset(tmp_path: Path) -> None:
    assert "select_all" in runs.method_table(bench(tmp_path, SELECT))
    assert "select_all" not in runs.method_table(bench(tmp_path, SELECT, ex.EURLEX))


def test_funnel_settings_do_not_narrow_humset(tmp_path: Path) -> None:
    settings = bench(tmp_path, SELECT).funnel_settings()
    assert settings.keep_candidates == len(TREE.nodes)
    assert bench(tmp_path, SELECT, ex.EURLEX).funnel_settings().keep_candidates == 3


def test_run_and_score_saves_labels_and_scores_them(tmp_path: Path) -> None:
    current = bench(tmp_path, SELECT)
    items = current.test_set.test
    runs.embed_inputs(current, items)
    result = runs.run_and_score(current, "funnel_names", items)
    assert result["scores"]["exact"]["f1"] == 1.0
    assert result["run"]["calls"] == 1
    assert ex.load_predictions(tmp_path, ex.HUMSET, "funnel_names", ["1"]) == [{"Health->Water"}]


@pytest.mark.parametrize("name", ["walk_strict", "walk_branching", "walk_anchored", "select_all", "pipeline"])
def test_each_method_runs(tmp_path: Path, name: str) -> None:
    current = bench(tmp_path, '{"selected": [], "labels": [], "keywords": ["water"], "ratings": []}')
    result = runs.run_and_score(current, name, current.test_set.test)
    assert result["run"]["items"] == 1


def test_tune_pipeline_chooses_the_cut_off_on_the_items_given(tmp_path: Path) -> None:
    content = '{"keywords": ["water"], "ratings": [{"id": "Health->Water", "score": 10}]}'
    current = bench(tmp_path, content)
    chosen = runs.tune_pipeline(current, current.test_set.test)
    assert chosen["keep_score"] in (0.1, 0.9)
    assert current.pipeline_keep == chosen["keep_score"]
    assert chosen["items"] == 1


def test_evidence_runs_compare_three_versions(tmp_path: Path) -> None:
    content = (
        '{"labels": [{"id": "Health->Water", "confidence": 0.9, "quote": "water"}],'
        ' "checks": [{"id": "Health->Water", "supported": false, "quote": ""}]}'
    )
    current = bench(tmp_path, content)
    results = runs.evidence_runs(current, "funnel_names", current.test_set.test)
    assert results["funnel_names"]["scores"]["exact"]["f1"] == 1.0
    assert results["funnel_names_quote_filter"]["scores"]["exact"]["f1"] == 1.0
    assert results["funnel_names_checked"]["scores"]["exact"]["f1"] == 0.0


def test_candidate_grid_grows_with_the_candidate_count(tmp_path: Path) -> None:
    current = bench(tmp_path, SELECT)
    # Three labels given once make the least-used quarter, so the label given 50 times is common.
    current.test_set.train_counts = Counter({"Health->Water": 50, "a": 1, "b": 1, "c": 1})
    grid = runs.candidate_grid(current, current.test_set.test, current.index, [1, 3])
    assert grid["1"]["all"] <= grid["3"]["all"]
    assert grid["3"]["common"] == 1.0
    assert runs.retrieval_labels(current, current.test_set.test, 1) == [{"Health->Water"}]


def test_load_cards_writes_them_once(tmp_path: Path) -> None:
    current = bench(tmp_path, '{"cards": [{"id": "Health", "definition": "Care."}]}')
    carded, missing = runs.load_cards(current)
    assert carded.nodes["Health"].definition == "Care."
    assert missing == len(TREE.nodes) - 1
    assert current.card_index is not None
    assert "funnel_cards" in runs.method_table(current)
    saved = json.loads((tmp_path / "cards" / f"{ex.HUMSET}.json").read_text(encoding="utf-8"))
    assert saved == {"Health": "Care."}
    assert runs.load_cards(current)[1] == missing


def test_hybrid_method_appears_once_ranked_labels_exist(tmp_path: Path) -> None:
    current = bench(tmp_path, SELECT, ex.EURLEX)
    runs.load_cards(current)
    assert "funnel_hybrid" not in runs.method_table(current)
    runs.save_trained_ranked(tmp_path, ex.EURLEX, ["1"], [{"Health->Water": 0.9}])
    CONFIG["funnel"]["hybrid_candidates_each"] = 1
    result = runs.run_and_score(current, "funnel_hybrid", current.test_set.test)
    assert result["scores"]["exact"]["f1"] == 1.0


def test_rollup_sample_takes_the_longest_first() -> None:
    long_text = " ".join(["w"] * 20_000)
    mid_text = " ".join(["w"] * 6_000)
    test = [
        item("a", set(), long_text),
        item("b", set(), mid_text),
        item("c", set(), mid_text),
        item("d", set(), "w"),
    ]
    data = ex.TestSet(ex.EURLEX, TREE, (1, 2), [], [], test, Counter())
    assert runs.rollup_sample_eurlex(data, 2, 5000, 1)[0].id == "a"
    assert len(runs.rollup_sample_eurlex(data, 2, 5000, 1)) == 2
    assert "d" not in {entry.id for entry in runs.rollup_sample_eurlex(data, 10, 5000, 1)}


def test_group_items_and_merge_group() -> None:
    entries = [
        item("1", {"Health->Water"}, group="L"),
        item("2", {"Context->Economy"}, group="L"),
        item("3", set()),
    ]
    groups = runs.group_items(entries, 2)
    assert list(groups) == ["L"]
    merged = runs.merge_group("L", groups["L"], TREE)
    assert merged.labels == {"Health->Water", "Context->Economy"}
    assert merged.labels_by_tier[1] == {"Health", "Context"}
    assert "water" in merged.text


def test_rollup_compare_scores_every_rule(tmp_path: Path) -> None:
    current = bench(tmp_path, SELECT)
    document = item("d", {"Health->Water"})
    results = runs.rollup_compare(current, [(document, ["water here", "water there"])])
    assert results["chunks"] == 2
    assert results["any_chunk"]["exact"]["f1"] == 1.0
    assert results["share_of_chunks"]["exact"]["f1"] == 1.0
    assert results["certainty_weighted"]["exact"]["f1"] == 1.0
    assert results["first_chunk_only"]["exact"]["f1"] == 1.0
    assert results["differences"]["certainty_weighted minus any_chunk"] == [0.0, 0.0, 0.0]


def recipe_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[runs.Bench, runs.RecipeTools]:
    current = bench(tmp_path, RECIPE_REPLY)
    monkeypatch.setattr(runs, "Client", lambda spec, replies: Client(spec, replies, Always(RECIPE_REPLY)))
    train = [item("t1", {"Health->Water"}, "water supply"), item("t2", {"Context->Economy"}, "economy")]
    current.test_set = ex.TestSet(
        ex.HUMSET, TREE, (1, 2), train, [item("d1", {"Health->Water"})], [], Counter()
    )
    folder = tmp_path / "corpus_embeddings" / ex.HUMSET
    folder.mkdir(parents=True)
    np.save(folder / "train.npy", np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.save(folder / "dev.npy", np.array([[1.0, 0.0]]))
    config = {
        "seed": 1,
        "paper_models": {
            "generators": [{"provider": "fake", "model": "g"}, {"provider": "fake", "model": "h"}],
            "ki_fsprompt_generators": 1,
            "ranker": {"provider": "fake", "model": "r", "json_mode": False},
            "few_shot_sizes": [1, 2],
            "similar_documents": 1,
        },
    }
    current.config = {**CONFIG, **config}
    return current, runs.make_recipe_tools(current)


RECIPE_REPLY = '{"keywords": ["water"], "ratings": [{"id": "Health->Water", "score": 10}]}'


@pytest.mark.parametrize(("recipe", "calls"), [("llm_ensemble", 5), ("ki_fsprompt", 2)])
def test_recipes_call_each_model_the_expected_number_of_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe: str, calls: int
) -> None:
    current, tools = recipe_bench(tmp_path, monkeypatch)
    rankings, record = runs.run_recipe(current, tools, recipe, "dev", current.test_set.dev)
    assert record.calls == calls
    assert next(iter(rankings[0])) == "Health->Water"


def test_recipe_signals_are_saved_by_item_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current, tools = recipe_bench(tmp_path, monkeypatch)
    items = current.test_set.dev[:1]
    record = runs.recipe_signals(current, tools, "ki_fsprompt", "dev", items)
    saved = load_signal(current.root, current.test_set.name, "ki_fsprompt", "dev")
    assert set(saved) == {items[0].id}
    assert record.items == 1


def test_a_recipe_refuses_items_from_another_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current, tools = recipe_bench(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not in the dev split"):
        runs.recipe_work(current, tools, "llm_ensemble", "dev", [item("other", set())])


def test_review_only_asks_about_labels_in_need_of_it(tmp_path: Path) -> None:
    reply = '{"ratings": [{"id": "Health->Water", "score": 7}]}'
    current = bench(tmp_path, reply)
    items = [item("1", {"Health->Water"}), item("2", set())]
    rankings = [{"Health->Water": 0.5, "Health": 0.99}, {"Health": 0.99}]
    current.test_set.train_counts.update({"Health->Water": 100, "Health": 100})
    settings = {"llm_band": [0.1, 0.6], "rare_below": 10, "most_reviewed": 5}
    ratings, record = runs.review_candidates(current, items, rankings, settings)
    assert ratings == [{"Health->Water": 0.7}, {}]
    assert record.calls == 1


def test_the_trained_baseline_is_trained_once_and_its_rankings_reused(tmp_path: Path) -> None:
    calls = []

    def train() -> tuple[int, float, float, list[dict[str, float]]]:
        calls.append(1)
        return 2, 0.5, 9.0, [{"Health": 0.9}, {"Water": 0.4}]

    first = runs.trained_baseline(tmp_path, ex.EURLEX, ["a", "b"], train)
    again = runs.trained_baseline(tmp_path, ex.EURLEX, ["b", "a"], train)
    assert len(calls) == 1
    assert (again.k, again.dev_f1, again.train_seconds) == (2, 0.5, 9.0)
    assert again.ranked == [{"Water": 0.4}, {"Health": 0.9}] and first.ranked[0] == {"Health": 0.9}
    with pytest.raises(ValueError, match="no ranking for 1 sample items"):
        runs.trained_baseline(tmp_path, ex.EURLEX, ["a", "c"], train)
