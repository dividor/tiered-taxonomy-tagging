"""Tests for saving and combining each method's scores, on invented data."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tiered_taxonomy import experiment as ex
from tiered_taxonomy import signals as sg
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.methods import combined as cb
from tiered_taxonomy.methods.base import Prediction
from tiered_taxonomy.taxonomy import from_paths

TREE = from_paths(["Health->Water", "Trade->Export"])


def item(item_id: str, labels: set[str], language: str = "en") -> Item:
    return Item(item_id, f"text about {item_id}", language, frozenset(labels))


def test_signals_round_trip_and_a_missing_one_raises(tmp_path: Path) -> None:
    sg.save_signal(tmp_path, "toy", "xr", "dev", {"1": {"a": 0.5}})
    assert sg.load_signal(tmp_path, "toy", "xr", "dev") == {"1": {"a": 0.5}}
    with pytest.raises(FileNotFoundError):
        sg.load_signal(tmp_path, "toy", "xr", "test")


def test_by_id_and_rankings_from_predictions() -> None:
    rankings = sg.rankings_from_predictions([Prediction(labels={"a": 0.5})])
    assert sg.by_id([item("1", set())], rankings) == {"1": {"a": 0.5}}


def test_neighbour_rankings_use_training_labels() -> None:
    train = [item("a", {"Health"}), item("b", {"Trade"})]
    documents = np.array([[1.0, 0.0], [0.0, 1.0]])
    ranked = sg.neighbour_rankings(np.array([[1.0, 0.0]]), documents, train, 1)
    assert list(ranked[0]) == ["Health"]


def test_few_shot_examples_only_use_labelled_items_and_repeat() -> None:
    train = [item("a", {"Health->Water"}), item("b", set()), item("c", {"Trade"})]
    examples = sg.few_shot_examples(train, TREE, 2, seed=1)
    assert len(examples) == 2
    assert {name for example in examples for name in example.keywords} == {"Water", "Trade"}
    assert examples == sg.few_shot_examples(train, TREE, 2, seed=1)


def test_similar_examples_stay_in_the_texts_language() -> None:
    train = [item("a", {"Health"}, "fr"), item("b", {"Trade"}, "en")]
    documents = np.array([[1.0, 0.0], [0.9, 0.1]])
    examples = sg.similar_examples(np.array([1.0, 0.0]), documents, train, TREE, "en", 1)
    assert examples[0].keywords == ("Trade",)


def test_assemble_gives_empty_rankings_for_missing_items() -> None:
    assembled = sg.assemble([item("1", set())], {"xr": {}, "knn": {"1": {"a": 1.0}}}, ["xr", "knn"])
    assert assembled == [{"xr": {}, "knn": {"a": 1.0}}]


def test_strongest_keeps_the_top_scores() -> None:
    assert sg.strongest({"a": 0.1, "b": 0.9, "c": 0.5}, 2) == {"b": 0.9, "c": 0.5}


def test_tuning_cut_reads_rankings_by_item_id() -> None:
    items = [item("1", {"a"}), item("2", {"b"})]
    rankings: sg.Rankings = {"1": {"a": 0.9, "x": 0.2}, "2": {"b": 0.8, "y": 0.1}}
    cut, value = sg.tuning_cut(items, rankings, [1, 2], [0.0])
    assert cut.limit == 1
    assert value == 1.0


def test_collect_signals_saves_every_model_and_does_not_repeat_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = [item(str(i), {"Health" if i % 2 else "Trade"}) for i in range(8)]
    train = [
        Item(e.id, ("health care" if "Health" in e.labels else "trade law") + " text", "en", e.labels)
        for e in train
    ]
    dev = [Item("d", "health care text", "en", frozenset({"Health"}))]
    test = [Item("t", "trade law text", "en", frozenset({"Trade"}))]
    data = ex.TestSet("toy", TREE, (1,), train, dev, test, Counter())
    folder = tmp_path / "corpus_embeddings" / "toy"
    folder.mkdir(parents=True)
    vector = {True: [1.0, 0.0], False: [0.0, 1.0]}
    np.save(folder / "train.npy", np.array([vector["Health" in e.labels] for e in train]))
    np.save(folder / "dev.npy", np.array([vector[True]]))
    np.save(folder / "test.npy", np.array([vector[False]]))
    settings = {
        "tuning_items": 1,
        "cut_limits": [1],
        "cut_thresholds": [0.0],
        "embedding_classifier": {"epochs": 2, "batch_size": 4, "learning_rates": [0.1]},
        "neighbour_choices": [1],
        "max_features": 50,
    }

    class Index:
        ids = ("Health", "Trade")

        def search_many(self, queries: np.ndarray, _k: int) -> list[dict[str, float]]:
            return [{"Health": float(query[0]), "Trade": float(query[1])} for query in queries]

    index: Any = Index()
    details = sg.collect_signals(data, tmp_path, settings, 1, 50, "cpu", index)
    assert set(details) == {"omikuji", "embedding_classifier", "knn", "label_search"}
    assert next(iter(sg.load_signal(tmp_path, "toy", "knn", "dev")["d"])) == "Health"
    assert sg.load_signal(tmp_path, "toy", "label_search", "test")["t"]["Trade"] == 1.0
    monkeypatch.setattr(sg, "omikuji_signals", lambda *_: pytest.fail("repeated a finished step"))
    assert sg.collect_signals(data, tmp_path, settings, 1, 50, "cpu", index) == details


def test_xr_signals_refuse_rankings_that_do_not_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev, test = [item("d", {"Health"})], [item("t", {"Trade"})]
    data = ex.TestSet("toy", TREE, (1,), [], dev, test, Counter())
    monkeypatch.setattr(sg, "fetch_xr_rankings", lambda *_, **__: {"d": {"Health": 0.5}, "t": {"Trade": 0.5}})
    sg.xr_signals(data, tmp_path)
    assert sg.load_signal(tmp_path, "toy", "xr_transformer", "test")["t"] == {"Trade": 0.5}
    monkeypatch.setattr(sg, "fetch_xr_rankings", lambda *_, **__: {"d": {"Health": 0.5}})
    with pytest.raises(ValueError, match="no ranking"):
        sg.xr_signals(data, tmp_path)
    monkeypatch.setattr(sg, "fetch_xr_rankings", lambda *_, **__: {"d": {"Nope": 0.5}, "t": {"Trade": 0.5}})
    with pytest.raises(ValueError, match="not in"):
        sg.xr_signals(data, tmp_path)


def test_drop_outside_tree_reports_a_few_and_refuses_many(monkeypatch: pytest.MonkeyPatch) -> None:
    rankings = {"d": {"Health": 0.5, "Nope": 0.4}}
    monkeypatch.setattr(sg, "MAX_OUTSIDE_SHARE", 0.5)
    assert sg.drop_outside_tree(rankings, TREE) == ({"d": {"Health": 0.5}}, {"Nope"})
    monkeypatch.setattr(sg, "MAX_OUTSIDE_SHARE", 0.4)
    with pytest.raises(ValueError, match="not in this taxonomy"):
        sg.drop_outside_tree(rankings, TREE)


def test_split_for_combining_has_no_overlap_and_refuses_too_many() -> None:
    items = [item(str(i), set()) for i in range(10)]
    fit, review, cut = sg.split_for_combining(items, 5, 3, 2, seed=1)
    assert (len(fit), len(review), len(cut)) == (5, 3, 2)
    assert len({entry.id for entry in [*fit, *review, *cut]}) == 10
    with pytest.raises(ValueError, match="development items"):
        sg.split_for_combining(items, 8, 3, 2, seed=1)


def test_the_second_stage_learns_from_the_llms_ratings() -> None:
    graph = cb.LabelGraph(TREE, [])
    rankings = [{"Health": 0.5, "Trade": 0.5} for _ in range(200)]
    ratings = []
    gold: list[set[str]] = []
    for i in range(200):
        good, bad = ("Health", "Trade") if i % 2 else ("Trade", "Health")
        ratings.append({good: 0.9, bad: 0.1})
        gold.append({good})
    model = sg.second_stage(rankings, ratings, gold, graph, cb.CombinedSettings(max_iterations=20))
    revised = sg.apply_second_stage(model, rankings[:2], ratings[:2], graph)
    assert revised[0]["Trade"] > 0.8 > 0.2 > revised[0]["Health"]
    assert revised[1]["Health"] > 0.8


def test_newer_laws_view_moves_earlier_laws_into_training(tmp_path: Path) -> None:
    train = [item("31990R0001", {"Health"})]
    dev = [item("32010R0001", {"Trade"}), item("32012R0001", {"Health"})]
    test = [item("32014R0001", {"Trade"})]
    data = ex.TestSet("toy", TREE, (1,), train, dev, test, Counter())
    folder = tmp_path / "corpus_embeddings" / "toy"
    folder.mkdir(parents=True)
    np.save(folder / "train.npy", np.array([[1.0, 0.0]]))
    np.save(folder / "dev.npy", np.array([[0.0, 1.0], [0.5, 0.5]]))
    np.save(folder / "test.npy", np.array([[0.2, 0.8]]))
    view = sg.newer_laws_view(data, tmp_path)
    assert view.name == "toy_newer"
    assert [entry.id for entry in view.train] == ["31990R0001", "32010R0001"]
    assert [entry.id for entry in view.dev] == ["32012R0001"]
    assert view.train_counts == Counter({"Health": 1, "Trade": 1})
    assert sg.load_embeddings(tmp_path, "toy_newer", "train").tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert sg.load_embeddings(tmp_path, "toy_newer", "dev").tolist() == [[0.5, 0.5]]


def test_candidate_lists_join_sources_and_skip_unranked_items() -> None:
    sources: dict[str, sg.Rankings] = {"a": {"1": {"x": 0.9, "y": 0.1}}, "b": {"1": {"z": 0.5}}}
    assert sg.candidate_lists(sources, ["1", "2"], 1) == {"1": ["x", "z"]}


def test_training_candidates_never_use_an_item_as_its_own_neighbour(tmp_path: Path) -> None:
    train = [item(str(i), {f"L{i}"}) for i in range(25)]
    data = ex.TestSet("toy", from_paths([f"L{i}" for i in range(25)]), (1,), train, [], [], Counter())
    folder = tmp_path / "corpus_embeddings" / "toy"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(1)
    np.save(folder / "train.npy", rng.normal(size=(25, 4)))

    class Index:
        def search(self, _query: np.ndarray, _k: int) -> dict[str, float]:
            return {"found_by_search": 1.0}

    index: Any = Index()
    lists = sg.training_candidates(data, tmp_path, index, 5, 30, seed=1)
    assert len(lists) == 5
    for item_id, labels in lists.items():
        assert "found_by_search" in labels
        assert f"L{item_id}" not in labels


def test_label_names_are_readable_paths() -> None:
    assert sg.label_names(TREE)["Health->Water"] == TREE.path("Health->Water")


def test_load_sources_reads_each_one(tmp_path: Path) -> None:
    sg.save_signal(tmp_path, "toy", "a", "dev", {"1": {"x": 1.0}})
    assert sg.load_sources(tmp_path, "toy", ["a"], "dev") == {"a": {"1": {"x": 1.0}}}
    with pytest.raises(FileNotFoundError):
        sg.load_sources(tmp_path, "toy", ["missing"], "dev")


def test_combining_trains_scores_and_measures_signal_importance() -> None:
    rng = np.random.default_rng(1)
    items = []
    reliable: sg.Rankings = {}
    noisy: sg.Rankings = {}
    for i in range(200):
        good, bad = ("Health", "Trade") if rng.random() < 0.5 else ("Trade", "Health")
        items.append(item(str(i), {good}))
        reliable[str(i)] = {good: 0.9}
        noisy[str(i)] = {bad: 0.9}
    sources = {"reliable": reliable, "noisy": noisy}
    settings = cb.CombinedSettings(max_iterations=30)
    graph = cb.LabelGraph(TREE, [])
    names = ["reliable", "noisy"]
    trained = sg.combine_and_score(items[:150], sources, items[150:], sources, names, graph, 50, settings)
    assert trained["tune_f1"] == 1.0
    scored = sg.apply_combiner(trained["model"], items[150:], sources, names, graph, 50, settings)
    kept = [
        {label for label, value in ranking.items() if value >= trained["threshold"]} for ranking in scored
    ]
    assert kept == [set(entry.labels) for entry in items[150:]]
    rows, keys = cb.build_rows(sg.assemble(items, sources, names), ["t"] * 200, graph, names, 5)
    target = np.array([label in items[position].labels for position, label in keys])
    importance = sg.signal_importance(trained["model"], rows, target, cb.feature_names(names), seed=1)
    assert importance["reliable_found"] + importance["reliable_score"] > 0.1
    assert importance["tier"] <= 0.01


def test_most_frequent_breaks_ties_by_label_whatever_the_insertion_order() -> None:
    first = Counter({"b": 2, "c": 1, "a": 1, "d": 3})
    second = Counter({"a": 1, "d": 3, "c": 1, "b": 2})
    assert sg.most_frequent(first, 3) == sg.most_frequent(second, 3) == ["d", "b", "a"]
