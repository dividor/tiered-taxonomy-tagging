"""Tests for the new technique's scoring and choices, on invented data."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy import new_technique as nt
from tiered_taxonomy import signals as sg
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.methods import combined as cb
from tiered_taxonomy.taxonomy import from_paths

PATHS = [f"{pillar}->x" for pillar in sorted(bm.HUMSET_PILLARS_1D)]
HUMSET_TREE = from_paths([*PATHS, "Impact->y", "Agriculture"])
EUROVOC = from_paths(["Top->Middle->Low->Leaf", "Top->Middle->Low->Other"])


def tiered(item_id: str, labels: set[str], by_tier: dict[int, frozenset[str]]) -> Item:
    return Item(item_id, f"text about {item_id}", "en", frozenset(labels), labels_by_tier=by_tier)


def test_fetch_saves_each_split_and_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rankings = tmp_path / "fetched.json"
    tree = from_paths(["a", *(f"b{i}" for i in range(200))])
    rankings.write_text(
        json.dumps({"1": {"a": 0.5, "z": 0.4, **{f"b{i}": 0.1 for i in range(200)}}}), encoding="utf-8"
    )
    monkeypatch.setattr(nt.gj, "fetch", lambda _root, _name: rankings)
    assert nt.fetch_fine_tuned(tmp_path, "toy", "run", "source", tree) == {"z"}
    saved = sg.load_signal(tmp_path, "toy", "source", "test")["1"]
    assert saved["a"] == 0.5
    assert "z" not in saved

    def missing(_root: Path, _name: str) -> Path:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(nt.gj, "fetch", missing)
    with pytest.raises(FileNotFoundError):
        nt.fetch_fine_tuned(tmp_path, "toy", "other", "source", tree)


def test_with_all_tiers_adds_parents_to_labels() -> None:
    item = tiered("1", {"Impact->y"}, {1: frozenset({"Impact"}), 2: frozenset({"Impact->y"})})
    assert nt.with_all_tiers([item])[0].labels == {"Impact", "Impact->y"}


def test_expand_sources_adds_parent_scores() -> None:
    assert nt.expand_sources({"a": {"1": {"Impact->y": 0.4}}})["a"]["1"]["Impact"] == 0.4


def test_humset_score_uses_thresholds_from_the_tuning_items() -> None:
    labels = bm.humset_categories(HUMSET_TREE)
    gold = [{"Agriculture"}] * 20 + [set()] * 80
    rankings = [{"Agriculture": 0.9}] * 20 + [{"Agriculture": 0.1}] * 80
    scores = nt.humset_score(gold, rankings, gold, rankings, HUMSET_TREE)
    assert set(scores) == set(labels)
    assert scores["Sectors"] == 1.0
    assert scores["Pillars 2D"] == 0.0


def test_eurlex_score_covers_all_labels_and_level_three() -> None:
    gold = [{"Top->Middle->Low->Leaf"}]
    rankings = [{"Top->Middle->Low->Other": 0.9, "Top->Middle->Low->Leaf": 0.5}]
    scores = nt.eurlex_score(gold, rankings, EUROVOC)
    assert scores["all"] == 0.0
    assert scores["level_3"] == 100.0


def test_headline_is_the_mean_on_humset_and_all_labels_on_eurlex() -> None:
    assert nt.headline("humset", {"a": 0.2, "b": 0.4}) == pytest.approx(0.3)
    assert nt.headline("eurlex", {"all": 50.0, "level_3": 70.0}) == 50.0


def test_choose_sources_keeps_only_those_whose_removal_hurts() -> None:
    scores = {
        "all sources": {"all": 50.0},
        "useful": {"all": 49.0},
        "useless": {"all": 49.95},
        "harmful": {"all": 51.0},
    }
    assert nt.choose_sources("eurlex", scores) == ["useful"]


def test_passes_need_the_whole_interval_above_the_printed_score() -> None:
    assert nt.eurlex_passes({"all": (48.5, 50.0), "level_3": (67.0, 69.0)}) == {"all": True, "level_3": False}
    close = {category: (value + 0.01, value + 0.1) for category, value in bm.HUMBERT_BEST.items()}
    assert all(nt.humset_passes(close).values())
    close["Sectors"] = (bm.HUMBERT_BEST["Sectors"] - 0.01, bm.HUMBERT_BEST["Sectors"] + 0.1)
    assert not nt.humset_passes(close)["Sectors"]


def test_intervals_bracket_the_score() -> None:
    gold = [{"a"} if i % 2 else {"b"} for i in range(200)]
    predicted = [{"a"} if i % 4 == 1 else {"b"} for i in range(200)]
    labels = frozenset({"a", "b"})
    low, high = nt.macro_interval(gold, predicted, labels, seed=1)
    assert low < bm.macro_f1(gold, predicted, labels) < high
    rankings = [{next(iter(item)): 1.0} for item in predicted]
    low, high = nt.mrp_interval(gold, rankings, seed=1)
    assert low < bm.mean_r_precision(gold, rankings) < high


def test_source_scores_and_combining_on_development_items() -> None:
    rng = np.random.default_rng(1)
    health, trade = "Top->Mid->Health", "Top->Mid->Trade"
    tree = from_paths([health, trade])
    items = []
    good: sg.Rankings = {}
    for i in range(200):
        label = health if rng.random() < 0.5 else trade
        items.append(Item(str(i), "text", "en", frozenset({label})))
        good[str(i)] = {label: 0.9}
    sources = {"good": good}
    graph = cb.LabelGraph(tree, [])
    settings = cb.CombinedSettings(max_iterations=20)
    combined = nt.combine(items[:150], items[150:], sources, ["good"], graph, 50, settings)
    assert len(combined["tune_rankings"]) == 50
    scores = nt.source_scores("eurlex", items[150:], items[:150], sources, tree)
    assert scores["good"]["all"] == 100.0


def test_exact_micro_f1_matches_a_known_value() -> None:
    assert nt.exact_micro_f1([{"a"}], [{"a"}]) == 1.0


def test_eurlex_sources_read_the_right_folders(tmp_path: Path) -> None:
    for folder, source, value in (
        ("eurlex", "classifier", 1.0),
        ("eurlex_newer", "classifier", 2.0),
        ("eurlex", "xr_transformer", 3.0),
        ("eurlex_newer", "xr_transformer", 9.0),
    ):
        sg.save_signal(tmp_path, folder, source, "dev", {"1": {"a": value}})
    names = ["classifier", "classifier_train_only", "xr_transformer"]
    loaded = nt.eurlex_sources(tmp_path, names, "dev", with_newer=True)
    assert {name: rankings["1"]["a"] for name, rankings in loaded.items()} == {
        "classifier": 2.0,
        "classifier_train_only": 1.0,
        "xr_transformer": 3.0,
    }
    assert nt.eurlex_sources(tmp_path, ["classifier"], "dev", with_newer=False)["classifier"]["1"]["a"] == 1.0


def test_without_newer_swaps_and_drops_sources() -> None:
    assert nt.without_newer(["classifier", "classifier_train_only", "llm", "reranker", "knn"]) == [
        "classifier",
        "knn",
    ]


def test_top_r_keeps_as_many_labels_as_gold() -> None:
    assert nt.top_r([{"a", "b"}, {"c"}], [{"a": 0.9, "x": 0.8, "b": 0.1}, {"c": 0.2}]) == [{"a", "x"}, {"c"}]


def test_humset_intervals_and_gain_on_resampled_items() -> None:
    categories = {"one": frozenset({"a"}), "two": frozenset({"b"})}
    gold = [{"a"} if i % 2 else {"b"} for i in range(100)]
    perfect = nt.category_matrices(gold, {"one": gold, "two": gold}, categories)
    nothing = nt.category_matrices(gold, {"one": [set()] * 100, "two": [set()] * 100}, categories)
    assert nt.macro_scores(perfect, np.arange(100)) == {"one": 1.0, "two": 1.0, "Mean": 1.0}
    assert nt.humset_intervals(perfect, seed=1)["Mean"] == (1.0, 1.0)
    assert nt.humset_gain(perfect, nothing, seed=1) == (1.0, 1.0, 1.0)


def test_mrp_gain_is_paired() -> None:
    gold = [{"a"}] * 50
    point, low, high = nt.mrp_gain(gold, [{"a": 1.0}] * 50, [{"b": 1.0}] * 50, seed=1)
    assert (point, low, high) == (100.0, 100.0, 100.0)


def test_development_choice_keeps_a_useful_source_and_drops_noise() -> None:
    rng = np.random.default_rng(3)
    health, trade = "Top->Mid->Health", "Top->Mid->Trade"
    tree = from_paths([health, trade])
    items, good, noise = [], {}, {}
    for i in range(300):
        label = health if rng.random() < 0.5 else trade
        items.append(Item(str(i), "text", "en", frozenset({label})))
        good[str(i)] = {label: 0.9, (trade if label == health else health): 0.1}
        noise[str(i)] = {health: float(rng.random()), trade: float(rng.random())}
    choice = nt.development_choice(
        "eurlex",
        items[:200],
        items[200:],
        items[200:],
        items[200:],
        {"good": good, "noise": noise},
        cb.LabelGraph(tree, []),
        50,
        cb.CombinedSettings(max_iterations=20),
        tree,
    )
    assert choice["single"]["good"]["all"] == 100.0
    assert choice["chosen"] == ["good"]
    assert choice["chosen_scores"]["all"] >= 90.0
    assert nt.best_single("eurlex", choice) == "good"


def test_development_choice_raises_when_nothing_is_kept() -> None:
    tree = from_paths(["Top->Mid->Health", "Top->Mid->Trade"])
    items = [Item(str(i), "text", "en", frozenset({"Top->Mid->Health"})) for i in range(60)]
    flat = {item.id: {"Top->Mid->Health": 0.5} for item in items}
    with pytest.raises(ValueError, match="none was chosen"):
        nt.development_choice(
            "eurlex",
            items[:40],
            items[40:],
            items[40:],
            items[40:],
            {"a": flat, "b": flat},
            cb.LabelGraph(tree, []),
            50,
            cb.CombinedSettings(max_iterations=5),
            tree,
        )


def toy_stack(
    labels: list[set[str]], names: tuple[str, ...], tree_paths: list[str], seed: int
) -> tuple[nt.Stack, list[Item]]:
    """Return a stack whose `good` source ranks the gold labels first and whose `noise` source is random,
    with the last 100 items held back as test items."""
    rng = np.random.default_rng(seed)
    everything = sorted(set().union(*labels))
    items = [Item(str(i), "text", "en", frozenset(gold)) for i, gold in enumerate(labels)]
    sources: dict[str, sg.Rankings] = {"good": {}, "noise": {}}
    for item in items:
        sources["good"][item.id] = {label: 0.9 if label in item.labels else 0.1 for label in everything}
        sources["noise"][item.id] = {label: float(rng.random()) for label in everything}
    stack = nt.Stack(
        names=names,
        fit=items[:200],
        cut=items[200:300],
        dev_sources=sources,
        test_sources=sources,
        graph=cb.LabelGraph(from_paths(tree_paths), []),
        words=50,
        settings=cb.CombinedSettings(max_iterations=20),
    )
    return stack, items[300:]


def test_require_rankings_raises_on_a_missing_item() -> None:
    items = [Item("1", "text", "en", frozenset({"a"})), Item("2", "text", "en", frozenset({"a"}))]
    nt.require_rankings({"good": {"1": {}, "2": {"a": 1.0}}}, items)
    with pytest.raises(ValueError, match="no ranking for 1 of 2"):
        nt.require_rankings({"good": {"1": {"a": 1.0}}}, items)


def test_humset_final_scores_the_stack_beside_the_best_single_source() -> None:
    rng = np.random.default_rng(5)
    first = sorted(bm.HUMSET_PILLARS_1D)[0]
    choices = [{"Agriculture"}, {"Impact", "Impact->y"}, {first, f"{first}->x"}]
    labels = [choices[int(rng.integers(3))] for _ in range(400)]
    stack, test = toy_stack(labels, ("good", "noise"), [*PATHS, "Impact->y", "Agriculture"], seed=5)
    result = nt.humset_final(stack, test, "good", HUMSET_TREE, seed=1, draws=20)
    assert result["best_single"]["Sectors"] == 1.0
    assert result["scores"]["Sectors"] >= 0.9
    assert set(result["passes"]) == set(bm.HUMBERT_BEST)
    low, high = result["intervals"]["Mean"]
    assert low <= result["scores"]["Mean"] <= high
    assert len(result["gain_over_best_single"]) == 3
    assert len(result["rankings"]) == len(test)
    assert {"good_score", "noise_score"} <= set(result["importance"])


def test_humset_final_gives_a_missing_pillar_its_strongest_subpillars_score() -> None:
    rng = np.random.default_rng(5)
    first = sorted(bm.HUMSET_PILLARS_1D)[0]
    choices = [{"Agriculture"}, {"Impact", "Impact->y"}, {first, f"{first}->x"}]
    labels = [choices[int(rng.integers(3))] for _ in range(400)]
    stack, test = toy_stack(labels, ("good", "noise"), [*PATHS, "Impact->y", "Agriculture"], seed=5)
    for rankings in stack.dev_sources.values():
        for ranking in rankings.values():
            ranking.pop(first, None)
    result = nt.humset_final(stack, test, "good", HUMSET_TREE, seed=1, draws=20)
    # The one pillar in the data scores perfectly through its subpillar; the group's other pillars, which no
    # item has, score zero, so the group's macro F1 is one in however many pillars it has.
    assert result["best_single"]["Pillars 1D"] == pytest.approx(1 / len(bm.HUMSET_PILLARS_1D))


def test_eurlex_final_scores_and_reports_rarity() -> None:
    rng = np.random.default_rng(6)
    leaf, other = "Top->Middle->Low->Leaf", "Top->Middle->Low->Other"
    labels = [{leaf} if rng.random() < 0.5 else {other} for _ in range(400)]
    stack, test = toy_stack(labels, ("good", "noise"), [leaf, other], seed=6)
    # Three labels given once make the least-used quarter, so the leaf, given 50 times, is common.
    counts = {leaf: 50, "a": 1, "b": 1, "c": 1}
    result = nt.eurlex_final(stack, test, "good", EUROVOC, counts, seed=1, draws=20)
    assert result["best_single"]["all"] == 100.0
    assert result["scores"]["all"] >= 90.0
    assert set(result["passes"]) == {"all", "level_3"}
    assert result["rarity"]["good"]["common"]["recall"] == 1.0
    assert set(result["rarity"]["good"]) == {"common", "unseen"}


def test_final_runs_raise_when_a_test_ranking_is_missing() -> None:
    leaf, other = "Top->Middle->Low->Leaf", "Top->Middle->Low->Other"
    stack, test = toy_stack([{leaf}] * 200 + [{other}] * 200, ("good",), [leaf, other], seed=7)
    partial = {"good": {key: value for key, value in stack.test_sources["good"].items() if key != test[0].id}}
    with pytest.raises(ValueError, match="no ranking"):
        nt.eurlex_final(replace(stack, test_sources=partial), test, "good", EUROVOC, {}, seed=1, draws=5)


def test_addition_choice_keeps_a_useful_new_source_and_drops_noise() -> None:
    rng = np.random.default_rng(5)
    health, trade = "Top->Mid->Health", "Top->Mid->Trade"
    tree = from_paths([health, trade])
    items, weak, good, noise = [], {}, {}, {}
    for i in range(300):
        label = health if rng.random() < 0.5 else trade
        other = trade if label == health else health
        items.append(Item(str(i), "text", "en", frozenset({label})))
        right = rng.random() < 0.6
        weak[str(i)] = {label: 0.6, other: 0.4} if right else {label: 0.4, other: 0.6}
        good[str(i)] = {label: 0.9, other: 0.1}
        noise[str(i)] = {health: float(rng.random()), trade: float(rng.random())}
    choice = nt.addition_choice(
        "eurlex",
        items[:200],
        items[200:],
        items[200:],
        items[200:],
        {"weak": weak, "good": good, "noise": noise},
        ["weak"],
        (cb.LabelGraph(tree, []), None),
        50,
        cb.CombinedSettings(max_iterations=20),
        tree,
    )
    assert set(choice["variants"]) == {"first attempt", "+ good", "+ noise"}
    assert choice["kept"] == ["good"]
    assert choice["names"] == ["good", "weak"]
    assert choice["priors"] is False
    assert choice["chosen_scores"]["all"] > choice["variants"]["first attempt"]["all"]


def test_addition_choice_keeps_priors_when_labels_drift_and_uses_their_graph() -> None:
    # Both labels are used equally often overall, so only their timing tells them apart. Ties sort by
    # label, which puts the old one first, so the stack without priors ranks the wrong label first.
    old, recent = "Top->Mid->Alpha", "Top->Mid->Zeta"
    tree = from_paths([old, recent])
    dated = [(2000, [old])] * 200 + [(2010, [recent])] * 200
    documents = [labels for _, labels in dated]
    priors = cb.TimedPriors.from_documents(dated, recent_documents=200)
    timed = cb.LabelGraph(tree, documents, priors=priors)
    items = [Item(str(i), "text", "en", frozenset({recent})) for i in range(200)]
    flat = {item.id: {old: 0.5, recent: 0.5} for item in items}
    choice = nt.addition_choice(
        "eurlex",
        items[:100],
        items[100:],
        items[100:],
        items[100:],
        {"flat": flat},
        ["flat"],
        (cb.LabelGraph(tree, documents), timed),
        50,
        cb.CombinedSettings(max_iterations=20),
        tree,
    )
    assert choice["variants"]["first attempt"]["all"] == 0.0
    assert choice["kept"] == [nt.PRIORS]
    assert choice["priors"] is True
    assert choice["graph"] is timed
    assert choice["chosen_scores"]["all"] == 100.0


def test_humset_against_shows_the_better_stack_gaining() -> None:
    rng = np.random.default_rng(8)
    first = sorted(bm.HUMSET_PILLARS_1D)[0]
    choices = [{"Agriculture"}, {"Impact", "Impact->y"}, {first, f"{first}->x"}]
    labels = [choices[int(rng.integers(3))] for _ in range(400)]
    stack, test = toy_stack(labels, ("good", "noise"), [*PATHS, "Impact->y", "Agriculture"], seed=8)
    good, noisy = replace(stack, names=("good",)), replace(stack, names=("noise",))
    result = nt.humset_against(good, noisy, test, HUMSET_TREE, seed=1, draws=20)
    point, low, high = result["gain"]
    assert result["second"]["Mean"] > result["first"]["Mean"]
    assert low <= point <= high
    assert point > 0


def test_eurlex_against_shows_the_better_stack_gaining_at_both_levels() -> None:
    rng = np.random.default_rng(9)
    leaf, other = "Top->Middle->Low->Leaf", "Top->Middle->Low->Other"
    labels = [{leaf} if rng.random() < 0.5 else {other} for _ in range(400)]
    stack, test = toy_stack(labels, ("good", "noise"), [leaf, other], seed=9)
    good, noisy = replace(stack, names=("good",)), replace(stack, names=("noise",))
    result = nt.eurlex_against(good, noisy, test, EUROVOC, seed=1, draws=20)
    assert set(result["gain"]) == {"all", "level_3"}
    assert result["second"]["all"] > result["first"]["all"]
    assert result["gain"]["all"][0] > 0


def test_fetch_history_reads_the_runs_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = tmp_path / "gliner_humset_history.json"
    record.write_text(json.dumps([{"stage": "fine-tuned", "development_mrp": 40.0}]), encoding="utf-8")
    asked = []

    def download(_root: Path, filename: str) -> Path:
        asked.append(filename)
        return record

    monkeypatch.setattr(nt.gj, "fetch", download)
    assert nt.fetch_history(tmp_path, "gliner_humset") == [{"stage": "fine-tuned", "development_mrp": 40.0}]
    assert asked == ["gliner_humset_history.json"]


def test_gliner_counts_only_with_newer_laws() -> None:
    assert nt.without_newer(["classifier_train_only", "gliner", "knn"]) == ["classifier", "knn"]
