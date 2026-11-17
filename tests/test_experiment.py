"""Tests for the shared experiment steps, using small invented test sets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from tiered_taxonomy import experiment as ex
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.llm import Client, ReplyStore
from tiered_taxonomy.methods.base import Prediction
from tiered_taxonomy.models import ModelSpec
from tiered_taxonomy.taxonomy import from_paths

TREE = from_paths(["P->x", "P->y", "Q->z"])


def item(item_id: str, labels: set[str], language: str = "en", text: str = "a b c") -> Item:
    by_tier = {tier: frozenset(TREE.roll_up(labels, tier)) for tier in (1, 2)}
    return Item(item_id, text, language, frozenset(labels), item_id, by_tier)


def toy_set() -> ex.TestSet:
    items = [item("1", {"P->x"}), item("2", {"Q->z"}, "fr")]
    return ex.TestSet("toy", TREE, (1, 2), [], [], items, Counter({"P->x": 50, "Q->z": 2}))


def test_keep_known_drops_labels_outside_the_tree_and_sets_gold_by_tier() -> None:
    kept = ex._keep_known([Item("1", "t", "en", frozenset({"P->x", "nope"}))], TREE)[0]
    assert kept.labels == {"P->x"}
    assert kept.labels_by_tier[1] == {"P"}


def test_rarity_bands() -> None:
    counts = {"a": 50, "b": 2}
    assert [ex.rarity(label, counts, 2) for label in ("a", "b", "c")] == ["common", "rare", "unseen"]


def test_rare_cutoff_is_the_least_used_quarter_with_ties_and_ignores_unused_labels() -> None:
    assert ex.rare_cutoff({"a": 1, "b": 3, "c": 8, "d": 40, "x": 0}) == 1
    assert ex.rare_cutoff({"a": 5, "b": 5, "c": 5, "d": 9, "e": 20}) == 5
    assert (
        ex.rare_cutoff({label: count for label, count in zip("abcdefgh", range(10, 90, 10), strict=True)})
        == 20
    )
    assert ex.rare_cutoff({"a": 0}) == 0
    assert ex.rarity("a", {"a": 0}, ex.rare_cutoff({"a": 0})) == "unseen"


def test_recall_by_rarity() -> None:
    results = ex.recall_by_rarity([{"a", "b"}], [{"a"}], {"a": 50, "b": 2})
    assert results["common"]["recall"] == 1.0
    assert results["rare"] == {"recall": 0.0, "gold_labels": 1}


def test_evaluate_scores_every_view() -> None:
    data = toy_set()
    results = ex.evaluate(data, data.test, [{"P->y"}, {"Q->z", "made up"}])
    assert results["tiers"]["1"]["f1"] == 1.0
    assert results["exact"]["f1"] == pytest.approx(0.5)
    assert results["languages"]["fr"]["f1_finest_tier"] == 1.0
    assert results["rarity"]["rare"]["recall"] == 1.0
    assert results["labels_per_item"] == 1.0


def test_evaluate_drops_a_parent_given_with_its_own_child_before_scoring() -> None:
    data = toy_set()
    results = ex.evaluate(data, data.test, [{"P", "P->x"}, {"Q"}])
    assert results["exact"]["precision"] == pytest.approx(0.5)
    assert results["exact"]["recall"] == pytest.approx(0.5)
    assert results["labels_per_item"] == 1.0
    assert results["tiers"]["1"]["f1"] == 1.0


def test_scoring_within_a_project_ignores_labels_it_never_used() -> None:
    data = toy_set()
    entries = [
        Item("1", "t", "en", frozenset({"P->x"}), "1", {1: frozenset({"P"}), 2: frozenset({"P->x"})}, "A"),
    ]
    used = ex.project_labels(entries, TREE)
    assert used["A"] == {"P->x", "P"}
    plain = ex.evaluate(data, entries, [{"P->x", "Q->z"}])
    within = ex.evaluate_within_project(data, entries, [{"P->x", "Q->z"}], used)
    assert plain["tiers"]["2"]["precision"] == 0.5
    assert within["2"]["precision"] == 1.0


def test_run_method_records_calls_and_cost(tmp_path: Path) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="{}")]))
    client = Client(ModelSpec("fake", "m"), ReplyStore(tmp_path / "r.sqlite"), model)

    def work(_item: Item) -> Prediction:
        return Prediction(labels={"P": 1.0}, calls=2, invalid=1)

    predictions, record = ex.run_method("toy", work, [item("1", {"P"})], [client], workers=1)
    assert ex.label_sets(predictions) == [{"P"}]
    assert (record.items, record.calls, record.invalid) == (1, 2, 1)


def test_predictions_round_trip_and_a_missing_item_raises(tmp_path: Path) -> None:
    ex.save_predictions(tmp_path, "toy", "m", ["1"], [{"b", "a"}])
    assert ex.load_predictions(tmp_path, "toy", "m", ["1"]) == [{"a", "b"}]
    with pytest.raises(KeyError):
        ex.load_predictions(tmp_path, "toy", "m", ["2"])


def test_save_results_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "RESULTS", tmp_path)
    path = ex.save_results("toy", {"a": 1})
    assert path.read_text(encoding="utf-8").strip().startswith("{")
    assert ex.load_results("toy") == {"a": 1}
    with pytest.raises(FileNotFoundError):
        ex.load_results("missing")


def test_sampling_is_repeatable_and_respects_limits() -> None:
    long_text = " ".join(["w"] * 20)
    data = toy_set()
    data.test = [item(str(i), {"P->x"}, text=long_text if i < 3 else "short") for i in range(10)]
    sample = ex.sample_eurlex(data, normal=2, long_documents=2, threshold=10, seed=1)
    assert sum(len(entry.text.split()) >= 10 for entry in sample) == 2
    assert len(sample) == 4
    ids = [entry.id for entry in sample]
    assert ids == [entry.id for entry in ex.sample_eurlex(data, 2, 2, 10, 1)]
    balanced = ex.sample_humset(toy_set(), 1, 1)
    assert {entry.language for entry in balanced} == {"en", "fr"}
    subset: Any = ex.pilot_subset(data.test, 3, 1)
    assert len(subset) == 3
    assert ex.input_text(item("1", set(), text="a b c"), 2) == "a b"


def test_both_humset_versions_count_as_humset_and_eurlex_does_not() -> None:
    assert ex.is_humset(ex.HUMSET) and ex.is_humset(ex.HUMSET_V2)
    assert not ex.is_humset(ex.EURLEX)
    assert ex.HUMSET_VERSIONS == {ex.HUMSET: "1.0.0", ex.HUMSET_V2: "2.0.0"}


def test_humset_sample_leaves_out_languages_outside_the_three() -> None:
    data = toy_set()
    data.test = [item("1", {"P->x"}), item("2", {"Q->z"}, "fr"), item("3", {"P->y"}, "pt")]
    sample = ex.sample_humset(data, 5, 1)
    assert sorted(entry.id for entry in sample) == ["1", "2"]


def test_split_overlap_counts_shared_documents_and_identical_text() -> None:
    train = [item("a", {"P->x"}, text="one"), item("b", {"P->x"}, text="two")]
    test = [item("a", {"P->x"}, text="other"), item("c", {"P->x"}, text=" two "), item("d", {"P->x"})]
    assert ex.split_overlap(train, test) == {"same_document": 0.3333, "same_text": 0.3333}
