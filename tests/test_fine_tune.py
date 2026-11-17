"""Tests for the fine-tuning job's own logic. The job file stands alone, so it is loaded from its path."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy import datasets as ds
from tiered_taxonomy import experiment as ex
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.taxonomy import from_paths

PATH = Path(__file__).parent.parent / "scripts" / "hf_jobs" / "fine_tune.py"
spec = importlib.util.spec_from_file_location("fine_tune", PATH)
assert spec is not None and spec.loader is not None
ft = importlib.util.module_from_spec(spec)
sys.modules["fine_tune"] = ft
spec.loader.exec_module(ft)


def split(*ids: str) -> ft.Split:
    return ft.Split(list(ids), [f"text {item}" for item in ids], [["a"] for _ in ids])


def test_law_year_and_date_order_match_the_repository() -> None:
    ids = ["32014R0001", "32010R0002", "32010R0001"]
    assert ft.law_year("32012R0782") == bm.law_year("32012R0782")
    assert ft.date_order(ids) == [2, 1, 0]
    early, late = bm.split_by_date(
        [Item(item, "t", "en", frozenset()) for item in ["32012R0001", "32011R0001"]]
    )
    assert (early[0].id, late[0].id) == ("32011R0001", "32012R0001")


def test_humset_labels_gain_their_pillars() -> None:
    assert ft.humset_with_parents(["Context->Economy", "Agriculture"]) == [
        "Agriculture",
        "Context",
        "Context->Economy",
    ]


def test_humset_rows_keep_a_pillar_tagged_without_its_subpillars_as_the_repository_does() -> None:
    row = {
        "sectors": ["Health"],
        "subpillars_1d": ["Context->Economy"],
        "subpillars_2d": [],
        "pillars_1d": ["Context", "Covid-19"],
        "pillars_2d": ["Impact"],
    }
    assert ft.humset_row_labels(row) == ["Context->Economy", "Covid-19", "Health", "Impact"]
    assert ft.humset_row_labels(row) == sorted(ds.humset_labels(row))


def test_the_xr_transformer_job_reads_humset_labels_as_the_fine_tuning_job_does() -> None:
    def function(path: Path, name: str) -> str:
        source = path.read_text(encoding="utf-8")
        found = [
            ast.get_source_segment(source, node)
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        assert len(found) == 1
        return str(found[0])

    xr = PATH.parent / "train_xr_transformer.py"
    assert function(xr, "humset_row_labels") == function(PATH, "humset_row_labels")


def test_newer_laws_are_earlier_and_never_choose_the_epoch() -> None:
    dev = split("32013R0001", "32010R0001", "32012R0001", "32011R0001")
    added, selection = ft.divide_development("eurlex", dev, True, seed=1)
    assert added is not None and added.ids == ["32010R0001", "32011R0001"]
    assert selection.ids == ["32012R0001", "32013R0001"]
    without, same = ft.divide_development("eurlex", dev, False, seed=1)
    assert without is None and same.ids == selection.ids


def test_humset_refuses_newer_laws_and_selects_repeatably() -> None:
    dev = split("1", "2", "3")
    with pytest.raises(ValueError, match="only applies"):
        ft.divide_development("humset", dev, True, seed=1)
    assert ft.divide_development("humset", dev, False, seed=1)[1].ids == (
        ft.divide_development("humset", dev, False, seed=1)[1].ids
    )


def test_mean_r_precision_matches_the_repository() -> None:
    gold = [["a", "b"], ["c"], []]
    rankings = [{"a": 0.9, "x": 0.8, "b": 0.1}, {"c": 0.5}, {"z": 1.0}]
    assert ft.mean_r_precision(gold, rankings) == pytest.approx(
        bm.mean_r_precision([set(item) for item in gold], rankings)
    )


def test_training_pairs_balance_gold_and_wrong_candidates() -> None:
    data = ft.Split(["1"], ["t"], [["a"]])
    pairs = ft.training_pairs(data, {"1": ["a", "b", "c"]}, random.Random(1), per_document=1)
    assert {(label in {"a"}, target) for _, label, target in pairs} == {(True, 1.0), (False, 0.0)}
    assert len(pairs) == 2


def test_training_pairs_leave_out_items_with_no_candidate_list() -> None:
    data = ft.Split(["1", "2"], ["t", "t"], [["a"], ["a"]])
    pairs = ft.training_pairs(data, {"1": ["b"]}, random.Random(1))
    assert {position for position, _, _ in pairs} == {0}


def test_targets_mark_known_labels_and_ignore_others() -> None:
    matrix = ft.targets([["a", "unknown"], ["b"]], {"a": 0, "b": 1})
    assert matrix.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_top_labels_keep_the_strongest() -> None:
    ranked = ft.top_labels(np.array([0.1, 0.9, 0.5]), ["a", "b", "c"], 2)
    assert set(ranked) == {"b", "c"}


def test_without_leaves_out_the_given_items() -> None:
    assert ft.without(split("1", "2"), ["1"]).ids == ["2"]


def test_merge_joins_splits() -> None:
    assert ft.merge(split("1"), split("2")).ids == ["1", "2"]


def test_a_label_model_reads_the_first_or_last_real_token() -> None:
    class Encoder(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
            hidden = input_ids.unsqueeze(-1).float()
            return type("Output", (), {"last_hidden_state": hidden})

    ids = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[1, 1, 0]])
    first = ft.LabelModel(Encoder(), 1, 1, last_token=False)
    last = ft.LabelModel(Encoder(), 1, 1, last_token=True)
    for model in (first, last):
        torch.nn.init.ones_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    assert first(ids, mask).item() == 1.0
    assert last(ids, mask).item() == 2.0


def test_the_reporter_saves_results_without_a_repository(tmp_path: Path) -> None:
    reporter = ft.Reporter(None, "run", tmp_path)
    reporter.results("dev", {"a": 1})
    reporter.progress(step=1)
    assert (tmp_path / "run_dev.json").exists()


def test_save_label_model_keeps_the_weights_and_the_labels_in_output_order(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    model = ft.LabelModel(torch.nn.Linear(2, 2), 2, 3, last_token=False)
    reporter = ft.Reporter(None, "run", tmp_path)
    path = ft.save_label_model(model, ["b", "a", "c"], reporter)
    assert path == tmp_path / "run_model.safetensors"
    saved = load_file(path)
    assert set(saved) == set(model.state_dict())
    assert torch.equal(saved["head.weight"], model.head.weight.detach())
    assert torch.equal(saved["encoder.bias"], model.encoder.bias.detach())
    assert (tmp_path / "run_labels.json").read_text(encoding="utf-8") == '["b", "a", "c"]'


def test_the_job_refuses_to_save_the_weights_of_any_kind_but_the_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = ["fine_tune.py", "--kind", "llm", "--test-set", "humset", "--base-model", "m", "--save-model"]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="--save-model is for the classifier kind, not llm"):
        ft.main()


HUMSET_LABELS = [
    "Agriculture",
    "Health",
    *[f"{pillar}->Detail" for pillar in sorted(bm.HUMSET_PILLARS_1D)],
    *sorted(bm.HUMSET_PILLARS_1D),
    "Impact",
    "Impact->Impact On People",
]


def test_humset_heads_match_the_repositorys_categories() -> None:
    heads = ft.humset_heads(HUMSET_LABELS)
    expected = bm.humset_categories(from_paths(HUMSET_LABELS))
    assert {name: set(labels) for name, labels in heads.items()} == {
        name: set(labels) for name, labels in expected.items()
    }


def test_humset_heads_refuse_labels_without_the_known_pillars() -> None:
    with pytest.raises(ValueError, match="unknown pillars"):
        ft.humset_heads(["Agriculture", "Impact->Impact On People"])


def test_gliner_head_reads_humset_paths_and_eurlex_names() -> None:
    head, back = ft.gliner_head(["Impact->Impact On People", "Agriculture"], None)
    assert head["labels"] == ["Impact: Impact On People", "Agriculture"]
    assert head["multi_label"] is True and head["cls_threshold"] == 0.0
    assert back["Impact: Impact On People"] == "Impact->Impact On People"
    _, named = ft.gliner_head(["100", "200"], {"100": "fisheries", "200": "tariffs"})
    assert named == {"fisheries": "100", "tariffs": "200"}


def test_gliner_head_refuses_labels_that_read_the_same() -> None:
    with pytest.raises(ValueError, match="read the same"):
        ft.gliner_head(["1", "2"], {"1": "water", "2": "water"})


def test_gliner_tasks_give_laws_their_own_named_candidates_and_refuse_a_missing_list() -> None:
    tasks = ft.GlinerTasks(None, {"law": ["1", "2", "9"]}, {"1": "fisheries", "2": "tariffs"})
    heads, backs = tasks.get("law")
    assert heads[ft.GLINER_TASK]["labels"] == ["fisheries", "tariffs"]
    assert backs[ft.GLINER_TASK]["tariffs"] == "2"
    assert not tasks.has("other")
    with pytest.raises(ValueError, match="no candidate list for other"):
        tasks.get("other")


def test_gliner_examples_mark_gold_labels_in_each_head_and_skip_unscorable_items() -> None:
    tasks = ft.GlinerTasks(None, {"a": ["1", "2"]}, {"1": "fisheries", "2": "tariffs"})
    examples = ft.gliner_examples(ft.Split(["a", "b"], ["text a", "text b"], [["2"], ["1"]]), tasks)
    assert len(examples) == 1
    (classification,) = examples[0].classifications
    assert classification.labels == ["fisheries", "tariffs"]
    assert classification.true_label == ["tariffs"]
    assert classification.multi_label is True


class EchoGliner:
    """Scores each label by its position, as GLiNER's `classify_text` returns them."""

    def __init__(self, extra: str = "") -> None:
        self.extra = extra

    def classify_text(self, text: str, heads: dict, **options: object) -> dict:
        return {
            head: [
                {"label": name, "confidence": 1 / (rank + 2)}
                for rank, name in enumerate([*config["labels"], *([self.extra] if self.extra else [])])
            ]
            for head, config in heads.items()
        }

    def batch_classify_text(self, texts: list[str], heads: dict, **options: object) -> list[dict]:
        return [self.classify_text(text, heads) for text in texts]


def test_gliner_scores_cover_every_humset_label_in_batches() -> None:
    tasks = ft.GlinerTasks.for_humset(HUMSET_LABELS)
    args = argparse.Namespace(eval_batch=2, max_length=64)
    split = ft.Split(["a", "b", "c"], ["one", "two", "three"], [[], [], []])
    scores = ft.gliner_scores(EchoGliner(), split, tasks, args)
    assert set(scores) == {"a", "b", "c"}
    assert set(scores["a"]) == set(HUMSET_LABELS)


def test_gliner_scores_score_each_law_on_its_own_candidates() -> None:
    tasks = ft.GlinerTasks(None, {"a": ["1", "2"]}, {"1": "fisheries", "2": "tariffs"})
    args = argparse.Namespace(eval_batch=2, max_length=64)
    scores = ft.gliner_scores(EchoGliner(), ft.Split(["a", "b"], ["x", "y"], [[], []]), tasks, args)
    assert scores == {"a": {"1": 0.5, "2": pytest.approx(1 / 3)}}


def test_read_gliner_raises_on_a_missing_head_or_an_unknown_label() -> None:
    backs = {"labels": {"fisheries": "1"}}
    with pytest.raises(ValueError, match="no 'labels' head"):
        ft.read_gliner({}, backs)
    with pytest.raises(ValueError, match="was not given"):
        ft.read_gliner({"labels": [{"label": "mining", "confidence": 0.9}]}, backs)


def test_the_job_scripts_name_the_same_humset_versions_as_the_repository() -> None:
    assert ft.HUMSET_VERSIONS == ex.HUMSET_VERSIONS
    source = (PATH.parent / "train_xr_transformer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "HUMSET_VERSIONS" for t in node.targets)
    ]
    assert found == [ex.HUMSET_VERSIONS]
