"""Tests for the HumBERT reproduction job's own logic. The job file stands alone, so it is loaded from its
path; the authors' trainer is imported only inside the job."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parent.parent / "scripts" / "hf_jobs" / "reproduce_humbert.py"
spec = importlib.util.spec_from_file_location("reproduce_humbert", PATH)
assert spec is not None and spec.loader is not None
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)

ROW = {
    "entry_id": 12,
    "excerpt": "Floods cut roads.",
    "sectors": ["Logistics"],
    "pillars_1d": ["Shock/event"],
    "pillars_2d": [],
    "subpillars_1d": ["Shock/event->Type and characteristics"],
    "subpillars_2d": None,
}


def test_targets_name_labels_as_the_papers_code_groups_them() -> None:
    assert job.targets(ROW) == [
        "first_level_tags->sectors->Logistics",
        "first_level_tags->pillars_1d->Shock/event",
        "subpillars_1d->Shock/event->Type and characteristics",
    ]


def test_frame_rows_pass_labels_as_text_the_trainer_parses_back() -> None:
    (row,) = job.frame_rows([ROW])
    assert row["entry_id"] == "12"
    assert ast.literal_eval(row["target_classification"]) == row["labels"] == job.targets(ROW)
    hash(row["target_classification"])


def test_keep_share_is_seeded_and_keeps_at_least_one() -> None:
    rows = [{"n": n} for n in range(100)]
    assert job.keep_share(rows, 0.1, 7) == job.keep_share(rows, 0.1, 7)
    assert len(job.keep_share(rows, 0.1, 7)) == 10
    assert len(job.keep_share(rows[:3], 0.01, 7)) == 1


def test_prediction_records_pair_gold_and_predictions_and_raise_on_a_mismatch() -> None:
    rows = job.frame_rows([ROW])
    (record,) = job.prediction_records(rows, [["b", "a"]])
    assert record == {"entry_id": "12", "gold": job.targets(ROW), "predicted": ["a", "b"]}
    with pytest.raises(ValueError, match="1 test entries but 2 predictions"):
        job.prediction_records(rows, [[], []])


def test_settings_are_the_paper_notebooks() -> None:
    assert job.SETTINGS["learning_rate"] == 3e-5 and job.SETTINGS["max_len"] == 200
    assert job.SETTINGS["n_epochs"] == 3 and job.SETTINGS["train_batch_size"] == 8
    assert job.FILES["test"] == "data/test_1_1.jsonl"


def test_ranking_records_keep_likely_labels_and_raise_on_short_columns() -> None:
    rows = [{"entry_id": "a"}, {"entry_id": "b"}]
    records = job.ranking_records(rows, {"x": [0.91234, 0.0001], "y": [0.002, 0.5]})
    assert records == [
        {"entry_id": "a", "ranking": {"x": 0.9123, "y": 0.002}},
        {"entry_id": "b", "ranking": {"y": 0.5}},
    ]
    with pytest.raises(ValueError, match="1 probabilities for 2 entries"):
        job.ranking_records(rows, {"x": [0.5]})


def test_both_versions_name_every_split() -> None:
    assert set(job.VERSIONS) == {"1.0.0", "2.0.0"}
    assert all(set(files) == {"train", "validation", "test"} for files in job.VERSIONS.values())
    assert job.VERSIONS["1.0.0"]["test"] == "data/test.jsonl"
