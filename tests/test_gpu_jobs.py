"""Tests for the fine-tuning run list: reading it, deciding what is still to do, launching, costing and
recording which commit of the jobs dataset the notebooks read."""

from __future__ import annotations

import base64
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tiered_taxonomy import gpu_jobs as gj
from tiered_taxonomy.config import load_config

CONFIG = {
    "gpu_jobs": {
        "runs": {
            "classifier_toy": {
                "script": "fine_tune.py",
                "flavor": "a10g-small",
                "timeout": "4h",
                "base": "some/model",
                "args": ["--kind", "classifier", "--test-set", "toy", "--epochs", 3],
                "outputs": ["classifier_toy_dev.json", "classifier_toy_test.json"],
            },
            "reranker_toy": {
                "script": "fine_tune.py",
                "flavor": "a10g-small",
                "timeout": "4h",
                "base": "other/model",
                "needs_prepared": True,
                "args": ["--kind", "reranker", "--test-set", "toy"],
                "outputs": ["reranker_toy_test.json"],
            },
            "humbert_reproduction_full": {
                "script": "reproduce_humbert.py",
                "flavor": "a10g-small",
                "timeout": "6h",
                "base": "nlp-thedeep/humbert",
                "humbert_version": "2.0.0",
                "outputs": ["humbert_reproduction/full/run.json"],
            },
        }
    }
}


def runs() -> dict[str, gj.Run]:
    return {run.name: run for run in gj.runs_from_config(CONFIG)}


def test_runs_are_read_in_order_with_their_arguments_as_text() -> None:
    read = runs()
    assert list(read) == ["classifier_toy", "reranker_toy", "humbert_reproduction_full"]
    assert read["classifier_toy"].args[-1] == "3" and not read["classifier_toy"].needs_prepared
    assert (
        read["reranker_toy"].needs_prepared and read["humbert_reproduction_full"].humbert_version == "2.0.0"
    )
    broken = {"gpu_jobs": {"runs": {"x": {**CONFIG["gpu_jobs"]["runs"]["classifier_toy"], "outputs": []}}}}
    with pytest.raises(ValueError, match="lists no outputs"):
        gj.runs_from_config(broken)


def test_the_projects_own_run_list_reads_and_every_reranker_needs_prepared_inputs() -> None:
    project = gj.runs_from_config(load_config())
    assert len(project) == len({run.name for run in project})
    assert all(run.needs_prepared for run in project if run.name.startswith("reranker_"))
    assert gj.prepared_test_sets(project) == ["eurlex", "humset", "humset_v2"]


def test_a_run_is_still_to_do_until_every_output_is_in_the_dataset() -> None:
    present = ["classifier_toy_dev.json", "classifier_toy_test.json", "humbert_reproduction/full/run.json"]
    assert [run.name for run in gj.missing_runs(runs().values(), present)] == ["reranker_toy"]
    assert [run.name for run in gj.missing_runs(runs().values(), present[:1])] == list(runs())


def test_the_jobs_dataset_comes_from_the_environment_and_must_name_an_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gj, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(gj.DATASET_SETTING, "someone/jobs")
    assert gj.jobs_dataset() == "someone/jobs"
    monkeypatch.setenv(gj.DATASET_SETTING, "")
    with pytest.raises(RuntimeError, match="TAXONOMY_JOBS_DATASET is not set"):
        gj.jobs_dataset()


def test_uv_arguments_add_the_dataset_and_the_prepared_commit_only_where_read() -> None:
    read = runs()
    assert gj.uv_arguments(read["classifier_toy"], "me/jobs", "abc")[-2:] == ["--upload-to", "me/jobs"]
    reranker = gj.uv_arguments(read["reranker_toy"], "me/jobs", "abc")
    assert reranker[-4:] == ["--upload-to", "me/jobs", "--prepared-revision", "abc"]
    with pytest.raises(ValueError, match="no commit of them was given"):
        gj.uv_arguments(read["reranker_toy"], "me/jobs", None)
    assert gj.prepared_test_sets(read.values()) == ["toy"]


def test_the_humbert_command_carries_the_script_and_names_the_run_and_version() -> None:
    command = gj.humbert_command(runs()["humbert_reproduction_full"], "me/jobs", "print('hello')")
    assert command[:2] == ["bash", "-c"]
    assert base64.b64encode(b"print('hello')").decode() in command[2]
    assert "--run-name full --version 2.0.0 --upload-to me/jobs" in command[2]
    with pytest.raises(ValueError, match="not a HumBERT reproduction run"):
        gj.humbert_command(runs()["classifier_toy"], "me/jobs", "")


def test_hours_round_a_jobs_running_time_and_refuse_a_job_that_never_finished() -> None:
    start = datetime(2026, 11, 13, 17, 47, 46, tzinfo=UTC)
    finish = datetime(2026, 11, 13, 18, 28, 8, tzinfo=UTC)
    assert gj.hours(start, finish) == 0.67
    with pytest.raises(ValueError, match="start and a finish"):
        gj.hours(start, None)


def test_costs_count_a_runs_extra_launches_and_merging_keeps_earlier_records() -> None:
    records = {
        "prices": {"a10g-small": 1.0, "t4-small": 0.4},
        "runs": {
            "old": {"job": "j0", "flavor": "a10g-small", "hours": 2.0, "base": "b"},
            "classifier_toy": {
                "job": "j1",
                "flavor": "a10g-small",
                "hours": 1.0,
                "base": "b",
                "also_paid": [{"job": "j2", "flavor": "t4-small", "hours": 0.5, "why": "smoke test"}],
            },
        },
        "unrecorded": ["reranker_toy"],
    }
    assert gj.run_cost(records, "classifier_toy") == pytest.approx(1.2)
    merged = gj.merge_costs(
        records, [runs()["reranker_toy"]], {"reranker_toy": ("j3", "a10g-small", 3.0)}, {"a10g-small": 1.0}
    )
    assert merged["runs"]["old"]["hours"] == 2.0 and merged["unrecorded"] == []
    assert merged["runs"]["reranker_toy"] == {
        "job": "j3",
        "flavor": "a10g-small",
        "hours": 3.0,
        "base": "other/model",
    }
    assert gj.stage_name("COMPLETED") == "COMPLETED"


def test_the_recorded_commit_is_read_back_and_must_match_the_dataset_in_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gj, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(gj.DATASET_SETTING, "me/jobs")
    with pytest.raises(FileNotFoundError, match=re.escape("uv run python scripts/gpu_jobs.py")):
        gj.load_state(tmp_path)
    gj.save_state(tmp_path, "me/jobs", "abc123")
    assert gj.load_state(tmp_path) == {"dataset": "me/jobs", "revision": "abc123"}
    asked = []

    def download(repo: str, filename: str, **options: str) -> str:
        asked.append((repo, filename, options["revision"]))
        return str(tmp_path / filename)

    monkeypatch.setattr(gj, "hf_hub_download", download)
    assert gj.fetch(tmp_path, "run_test.json") == tmp_path / "run_test.json"
    assert asked == [("me/jobs", "run_test.json", "abc123")]
    monkeypatch.setenv(gj.DATASET_SETTING, "someone/else")
    with pytest.raises(ValueError, match="records me/jobs"):
        gj.load_state(tmp_path)
    assert json.loads(gj.state_path(tmp_path).read_text(encoding="utf-8"))["revision"] == "abc123"
