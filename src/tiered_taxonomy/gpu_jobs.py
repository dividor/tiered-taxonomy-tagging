"""The fine-tuning runs on Hugging Face Jobs: which runs exist, how each is launched, whether its outputs are
already in the jobs dataset, and where a notebook downloads them from.

Every run uploads its outputs to one Hugging Face dataset of yours, named by `TAXONOMY_JOBS_DATASET` in
`.env`. `scripts/gpu_jobs.py` launches the runs whose outputs are missing, waits for them, and then records
the dataset's commit in the data folder, so every notebook reads the same files. No hand edits needed.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from tiered_taxonomy.config import PROJECT_ROOT

DATASET_SETTING = "TAXONOMY_JOBS_DATASET"
STATE_FILE = "gpu_jobs.json"
JOB_SCRIPTS = PROJECT_ROOT / "scripts" / "hf_jobs"
# The HumBERT authors' trainer runs in its own image, with the packages it was written for.
HUMBERT_IMAGE = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
HUMBERT_TRAINER = (
    "https://github.com/the-deep-nlp/classifier-trainer/archive/"
    "21679ae9c64dc1542696b18433a3cc8048777301.tar.gz"
)
HUMBERT_PACKAGES = (
    "pytorch-lightning==1.9.5 transformers==4.30.2 torchmetrics==0.11.4 numpy==1.24.4 pandas==1.5.3 "
    "scikit-learn==1.2.2 matplotlib==3.7.1 seaborn==0.12.2 sentencepiece==0.1.99 protobuf==3.20.3 "
    "tqdm==4.65.0 huggingface_hub==0.16.4"
)


@dataclass(frozen=True)
class Run:
    """One fine-tuning run, as `configs/experiment.yaml` describes it under `gpu_jobs.runs`."""

    name: str
    script: str
    flavor: str
    timeout: str
    args: tuple[str, ...]
    outputs: tuple[str, ...]
    base: str
    needs_prepared: bool = False
    humbert_version: str | None = None


def runs_from_config(config: Mapping[str, Any]) -> list[Run]:
    """Return every run in the configuration, in the order given. A run without outputs raises, because then
    nothing could show it had finished."""
    found = []
    for name, entry in config["gpu_jobs"]["runs"].items():
        if not entry.get("outputs"):
            raise ValueError(f"gpu run {name} lists no outputs")
        found.append(
            Run(
                name=name,
                script=entry["script"],
                flavor=entry["flavor"],
                timeout=entry["timeout"],
                args=tuple(str(arg) for arg in entry.get("args", [])),
                outputs=tuple(entry["outputs"]),
                base=entry["base"],
                needs_prepared=bool(entry.get("needs_prepared", False)),
                humbert_version=entry.get("humbert_version"),
            )
        )
    return found


def jobs_dataset() -> str:
    """Return the Hugging Face dataset the runs upload to. Raises if `.env` does not name one."""
    load_dotenv(PROJECT_ROOT / ".env")
    value = os.environ.get(DATASET_SETTING, "")
    if "/" not in value:
        raise RuntimeError(
            f"{DATASET_SETTING} is not set: in .env, name a dataset of yours, like <you>/taxonomy-jobs"
        )
    return value


def missing_runs(runs: Iterable[Run], present: Iterable[str]) -> list[Run]:
    """Return the runs with at least one output not among the dataset's files."""
    files = set(present)
    return [run for run in runs if not set(run.outputs) <= files]


def uv_arguments(run: Run, dataset: str, prepared: str | None) -> list[str]:
    """Return the script arguments for a run of one of this repository's job scripts."""
    arguments = [*run.args, "--upload-to", dataset]
    if run.needs_prepared:
        if prepared is None:
            raise ValueError(f"{run.name} reads the prepared inputs, but no commit of them was given")
        arguments += ["--prepared-revision", prepared]
    return arguments


def humbert_command(run: Run, dataset: str, script: str) -> list[str]:
    """Return the shell command that installs the HumBERT authors' trainer and runs `script` (the text of
    `reproduce_humbert.py`) with it. The script is passed encoded, so no file needs mounting."""
    if run.humbert_version is None:
        raise ValueError(f"{run.name} is not a HumBERT reproduction run")
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    run_name = run.name.removeprefix("humbert_reproduction_")
    return [
        "bash",
        "-c",
        "set -euo pipefail; python --version; "
        f"pip install --quiet {HUMBERT_PACKAGES} && pip install --quiet --no-deps {HUMBERT_TRAINER} && "
        f"echo {encoded} | base64 -d > /tmp/reproduce_humbert.py && "
        f"python /tmp/reproduce_humbert.py --run-name {run_name} --version {run.humbert_version} "
        f"--upload-to {dataset}",
    ]


def hours(started: datetime | None, finished: datetime | None) -> float:
    """Return a finished job's running time in hours, to two decimals. A job without both times raises."""
    if started is None or finished is None:
        raise ValueError("a job needs a start and a finish time to be costed")
    return round((finished - started).total_seconds() / 3600, 2)


def record(run: Run, job_id: str, flavor: str, running_hours: float) -> dict[str, Any]:
    """Return the cost record of one finished run, as the results file keeps it."""
    return {"job": job_id, "flavor": flavor, "hours": running_hours, "base": run.base}


def run_cost(records: Mapping[str, Any], run: str) -> float:
    """Return what one run cost in US dollars, from its hours and its machine's price, counting any launch it
    also needed, such as one that failed first. `records` is the results file `gpu_jobs.json`."""
    record_ = records["runs"][run]
    return float(
        sum(
            job["hours"] * records["prices"][job["flavor"]]
            for job in [record_, *record_.get("also_paid", [])]
        )
    )


FINISHED = "COMPLETED"
STOPPED = frozenset({"COMPLETED", "ERROR", "CANCELED", "DELETED"})


def stage_name(stage: object) -> str:
    """Return a job stage's name, whether the library gives it as text or as an enumeration."""
    return str(getattr(stage, "value", stage))


def prepared_test_sets(runs: Iterable[Run]) -> list[str]:
    """Return the test sets whose reranker inputs the given runs read."""
    return sorted({run.args[run.args.index("--test-set") + 1] for run in runs if run.needs_prepared})


def merge_costs(
    records: Mapping[str, Any],
    runs: Iterable[Run],
    finished: Mapping[str, tuple[str, str, float]],
    prices: Mapping[str, float],
) -> dict[str, Any]:
    """Return the cost records with the runs just finished added. `finished` gives each run's job id,
    machine and hours; records of runs that were already done are kept."""
    merged: dict[str, Any] = {
        "prices": dict(prices),
        "runs": dict(records.get("runs", {})),
        "unrecorded": [name for name in records.get("unrecorded", []) if name not in finished],
    }
    for run in runs:
        job_id, flavor, running_hours = finished[run.name]
        merged["runs"][run.name] = record(run, job_id, flavor, running_hours)
    return merged


def state_path(root: Path) -> Path:
    """Return where the data folder keeps the dataset commit that the notebooks read."""
    return root / STATE_FILE


def save_state(root: Path, dataset: str, revision: str) -> Path:
    """Record the dataset and the commit holding every run's outputs."""
    path = state_path(root)
    path.write_text(json.dumps({"dataset": dataset, "revision": revision}, indent=1), encoding="utf-8")
    return path


def load_state(root: Path) -> dict[str, str]:
    """Return the recorded dataset and commit. Raises until `scripts/gpu_jobs.py` has finished, or if the
    record names a different dataset from `.env`."""
    path = state_path(root)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run `uv run python scripts/gpu_jobs.py` first")
    state: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    if state["dataset"] != jobs_dataset():
        raise ValueError(f"{path} records {state['dataset']}, but .env names {jobs_dataset()}")
    return state


def fetch(root: Path, filename: str) -> Path:
    """Download one run output from the jobs dataset at the recorded commit, and return its local path."""
    state = load_state(root)
    return Path(hf_hub_download(state["dataset"], filename, repo_type="dataset", revision=state["revision"]))
