#!/usr/bin/env python3
"""Run the fine-tuning jobs on Hugging Face Jobs, wait for them, and record what the notebooks should read.

    uv run python scripts/gpu_jobs.py --dry-run   # list the runs still to do, and the most they could cost
    uv run python scripts/gpu_jobs.py             # launch them, wait, and record the dataset's commit

Every run uploads to the dataset named by TAXONOMY_JOBS_DATASET in .env, which is created private if it does
not exist. A run whose outputs are already there is not launched again, so the script can be rerun after a
failure. The reranker reads candidate labels from the local models, so if a run that reads them is still to
do, they are written and uploaded first (scripts/hf_jobs/prepare_inputs.py). The token needs permission to run
jobs and to write to the dataset; jobs are billed to your Hugging Face account.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence

from huggingface_hub import HfApi, JobInfo

from tiered_taxonomy import gpu_jobs as gj
from tiered_taxonomy.config import PROJECT_ROOT, RESULTS, data_root, load_config
from tiered_taxonomy.models import hf_token


def plan(runs: Sequence[gj.Run], prices: dict[str, float]) -> None:
    """Print the runs still to do, with the most each could cost if it ran to its time limit."""
    for run in runs:
        limit = float(run.timeout.removesuffix("h"))
        print(f"  {run.name}: {run.flavor}, up to {run.timeout} (at most ${limit * prices[run.flavor]:.2f})")


def prepare(api: HfApi, dataset: str, runs: Sequence[gj.Run]) -> str:
    """Write and upload the reranker's inputs for the test sets the given runs read, and return the dataset's
    commit that holds them."""
    with tempfile.TemporaryDirectory() as folder:
        script = str(gj.JOB_SCRIPTS / "prepare_inputs.py")
        sets = gj.prepared_test_sets(runs)
        command = [sys.executable, script, "--output", folder, "--upload-to", dataset, "--test-sets", *sets]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return str(api.dataset_info(dataset).sha)


def launch(api: HfApi, run: gj.Run, dataset: str, prepared: str | None) -> JobInfo:
    """Start one run on Hugging Face Jobs and return its job."""
    secrets = {"HF_TOKEN": hf_token()}
    if run.humbert_version is not None:
        script = (gj.JOB_SCRIPTS / run.script).read_text(encoding="utf-8")
        command = gj.humbert_command(run, dataset, script)
        return api.run_job(
            image=gj.HUMBERT_IMAGE, command=command, flavor=run.flavor, timeout=run.timeout, secrets=secrets
        )
    return api.run_uv_job(
        str(gj.JOB_SCRIPTS / run.script),
        script_args=gj.uv_arguments(run, dataset, prepared),
        flavor=run.flavor,
        timeout=run.timeout,
        secrets=secrets,
    )


def wait(api: HfApi, jobs: dict[str, JobInfo], poll_seconds: int) -> dict[str, JobInfo]:
    """Wait until every job has stopped, printing each as it does. Raises if any did not complete."""
    pending, stopped = dict(jobs), {}
    while pending:
        time.sleep(poll_seconds)
        for name, job in list(pending.items()):
            current = api.inspect_job(job_id=job.id)
            stage = gj.stage_name(current.status.stage)
            if stage in gj.STOPPED:
                print(f"{name}: {stage}", flush=True)
                stopped[name] = current
                del pending[name]
    failed = sorted(name for name, job in stopped.items() if gj.stage_name(job.status.stage) != gj.FINISHED)
    if failed:
        raise RuntimeError(f"these runs did not complete: {', '.join(failed)}; see `hf jobs ps -a`")
    return stopped


def save_costs(runs: Sequence[gj.Run], stopped: dict[str, JobInfo], prices: dict[str, float]) -> None:
    """Add the cost records of the runs just finished to outputs/results/gpu_jobs.json."""
    path = RESULTS / "gpu_jobs.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    finished = {
        name: (job.id, str(job.flavor), gj.hours(job.started_at, job.finished_at))
        for name, job in stopped.items()
    }
    merged = gj.merge_costs(records, runs, finished, prices)
    path.write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
    print("costs written to", path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list the runs still to do and stop")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    config, root, dataset = load_config(), data_root(), gj.jobs_dataset()
    prices = config["gpu_jobs"]["prices"]
    api = HfApi(token=hf_token())
    runs = gj.runs_from_config(config)
    if not args.dry_run:
        api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    present = (
        api.list_repo_files(dataset, repo_type="dataset")
        if api.repo_exists(dataset, repo_type="dataset")
        else []
    )
    todo = gj.missing_runs(runs, present)
    print(f"{len(runs) - len(todo)} of {len(runs)} runs are already in {dataset}; {len(todo)} to do:")
    plan(todo, prices)
    if args.dry_run:
        return 0
    if todo:
        prepared = prepare(api, dataset, todo) if gj.prepared_test_sets(todo) else None
        jobs = {run.name: launch(api, run, dataset, prepared) for run in todo}
        for name, job in jobs.items():
            print(f"launched {name}: {job.url}", flush=True)
        stopped = wait(api, jobs, args.poll_seconds)
        still = gj.missing_runs(runs, api.list_repo_files(dataset, repo_type="dataset"))
        if still:
            raise RuntimeError(f"runs finished without all their outputs: {', '.join(r.name for r in still)}")
        save_costs(todo, stopped, prices)
    revision = str(api.dataset_info(dataset).sha)
    print("recorded", gj.save_state(root, dataset, revision), "at commit", revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
