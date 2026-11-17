"""Reproduce HumBERT's fine-tuned HumSet result with its authors' trainer, on the data it most likely used.

HumBERT's paper (Tamagnone et al., IJCAI 2023, arXiv 2305.16756) reports macro F1 per HumSet category. Its
code, github.com/the-deep-nlp/bias-aware-humanitarian-entry-classification, loads `nlp-thedeep/humset` with
no version, which since February 2023 means version 2.0.0: a different test set from version 1.0.0, which
this repository uses. This job runs the authors' packaged trainer, github.com/the-deep-nlp/classifier-trainer
at a fixed commit, with the paper notebook's settings, on version 2.0.0's own train, validation and
test files, or with --version 1.0.0 on the version this repository uses. It changes none of their code.

It runs in an old PyTorch container because the trainer needs PyTorch Lightning 1.x and a transformers
release that still exports AdamW. scripts/gpu_jobs.py starts it. It uploads,
under `humbert_reproduction/<run name>/` in the dataset given by --upload-to: the trainer's own results
table, each test entry's predicted labels, every validation and test entry's label probabilities, the
thresholds chosen on validation, and the settings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HUMSET = "nlp-thedeep/humset"
# The dataset commit the reproduction read; its last change was in May 2023.
HUMSET_REVISION = "0dcf8311cae5a45dee0ded3fea676a1551c1cd68"
# Each version's files, as the dataset's loading script maps them. The paper's code loaded 2.0.0, the default;
# this repository's other HumSet results use 1.0.0.
VERSIONS = {
    "1.0.0": {"train": "data/train.jsonl", "validation": "data/validation.jsonl", "test": "data/test.jsonl"},
    "2.0.0": {
        "train": "data/train_1_1.jsonl",
        "validation": "data/validation_1_1.jsonl",
        "test": "data/test_1_1.jsonl",
    },
}
FILES = VERSIONS["2.0.0"]
# Probabilities below this are left out of the saved rankings, which only need each entry's likely labels.
KEEP_PROBABILITY = 0.001
# The settings in the paper's models_training.ipynb. The focal loss settings are the trainer's defaults.
SETTINGS = {
    "architecture_setup": "multiabel_architecture",
    "backbone_name": "nlp-thedeep/humbert",
    "n_epochs": 3,
    "dropout": 0.2,
    "weight_decay": 1e-2,
    "learning_rate": 3e-5,
    "max_len": 200,
    "n_freezed_layers": 1,
    "train_batch_size": 8,
    "val_batch_size": 16,
    "loss_gamma": 1,
    "proportions_pow": 0.5,
}
# The share of each split kept in a smoke test, and its epochs, as in the paper notebook's `sample` setting.
SAMPLE = {"train": 0.05, "validation": 0.1, "test": 0.1, "n_epochs": 1}


def targets(row: Mapping[str, Any]) -> list[str]:
    """Return an entry's labels named as the paper's code names them, so its averages group them alike."""
    names = [f"first_level_tags->sectors->{tag}" for tag in row["sectors"] or []]
    names += [f"first_level_tags->pillars_2d->{tag}" for tag in row["pillars_2d"] or []]
    names += [f"first_level_tags->pillars_1d->{tag}" for tag in row["pillars_1d"] or []]
    names += [f"subpillars_1d->{tag}" for tag in row["subpillars_1d"] or []]
    names += [f"subpillars_2d->{tag}" for tag in row["subpillars_2d"] or []]
    return names


def frame_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the columns the trainer reads, keeping each entry's id to match predictions back. The trainer
    drops duplicate rows first, which needs hashable cells, and parses a label list from its text form, as the
    paper's notebooks passed it; so the labels go in as text and are also kept as a list."""
    made = []
    for row in rows:
        labels = targets(row)
        made.append(
            {
                "entry_id": str(row["entry_id"]),
                "excerpt": row["excerpt"],
                "target_classification": str(labels),
                "labels": labels,
            }
        )
    return made


def keep_share(rows: Sequence[dict[str, Any]], share: float, seed: int) -> list[dict[str, Any]]:
    """Return a seeded random share of rows, at least one."""
    count = max(1, round(len(rows) * share))
    return random.Random(seed).sample(list(rows), count)


def prediction_records(rows: Sequence[Mapping[str, Any]], predicted: Sequence[Sequence[str]]) -> list[dict]:
    """Pair each test entry with its gold and predicted labels. Raises if the counts differ."""
    if len(rows) != len(predicted):
        raise ValueError(f"{len(rows)} test entries but {len(predicted)} predictions")
    return [
        {"entry_id": row["entry_id"], "gold": row["labels"], "predicted": sorted(labels)}
        for row, labels in zip(rows, predicted, strict=True)
    ]


def ranking_records(
    rows: Sequence[Mapping[str, Any]], probabilities: Mapping[str, Sequence[float]]
) -> list[dict[str, Any]]:
    """Return each entry's label probabilities, keeping those of at least KEEP_PROBABILITY, rounded to four
    places. Raises if a label's probabilities do not cover every entry."""
    for tag, values in probabilities.items():
        if len(values) != len(rows):
            raise ValueError(f"{tag}: {len(values)} probabilities for {len(rows)} entries")
    return [
        {
            "entry_id": row["entry_id"],
            "ranking": {
                tag: round(float(values[i]), 4)
                for tag, values in probabilities.items()
                if float(values[i]) >= KEEP_PROBABILITY
            },
        }
        for i, row in enumerate(rows)
    ]


def read_split(version: str, name: str) -> list[dict[str, Any]]:
    """Download one split of a HumSet version and return its rows."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HUMSET, VERSIONS[version][name], repo_type="dataset", revision=HUMSET_REVISION)
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def upload(folder: Path, repo: str, prefix: str) -> None:
    """Upload every file in a folder to the dataset, under a prefix."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    for path in sorted(folder.iterdir()):
        api.upload_file(
            path_or_fileobj=str(path), path_in_repo=f"{prefix}/{path.name}", repo_id=repo, repo_type="dataset"
        )
        print("uploaded", path.name, flush=True)


def main() -> None:
    """Train, choose thresholds on validation, score on test with the trainer's own code, and upload."""
    import pandas as pd
    import torch
    from classifier_trainer import ClassifierTrainer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--upload-to", required=True)
    parser.add_argument("--sample", action="store_true", help="a smoke test on a small share, for one epoch")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--version", choices=sorted(VERSIONS), default="2.0.0")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    started = time.time()
    splits = {name: frame_rows(read_split(args.version, name)) for name in FILES}
    settings = dict(SETTINGS)
    if args.sample:
        splits = {name: keep_share(rows, SAMPLE[name], args.seed) for name, rows in splits.items()}
        settings["n_epochs"] = SAMPLE["n_epochs"]
    print({name: len(rows) for name, rows in splits.items()}, settings, flush=True)
    frames = {name: pd.DataFrame(rows) for name, rows in splits.items()}

    trainer = ClassifierTrainer()
    model = trainer.train_classification_model(
        frames["train"][["excerpt", "target_classification"]],
        frames["validation"][["excerpt", "target_classification"]],
        results_dir="results",
        enable_checkpointing=False,
        save_model=False,
        **settings,
    )
    results = trainer.generate_test_results(
        frames["test"][["excerpt", "target_classification"]], generate_visulizations=False, save_results=False
    )
    predicted = trainer.generate_test_predictions(frames["test"].excerpt.tolist())

    out = Path("reproduction")
    out.mkdir(exist_ok=True)
    results.to_csv(out / "results.csv", index=False)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for record in prediction_records(splits["test"], predicted):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for name in ("validation", "test"):
        raw = model.custom_predict(frames[name].excerpt.tolist(), setup="raw_predictions")
        probabilities = {tag: values.tolist() for tag, values in raw.items()}
        with open(out / f"rankings_{name}.jsonl", "w", encoding="utf-8") as handle:
            for record in ranking_records(splits[name], probabilities):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    thresholds = {name: float(value) for name, value in model.optimal_thresholds.items()}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    run = {
        "settings": settings,
        "sample": args.sample,
        "version": args.version,
        "seed": args.seed,
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "hours": round((time.time() - started) / 3600, 3),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(results.to_string(), flush=True)
    upload(out, args.upload_to, f"humbert_reproduction/{args.run_name}")


if __name__ == "__main__":
    main()
