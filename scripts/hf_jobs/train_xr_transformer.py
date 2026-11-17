# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "libpecos==1.2.8",
#     "transformers>=4.36,<4.49",
#     "torch>=2.2",
#     "numpy<2",
#     "scipy",
#     "scikit-learn",
#     "pandas",
#     "pyarrow>=14,<19",
#     "huggingface_hub",
# ]
# ///
"""Train XR-Transformer (Zhang et al., 2021), the leading trained model in Kähler et al. (arXiv 2607.14882).

PECOS, which implements it, has no build for macOS, so this runs on a Linux machine with a GPU, such as a
Hugging Face Job. It downloads the public test sets itself, trains on the training split and writes each
development and test item's top labels, which the notebooks then score. Nothing here depends on the rest of
this repository.

    hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \\
        scripts/hf_jobs/train_xr_transformer.py --test-set eurlex --upload-to <your-name>/<private-dataset>

As in Kähler et al., documents are cut to their first 30,000 characters. The paper does not name its encoders,
so these are our choice: BERT for English laws, and multilingual BERT for HumSet's three languages.
`--smoke` runs a tiny sample with a tiny encoder, to check the script works before paying for a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from huggingface_hub import HfApi, hf_hub_download
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_CHARACTERS = 30_000
TOP_LABELS = 100
ENCODERS = {
    "eurlex": "bert-base-uncased",
    "humset": "bert-base-multilingual-cased",
    "humset_v2": "bert-base-multilingual-cased",
}
SMOKE_ENCODER = "prajjwal1/bert-tiny"
SMOKE_ITEMS = 300
HUMSET_LABEL_COLUMNS = ("sectors", "subpillars_1d", "subpillars_2d")
HUMSET_PILLAR_COLUMNS = ("pillars_1d", "pillars_2d")
# Downloads are pinned to the commits the November 2026 runs used.
HUMSET_REVISION = "0a9719d8cd1387d03373347edb6f53b9ea715299"  # HumSet's parquet conversion
# HumSet's versions by test set name: 1.0.0 as the repository uses it, 2.0.0 as HumBERT's code loads it.
HUMSET_VERSIONS = {"humset": "1.0.0", "humset_v2": "2.0.0"}
MULTI_EURLEX_REVISION = "d0318b7f68f43f7d77797437ce172d899254cd0f"


@dataclass
class Split:
    """Item ids, texts and label sets for one split."""

    ids: list[str]
    texts: list[str]
    labels: list[list[str]]


def read_eurlex_split(archive: Path, split: str, limit: int | None) -> Split:
    """Read English text and all gold labels for one MultiEURLEX split from the archive."""
    data = Split([], [], [])
    for record in _archive_records(archive, f"{split}.jsonl"):
        data.ids.append(record["celex_id"])
        data.texts.append((record["text"].get("en") or "")[:MAX_CHARACTERS])
        data.labels.append([str(label) for label in record["eurovoc_concepts"]["all_levels"]])
        if limit is not None and len(data.ids) >= limit:
            break
    return data


def _archive_records(archive: Path, name: str) -> Iterator[dict]:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name == name:
                handle = tar.extractfile(member)
                if handle is None:
                    raise ValueError(f"{name} in {archive} is not a regular file")
                for line in handle:
                    yield json.loads(line)
                return
    raise FileNotFoundError(f"{name} not found in {archive}")


def humset_row_labels(row: dict[str, Any]) -> list[str]:
    """Return one HumSet row's labels, as the repository's datasets module reads them: sectors, subpillars,
    and any pillar tagged without one of its subpillars, which HumSet 2.0.0 has and 1.0.0 does not."""
    labels = {str(label) for column in HUMSET_LABEL_COLUMNS for label in row[column]}
    implied = {label.split("->")[0] for label in labels if "->" in label}
    pillars = {str(label) for column in HUMSET_PILLAR_COLUMNS for label in row[column]}
    return sorted(labels | (pillars - implied))


def read_humset_split(path: Path, limit: int | None) -> Split:
    """Read one HumSet split."""
    rows = pd.read_parquet(path)
    if limit is not None:
        rows = rows.head(limit)
    labels = [humset_row_labels(row) for row in rows.to_dict("records")]
    return Split(rows["entry_id"].astype(str).tolist(), rows["excerpt"].astype(str).tolist(), labels)


def fetch_data(test_set: str, data_root: Path | None) -> dict[str, Path]:
    """Return the files for a test set, from a local folder if given, otherwise from Hugging Face."""
    if test_set == "eurlex":
        if data_root is not None:
            return {"archive": data_root / "multi_eurlex" / "multi_eurlex.tar.gz"}
        path = hf_hub_download(
            "coastalcph/multi_eurlex",
            "data/multi_eurlex.tar.gz",
            repo_type="dataset",
            revision=MULTI_EURLEX_REVISION,
        )
        return {"archive": Path(path)}
    files = {}
    for split in ("train", "validation", "test"):
        if data_root is not None:
            files[split] = data_root / "humset" / HUMSET_VERSIONS[test_set] / f"{split}.parquet"
        else:
            name = f"{HUMSET_VERSIONS[test_set]}/{split}/0000.parquet"
            downloaded = hf_hub_download(
                "nlp-thedeep/humset", name, repo_type="dataset", revision=HUMSET_REVISION
            )
            files[split] = Path(downloaded)
    return files


def load_splits(test_set: str, files: dict[str, Path], limit: int | None) -> dict[str, Split]:
    """Load the train, dev and test splits of a test set."""
    if test_set == "eurlex":
        return {
            split: read_eurlex_split(files["archive"], split, limit) for split in ("train", "dev", "test")
        }
    return {
        "train": read_humset_split(files["train"], limit),
        "dev": read_humset_split(files["validation"], limit),
        "test": read_humset_split(files["test"], limit),
    }


def label_matrix(labels: list[list[str]], names: list[str]) -> sp.csr_matrix:
    """Return items by labels as the sparse matrix PECOS expects."""
    positions = {name: i for i, name in enumerate(names)}
    rows, columns = [], []
    for row, item in enumerate(labels):
        for label in set(item):
            if label in positions:
                rows.append(row)
                columns.append(positions[label])
    values = np.ones(len(rows), dtype=np.float32)
    matrix = sp.csr_matrix((values, (rows, columns)), shape=(len(labels), len(names)), dtype=np.float32)
    matrix.sort_indices()
    return matrix


def as_features(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Return features as the sorted single-precision sparse matrix PECOS expects."""
    features = sp.csr_matrix(matrix, dtype=np.float32)
    features.sort_indices()
    return features


def train(
    splits: dict[str, Split], encoder: str, smoke: bool
) -> tuple[XTransformer, TfidfVectorizer, list[str]]:
    """Train XR-Transformer on the training split, leaving out items with no labels."""
    train_split = splits["train"]
    keep = [i for i, labels in enumerate(train_split.labels) if labels]
    texts = [train_split.texts[i] for i in keep]
    labels = [train_split.labels[i] for i in keep]
    names = sorted({label for item in labels for label in item})
    vectoriser = TfidfVectorizer(max_features=300_000, sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    features = as_features(vectoriser.fit_transform(texts))
    problem = MLProblemWithText(texts, label_matrix(labels, names), X_feat=features)
    params = XTransformer.TrainParams.from_dict(
        {
            "do_fine_tune": True,
            "matcher_params_chain": {
                "model_shortcut": encoder,
                "num_train_epochs": 1 if smoke else 3,
                "batch_size": 8 if smoke else 32,
                "learning_rate": 5e-5,
                "max_active_matching_labels": 1000,
                "save_steps": 1000,
                "logging_steps": 100,
            },
        },
        recursive=True,
    )
    predict = XTransformer.PredParams.from_dict(
        {"matcher_params_chain": {"truncate_length": 128 if smoke else 256}}, recursive=True
    )
    model = XTransformer.train(problem, train_params=params, pred_params=predict)
    return model, vectoriser, names


def rank_split(
    model: XTransformer, vectoriser: TfidfVectorizer, names: list[str], split: Split
) -> dict[str, dict[str, float]]:
    """Return each item's top labels and scores."""
    features = as_features(vectoriser.transform(split.texts))
    scores = model.predict(split.texts, X_feat=features, only_topk=TOP_LABELS).tocsr()
    ranked: dict[str, dict[str, float]] = {}
    for row, item_id in enumerate(split.ids):
        start, end = scores.indptr[row], scores.indptr[row + 1]
        ranked[item_id] = {
            names[column]: float(value)
            for column, value in zip(scores.indices[start:end], scores.data[start:end], strict=True)
        }
    return ranked


def main() -> None:
    """Train on one test set and save the rankings, locally and optionally to a private dataset repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--output", type=Path, default=Path("xr_output"))
    parser.add_argument(
        "--data-root", type=Path, help="use local copies rather than downloading the test sets"
    )
    parser.add_argument("--upload-to", help="private Hugging Face dataset repository for the results")
    parser.add_argument(
        "--smoke", action="store_true", help="tiny sample and tiny encoder, to check the script"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError("no GPU found: PECOS would silently run on the CPU, slowly and at full cost")
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
    limit = SMOKE_ITEMS if args.smoke else None
    encoder = SMOKE_ENCODER if args.smoke else ENCODERS[args.test_set]
    splits = load_splits(args.test_set, fetch_data(args.test_set, args.data_root), limit)
    model, vectoriser, names = train(splits, encoder, args.smoke)

    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in ("dev", "test"):
        path = args.output / f"xr_transformer_{args.test_set}_{split}.json"
        path.write_text(json.dumps(rank_split(model, vectoriser, names, splits[split])), encoding="utf-8")
        paths.append(path)
        print(path, path.stat().st_size, flush=True)
    if args.upload_to:
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(args.upload_to, repo_type="dataset", private=True, exist_ok=True)
        for path in paths:
            api.upload_file(
                path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.upload_to, repo_type="dataset"
            )


if __name__ == "__main__":
    main()
