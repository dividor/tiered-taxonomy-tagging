# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "transformers>=4.46,<5",
#     "peft>=0.13",
#     "accelerate>=1.0",
#     "sentencepiece",
#     "protobuf",
#     "torch>=2.2",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "huggingface_hub",
#     "safetensors",
#     "gliner2[train]>=2.1,<3",
# ]
# ///
"""Fine-tune a model on one test set on a Hugging Face Job, and write its ranked labels for later scoring.

Three kinds of model, chosen by `--kind`:

- classifier: an encoder with one output per label, as in HumBERT (arXiv 2305.16756) and MultiEURLEX
  (2109.00904).
- reranker: a cross-encoder reading a text and a label's name together, so it can score labels never seen
  in training. It needs candidate lists and label names prepared beforehand and uploaded to `--upload-to`.
- llm: a small LLM adapted with LoRA, with one output per label, as one branch of DualMLC (2609.12915). This
  is reduced from the paper: one model rather than two fused branches, and at most `--max-train` documents.
- gliner: GLiNER2.5, a small encoder that reads a text and every label's name together and scores each label
  in one pass. HumSet's labels fit in one pass, one head per published category; on MultiEURLEX each law's
  prepared candidate list is scored instead, as the reranker does. Development items are scored before
  training too, to record how it does without fine-tuning.

    hf jobs uv run --flavor a10g-small --timeout 4h --secrets HF_TOKEN scripts/hf_jobs/fine_tune.py \\
        --kind classifier --test-set humset --base-model nlp-thedeep/humbert \\
        --upload-to <you>/<private-dataset>

`--with-newer-laws` adds the earlier half of MultiEURLEX's development laws, by date, to training. Those laws
are then not used to choose the epoch. `--save-model`, for the classifier kind only, also uploads the weights
of the epoch whose rankings are kept, with the labels in the order of its outputs, so the model can be loaded
again later, for instance to publish it. Rankings are uploaded after every epoch, and a small
progress file every few minutes, because a job's logs and status cannot be relied on to show whether it is
working.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tarfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from gliner2 import AutoExtractor
from gliner2.training.data import Classification, InputExample
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MAX_CHARACTERS = 30_000
TOP_LABELS = 100
HUMSET_LABEL_COLUMNS = ("sectors", "subpillars_1d", "subpillars_2d")
HUMSET_PILLAR_COLUMNS = ("pillars_1d", "pillars_2d")
# Every download is pinned to a commit. These are the commits the November 2026 runs used: none of these
# repositories changed between those runs and the pinning.
HUMSET_REVISION = "0a9719d8cd1387d03373347edb6f53b9ea715299"  # HumSet's parquet conversion
# HumSet's versions by test set name: 1.0.0 as the repository uses it, 2.0.0 as HumBERT's code loads it.
HUMSET_VERSIONS = {"humset": "1.0.0", "humset_v2": "2.0.0"}
MULTI_EURLEX_REVISION = "d0318b7f68f43f7d77797437ce172d899254cd0f"
BASE_REVISIONS = {
    "nlp-thedeep/humbert": "74233ccb53ae90df6360dc976875dabd4601762e",
    "nlpaueb/legal-bert-base-uncased": "15b570cbf88259610b082a167dacc190124f60f6",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
    "prajjwal1/bert-tiny": "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837",
    "hf-internal-testing/tiny-random-LlamaForCausalLM": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "fastino/GLiNER2.5-multi-Decide": "36e46e4c94b7307e8bd59b23b6dfa36519a23269",
    "fastino/GLiNER2.5-Decide": "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
    "fastino/gliner2.5-small-v1": "b5d42bf7dc7d01c40304ae4ed43c2f13c325a75b",
}
# HumSet's 1D pillars, as in the repository's benchmarks module. The other pillars are 2D.
HUMSET_PILLARS_1D = frozenset(
    {
        "Casualties",
        "Context",
        "Covid-19",
        "Displacement",
        "Humanitarian Access",
        "Information And Communication",
        "Shock/Event",
    }
)


def pinned(model: str) -> str:
    """Return the commit to download a base model at. A model with none recorded raises."""
    if model not in BASE_REVISIONS:
        raise ValueError(f"no pinned commit for {model}: add the commit to use to BASE_REVISIONS")
    return BASE_REVISIONS[model]


SELECTION_ITEMS = 2_000
PROGRESS_SECONDS = 300
SMOKE_ITEMS = 200
SMOKE_MODEL = "prajjwal1/bert-tiny"
SMOKE_LLM = "hf-internal-testing/tiny-random-LlamaForCausalLM"
SMOKE_GLINER = "fastino/gliner2.5-small-v1"
GLINER_TASK = "labels"
HEAD_LEARNING_RATE = 1e-3
PAIRS_PER_DOCUMENT = 8


@dataclass
class Split:
    """Item ids, texts and label sets for one split."""

    ids: list[str]
    texts: list[str]
    labels: list[list[str]]

    def subset(self, positions: Sequence[int]) -> Split:
        """Return the items at the given positions."""
        return Split(
            [self.ids[i] for i in positions],
            [self.texts[i] for i in positions],
            [self.labels[i] for i in positions],
        )


def law_year(celex_id: str) -> int:
    """Return the year in a CELEX identifier such as 32012R0782."""
    match = re.match(r"^\d(\d{4})", celex_id)
    if match is None:
        raise ValueError(f"no year in {celex_id!r}")
    return int(match.group(1))


def date_order(ids: Sequence[str]) -> list[int]:
    """Return positions sorted by date, with ties broken by identifier."""
    return sorted(range(len(ids)), key=lambda i: (law_year(ids[i]), ids[i]))


def humset_with_parents(labels: Sequence[str]) -> list[str]:
    """Add each `Pillar->Subpillar` label's pillar, since the published scores cover pillars as well."""
    parents = {label.split("->")[0] for label in labels if "->" in label}
    return sorted(set(labels) | parents)


def without(split: Split, ids: Sequence[str]) -> Split:
    """Return a split without the given items, since items used in training need no scores."""
    left_out = set(ids)
    return split.subset([i for i, item_id in enumerate(split.ids) if item_id not in left_out])


def merge(first: Split, second: Split) -> Split:
    """Return two splits as one."""
    return Split(first.ids + second.ids, first.texts + second.texts, first.labels + second.labels)


def divide_development(test_set: str, dev: Split, with_newer: bool, seed: int) -> tuple[Split | None, Split]:
    """Return the development items added to training, if any, and those that may choose the epoch.

    For laws both come from date order, so that choosing the epoch always looks forward in time. Laws in the
    earlier half are only added to training when asked, but are never used to choose, so runs with and without
    them are judged on the same laws.
    """
    if test_set == "eurlex":
        ordered = date_order(dev.ids)
        middle = len(ordered) // 2
        added = dev.subset(ordered[:middle]) if with_newer else None
        return added, dev.subset(ordered[middle:][:SELECTION_ITEMS])
    if with_newer:
        raise ValueError("--with-newer-laws only applies to MultiEURLEX")
    order = list(range(len(dev.ids)))
    random.Random(seed).shuffle(order)
    return None, dev.subset(order[:SELECTION_ITEMS])


def mean_r_precision(gold: Sequence[Sequence[str]], rankings: Sequence[dict[str, float]]) -> float:
    """Return mean R-Precision as a percentage. Documents with no gold labels are left out."""
    values = []
    for truth, ranking in zip(gold, rankings, strict=True):
        if truth:
            top = sorted(ranking, key=lambda label: (-ranking[label], label))[: len(truth)]
            values.append(len(set(truth) & set(top)) / len(set(truth)))
    return 100 * float(np.mean(values))


def training_pairs(
    split: Split, candidates: dict[str, list[str]], rng: random.Random, per_document: int = PAIRS_PER_DOCUMENT
) -> list[tuple[int, str, float]]:
    """Return (item position, label, target) pairs: up to `per_document` gold labels and as many wrong
    candidates. Items with no candidate list are left out, so that targets stay balanced."""
    pairs: list[tuple[int, str, float]] = []
    for position, (item_id, gold) in enumerate(zip(split.ids, split.labels, strict=True)):
        if item_id not in candidates:
            continue
        truth = sorted(set(gold))
        wrong = [label for label in candidates[item_id] if label not in truth]
        rng.shuffle(truth)
        rng.shuffle(wrong)
        pairs.extend((position, label, 1.0) for label in truth[:per_document])
        pairs.extend((position, label, 0.0) for label in wrong[:per_document])
    rng.shuffle(pairs)
    return pairs


def targets(labels: Sequence[Sequence[str]], names: dict[str, int]) -> torch.Tensor:
    """Return items by labels as a dense matrix of zeros and ones. Labels not in `names` are ignored."""
    matrix = torch.zeros(len(labels), len(names))
    for row, item in enumerate(labels):
        for label in item:
            if label in names:
                matrix[row, names[label]] = 1.0
    return matrix


def top_labels(scores: np.ndarray, names: Sequence[str], count: int = TOP_LABELS) -> dict[str, float]:
    """Return one item's highest-scoring labels."""
    count = min(count, len(names))
    picks = np.argpartition(-scores, count - 1)[:count]
    return {names[int(i)]: float(scores[int(i)]) for i in picks}


class Reporter:
    """Writes results and a small progress file to a dataset repository, if one was given."""

    def __init__(self, repo: str | None, run: str, output: Path) -> None:
        self.repo, self.run, self.output = repo, run, output
        self.api = HfApi(token=os.environ["HF_TOKEN"]) if repo else None
        self.last = 0.0
        output.mkdir(parents=True, exist_ok=True)

    def progress(self, **state: object) -> None:
        """Report progress every few minutes. A failed report is printed and training carries on, because
        this file is only for watching the job; the results themselves are uploaded by `results`, which does
        raise."""
        now = time.monotonic()
        if now - self.last < PROGRESS_SECONDS:
            return
        self.last = now
        print(self.run, state, flush=True)
        if self.api is None or self.repo is None:
            return
        try:
            self.api.upload_file(
                path_or_fileobj=json.dumps({"time": time.time(), **state}).encode(),
                path_in_repo=f"progress/{self.run}.json",
                repo_id=self.repo,
                repo_type="dataset",
            )
        except OSError as error:
            print("progress upload failed:", error, flush=True)

    def results(self, name: str, content: object) -> None:
        """Save results and upload them."""
        path = self.output / f"{self.run}_{name}.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        print(path, path.stat().st_size, flush=True)
        self.upload(path)

    def upload(self, path: Path) -> None:
        """Upload one saved file, under its own name. A failed upload raises."""
        if self.api is not None and self.repo is not None:
            self.api.upload_file(
                path_or_fileobj=str(path), path_in_repo=path.name, repo_id=self.repo, repo_type="dataset"
            )


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


def read_eurlex(archive: Path, split: str, limit: int | None) -> Split:
    """Read English text and all gold labels for one MultiEURLEX split."""
    data = Split([], [], [])
    for record in _archive_records(archive, f"{split}.jsonl"):
        data.ids.append(record["celex_id"])
        data.texts.append((record["text"].get("en") or "")[:MAX_CHARACTERS])
        data.labels.append([str(label) for label in record["eurovoc_concepts"]["all_levels"]])
        if limit is not None and len(data.ids) >= limit:
            break
    return data


def humset_row_labels(row: dict[str, Any]) -> list[str]:
    """Return one HumSet row's labels, as the repository's datasets module reads them: sectors, subpillars,
    and any pillar tagged without one of its subpillars, which HumSet 2.0.0 has and 1.0.0 does not."""
    labels = {str(label) for column in HUMSET_LABEL_COLUMNS for label in row[column]}
    implied = {label.split("->")[0] for label in labels if "->" in label}
    pillars = {str(label) for column in HUMSET_PILLAR_COLUMNS for label in row[column]}
    return sorted(labels | (pillars - implied))


def read_humset(path: Path, limit: int | None) -> Split:
    """Read one HumSet split, with pillars as labels too."""
    rows = pd.read_parquet(path)
    if limit is not None:
        rows = rows.head(limit)
    labels = [humset_with_parents(humset_row_labels(row)) for row in rows.to_dict("records")]
    return Split(rows["entry_id"].astype(str).tolist(), rows["excerpt"].astype(str).tolist(), labels)


def load_splits(test_set: str, data_root: Path | None, limit: int | None) -> dict[str, Split]:
    """Load the train, dev and test splits, from a local folder if given, otherwise from Hugging Face."""
    if test_set == "eurlex":
        archive = (
            data_root / "multi_eurlex" / "multi_eurlex.tar.gz"
            if data_root
            else Path(
                hf_hub_download(
                    "coastalcph/multi_eurlex",
                    "data/multi_eurlex.tar.gz",
                    repo_type="dataset",
                    revision=MULTI_EURLEX_REVISION,
                )
            )
        )
        return {split: read_eurlex(archive, split, limit) for split in ("train", "dev", "test")}
    files = {"train": "train", "dev": "validation", "test": "test"}
    loaded = {}
    for split, name in files.items():
        path = (
            data_root / "humset" / HUMSET_VERSIONS[test_set] / f"{name}.parquet"
            if data_root
            else Path(
                hf_hub_download(
                    "nlp-thedeep/humset",
                    f"{HUMSET_VERSIONS[test_set]}/{name}/0000.parquet",
                    repo_type="dataset",
                    revision=HUMSET_REVISION,
                )
            )
        )
        loaded[split] = read_humset(path, limit)
    return loaded


class LabelModel(nn.Module):
    """An encoder or LLM with one output per label. An LLM's document vector is its last real token; an
    encoder's is its first."""

    def __init__(self, encoder: nn.Module, width: int, labels: int, last_token: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.last_token = last_token
        self.head = nn.Linear(width, labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return one logit per label."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.last_token:
            ends = attention_mask.sum(dim=1) - 1
            vectors = hidden[torch.arange(hidden.size(0)), ends]
        else:
            vectors = hidden[:, 0]
        return self.head(vectors.to(self.head.weight.dtype))


def save_label_model(model: LabelModel, names: Sequence[str], reporter: Reporter) -> Path:
    """Save a label model's weights, with the labels in the order of its outputs, and upload both."""
    path = reporter.output / f"{reporter.run}_model.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, path)
    print(path, path.stat().st_size, flush=True)
    reporter.upload(path)
    reporter.results("labels", list(names))
    return path


def amp_context(device: str) -> torch.amp.autocast:
    """Return mixed-precision autocast on a GPU, and do nothing elsewhere."""
    return torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")


def rank_with_model(
    model: LabelModel,
    tokenizer: object,
    split: Split,
    names: list[str],
    args: argparse.Namespace,
    device: str,
) -> dict[str, dict[str, float]]:
    """Return each item's top labels from a model with one output per label."""
    model.eval()
    ranked: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for start in range(0, len(split.ids), args.eval_batch):
            batch = tokenizer(
                split.texts[start : start + args.eval_batch],
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with amp_context(device):
                logits = model(batch["input_ids"], batch["attention_mask"])
            scores = torch.sigmoid(logits.float()).cpu().numpy()
            for item_id, row in zip(split.ids[start : start + args.eval_batch], scores, strict=True):
                ranked[item_id] = top_labels(row, names)
    return ranked


def make_label_model(args: argparse.Namespace, labels: int, device: str) -> tuple[LabelModel, object]:
    """Build the model and tokenizer for the classifier and LLM kinds."""
    revision = pinned(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    is_llm = args.kind == "llm"
    encoder = AutoModel.from_pretrained(
        args.base_model, revision=revision, torch_dtype=torch.bfloat16 if is_llm else None
    )
    width = encoder.config.hidden_size
    if is_llm:
        from peft import LoraConfig, get_peft_model

        encoder.gradient_checkpointing_enable()
        encoder.enable_input_require_grads()
        encoder = get_peft_model(
            encoder,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
    return LabelModel(encoder, width, labels, last_token=is_llm).to(device), tokenizer


def train_label_model(
    splits: dict[str, Split], args: argparse.Namespace, reporter: Reporter, device: str
) -> dict[str, object]:
    """Fine-tune a model with one output per label, keeping the rankings from the epoch best on development
    items."""
    added, selection = divide_development(args.test_set, splits["dev"], args.with_newer_laws, args.seed)
    pool = merge(splits["train"], added) if added else splits["train"]
    keep = [i for i, labels in enumerate(pool.labels) if labels]
    if args.max_train and len(keep) > args.max_train:
        keep = keep[-args.max_train :]
    pool = pool.subset(keep)
    names = sorted({label for item in pool.labels for label in item})
    positions = {name: i for i, name in enumerate(names)}
    model, tokenizer = make_label_model(args, len(names), device)
    parameters = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.learning_rate},
        {"params": list(model.head.parameters()), "lr": HEAD_LEARNING_RATE},
    ]
    optimiser = torch.optim.AdamW(parameters, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(pool.ids) / args.batch)
    schedule = get_linear_schedule_with_warmup(optimiser, steps // 20, steps)
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    best, history = -1.0, []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(pool.ids)))
        random.Random(args.seed + epoch).shuffle(order)
        for step, start in enumerate(range(0, len(order), args.batch)):
            picks = order[start : start + args.batch]
            batch = tokenizer(
                [pool.texts[i] for i in picks],
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            wanted = targets([pool.labels[i] for i in picks], positions).to(device)
            with amp_context(device):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = loss_function(logits.float(), wanted).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad()
            reporter.progress(
                epoch=epoch,
                step=step,
                loss=round(float(loss), 3),
                seconds=round(time.monotonic() - started),
                documents_per_second=round(
                    (epoch * len(order) + start) / max(time.monotonic() - started, 1), 1
                ),
            )
        value = mean_r_precision(
            selection.labels, list(rank_with_model(model, tokenizer, selection, names, args, device).values())
        )
        history.append(
            {"epoch": epoch, "development_mrp": round(value, 3), "seconds": round(time.monotonic() - started)}
        )
        print(history[-1], flush=True)
        reporter.results("history", history)
        if value > best:
            best = value
            for split in ("dev", "test"):
                reporter.results(split, rank_with_model(model, tokenizer, splits[split], names, args, device))
            if args.save_model:
                save_label_model(model, names, reporter)
    return {
        "best_development_mrp": best,
        "history": history,
        "documents": len(pool.ids),
        "labels": len(names),
    }


def load_prepared(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, dict[str, list[str]]]]:
    """Read label names and candidate lists, from a local folder or the dataset repository."""

    def fetch(name: str) -> dict:
        if args.prepared:
            return json.loads((args.prepared / name).read_text(encoding="utf-8"))
        if not args.prepared_revision:
            raise ValueError("give --prepared-revision, the commit holding the prepared files, or --prepared")
        path = hf_hub_download(
            args.upload_to,
            f"prepared/{name}",
            repo_type="dataset",
            revision=args.prepared_revision,
            token=os.environ["HF_TOKEN"],
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))

    names = fetch(f"{args.test_set}_label_names.json")
    candidates = {
        split: fetch(f"{args.test_set}_candidates_{split}.json") for split in ("train", "dev", "test")
    }
    return names, candidates


def score_pairs(
    model: nn.Module,
    tokenizer: object,
    split: Split,
    candidates: dict[str, list[str]],
    names: dict[str, str],
    args: argparse.Namespace,
    device: str,
) -> dict[str, dict[str, float]]:
    """Return each item's candidates with the reranker's scores. Pairs are sorted by length to run faster."""
    pairs = [(i, label) for i, item_id in enumerate(split.ids) for label in candidates.get(item_id, [])]
    pairs.sort(key=lambda pair: len(split.texts[pair[0]]))
    ranked: dict[str, dict[str, float]] = {item_id: {} for item_id in split.ids}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), args.eval_batch):
            chunk = pairs[start : start + args.eval_batch]
            batch = tokenizer(
                [split.texts[i] for i, _ in chunk],
                [names[label] for _, label in chunk],
                truncation="only_first",
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with amp_context(device):
                logits = model(**batch).logits.squeeze(-1)
            for (i, label), value in zip(chunk, torch.sigmoid(logits.float()).cpu().tolist(), strict=True):
                ranked[split.ids[i]][label] = value
    return ranked


def train_reranker(
    splits: dict[str, Split], args: argparse.Namespace, reporter: Reporter, device: str
) -> dict[str, object]:
    """Fine-tune a cross-encoder on gold labels and wrong candidates, then score every candidate."""
    names, candidates = load_prepared(args)
    added, selection = divide_development(args.test_set, splits["dev"], args.with_newer_laws, args.seed)
    pool = merge(splits["train"], added) if added else splits["train"]
    prepared = {**candidates["train"], **candidates["dev"]}
    rng = random.Random(args.seed)
    pairs = [pair for pair in training_pairs(pool, prepared, rng) if pair[1] in names][: args.max_pairs]
    revision = pinned(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, revision=revision, num_labels=1, ignore_mismatched_sizes=True
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(pairs) / args.batch)
    schedule = get_linear_schedule_with_warmup(optimiser, steps // 20, steps)
    loss_function = nn.BCEWithLogitsLoss()
    best, history = -1.0, []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        for step, start in enumerate(range(0, len(pairs), args.batch)):
            chunk = pairs[start : start + args.batch]
            batch = tokenizer(
                [pool.texts[i] for i, _, _ in chunk],
                [names[label] for _, label, _ in chunk],
                truncation="only_first",
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            wanted = torch.tensor([target for _, _, target in chunk], device=device)
            with amp_context(device):
                logits = model(**batch).logits.squeeze(-1)
            loss = loss_function(logits.float(), wanted)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad()
            reporter.progress(
                epoch=epoch,
                step=step,
                of=steps,
                loss=round(float(loss), 3),
                seconds=round(time.monotonic() - started),
            )
        chosen = score_pairs(model, tokenizer, selection, candidates["dev"], names, args, device)
        value = mean_r_precision(selection.labels, [chosen[item_id] for item_id in selection.ids])
        history.append(
            {"epoch": epoch, "development_mrp": round(value, 3), "seconds": round(time.monotonic() - started)}
        )
        print(history[-1], flush=True)
        reporter.results("history", history)
        if value > best:
            best = value
            for split in ("dev", "test"):
                wanted = without(splits[split], added.ids) if added and split == "dev" else splits[split]
                reporter.results(
                    split,
                    score_pairs(model, tokenizer, wanted, candidates[split], names, args, device),
                )
    return {"best_development_mrp": best, "history": history, "pairs": len(pairs)}


Heads = dict[str, dict[str, Any]]
Backs = dict[str, dict[str, str]]


def humset_heads(labels: Sequence[str]) -> dict[str, list[str]]:
    """Divide HumSet's labels into its five published categories, as the repository's benchmarks module
    does. Raises if a 1D pillar is missing, which would mean the wrong data."""
    found = set(labels)
    pillars = {label.split("->")[0] for label in found if "->" in label}
    if not HUMSET_PILLARS_1D <= pillars:
        raise ValueError(f"unknown pillars: {sorted(HUMSET_PILLARS_1D - pillars)}")
    roots = {label for label in found if "->" not in label}
    subpillars = {label for label in found if "->" in label}
    one_d = {label for label in subpillars if label.split("->")[0] in HUMSET_PILLARS_1D}
    return {
        "Sectors": sorted(roots - pillars),
        "Pillars 1D": sorted(HUMSET_PILLARS_1D),
        "Subpillars 1D": sorted(one_d),
        "Pillars 2D": sorted(pillars - HUMSET_PILLARS_1D),
        "Subpillars 2D": sorted(subpillars - one_d),
    }


def gliner_head(labels: Sequence[str], names: dict[str, str] | None) -> tuple[dict[str, Any], dict[str, str]]:
    """Return one multi-label head that scores every label, and the map from the text GLiNER reads back to
    label ids. HumSet's `Pillar->Subpillar` ids read as `Pillar: Subpillar`; MultiEURLEX ids read as their
    names. Raises if two labels would read the same, because their scores could not be told apart."""
    texts = {label: names[label] if names is not None else label.replace("->", ": ") for label in labels}
    back = {text: label for label, text in texts.items()}
    if len(back) != len(texts):
        raise ValueError("two labels read the same to GLiNER, so their scores cannot be told apart")
    return {"labels": list(back), "multi_label": True, "cls_threshold": 0.0}, back


class GlinerTasks:
    """The heads GLiNER is given for each item: HumSet's five categories for every excerpt, or each law's
    own candidate list on MultiEURLEX."""

    def __init__(
        self,
        fixed: tuple[Heads, Backs] | None,
        candidates: dict[str, list[str]],
        names: dict[str, str] | None,
    ) -> None:
        self.fixed, self.candidates, self.names = fixed, candidates, names

    @classmethod
    def for_humset(cls, labels: Sequence[str]) -> GlinerTasks:
        """Build the five category heads from every label the splits use."""
        heads: Heads = {}
        backs: Backs = {}
        for category, members in humset_heads(labels).items():
            heads[category], backs[category] = gliner_head(members, None)
        return cls((heads, backs), {}, None)

    def has(self, item_id: str) -> bool:
        """Return whether this item can be scored: always on HumSet, only with a candidate list otherwise."""
        return self.fixed is not None or item_id in self.candidates

    def get(self, item_id: str) -> tuple[Heads, Backs]:
        """Return the heads for one item. Raises if a law has no candidate list."""
        if self.fixed is not None:
            return self.fixed
        if item_id not in self.candidates:
            raise ValueError(f"no candidate list for {item_id}")
        labels = [label for label in self.candidates[item_id] if label in (self.names or {})]
        head, back = gliner_head(labels, self.names)
        return {GLINER_TASK: head}, {GLINER_TASK: back}


def gliner_examples(split: Split, tasks: GlinerTasks) -> list[InputExample]:
    """Return one training example per item that can be scored, with its gold labels in every head."""
    examples = []
    for item_id, text, gold in zip(split.ids, split.texts, split.labels, strict=True):
        if not tasks.has(item_id):
            continue
        heads, backs = tasks.get(item_id)
        truth = set(gold)
        classifications = [
            Classification(
                task=head,
                labels=list(config["labels"]),
                true_label=[seen for seen, label in backs[head].items() if label in truth],
                multi_label=True,
            )
            for head, config in heads.items()
        ]
        examples.append(InputExample(text=text, classifications=classifications))
    return examples


def read_gliner(result: dict, backs: Backs) -> dict[str, float]:
    """Return label id to confidence from one result. Raises if a head is missing or names a label it was
    not given."""
    scores: dict[str, float] = {}
    for head, back in backs.items():
        if head not in result:
            raise ValueError(f"GLiNER returned no {head!r} head")
        for entry in result[head]:
            if entry["label"] not in back:
                raise ValueError(f"GLiNER returned {entry['label']!r}, which {head!r} was not given")
            scores[back[entry["label"]]] = float(entry["confidence"])
    return scores


def gliner_scores(
    model: Any, split: Split, tasks: GlinerTasks, args: argparse.Namespace
) -> dict[str, dict[str, float]]:
    """Score every item that can be scored: HumSet in batches, since its heads are the same for every
    excerpt, and laws one at a time, since each has its own candidates."""
    ranked: dict[str, dict[str, float]] = {}
    keep = [i for i, item_id in enumerate(split.ids) if tasks.has(item_id)]
    if tasks.fixed is not None:
        heads, backs = tasks.fixed
        for start in range(0, len(keep), args.eval_batch):
            chunk = keep[start : start + args.eval_batch]
            results = model.batch_classify_text(
                [split.texts[i] for i in chunk],
                heads,
                batch_size=args.eval_batch,
                include_confidence=True,
                max_len=args.max_length,
            )
            for i, result in zip(chunk, results, strict=True):
                ranked[split.ids[i]] = read_gliner(result, backs)
        return ranked
    for i in keep:
        heads, backs = tasks.get(split.ids[i])
        result = model.classify_text(split.texts[i], heads, include_confidence=True, max_len=args.max_length)
        ranked[split.ids[i]] = read_gliner(result, backs)
    return ranked


def gliner_tasks(splits: dict[str, Split], args: argparse.Namespace) -> GlinerTasks:
    """Return the heads for this test set."""
    if args.test_set in HUMSET_VERSIONS:
        return GlinerTasks.for_humset(
            sorted({label for split in splits.values() for labels in split.labels for label in labels})
        )
    names, candidates = load_prepared(args)
    return GlinerTasks(
        None, {key: value for lists in candidates.values() for key, value in lists.items()}, names
    )


def train_gliner(
    splits: dict[str, Split], args: argparse.Namespace, reporter: Reporter, device: str
) -> dict[str, object]:
    """Score development items without fine-tuning, fine-tune GLiNER2.5 on the training items, then score
    development and test items."""
    tasks = gliner_tasks(splits, args)
    added, selection = divide_development(args.test_set, splits["dev"], args.with_newer_laws, args.seed)
    pool = merge(splits["train"], added) if added else splits["train"]
    keep = [i for i, labels in enumerate(pool.labels) if labels]
    if args.max_train and len(keep) > args.max_train:
        keep = keep[-args.max_train :]
    examples = gliner_examples(pool.subset(keep), tasks)
    model = AutoExtractor.from_pretrained(args.base_model, revision=pinned(args.base_model)).to(device)
    started = time.monotonic()
    history = []

    def judge(stage: str) -> None:
        chosen = gliner_scores(model, selection, tasks, args)
        scored = [i for i, item_id in enumerate(selection.ids) if item_id in chosen]
        value = mean_r_precision(
            [selection.labels[i] for i in scored], [chosen[selection.ids[i]] for i in scored]
        )
        history.append(
            {"stage": stage, "development_mrp": round(value, 3), "seconds": round(time.monotonic() - started)}
        )
        print(history[-1], flush=True)
        reporter.results("history", history)

    judge("without fine-tuning")
    config = TrainingConfig(
        output_dir=str(args.output / "gliner_checkpoints"),
        num_epochs=args.epochs,
        batch_size=args.batch,
        eval_batch_size=args.eval_batch,
        encoder_lr=args.learning_rate,
        eval_strategy="no",
        save_best=False,
        logging_steps=100,
        seed=args.seed,
        fp16=device == "cuda",
        max_len=args.max_length,
        num_workers=2,
    )
    ExtractorTrainer(model, config).train(train_data=examples)
    model.eval()
    judge("fine-tuned")
    for split in ("dev", "test"):
        wanted = without(splits[split], added.ids) if added and split == "dev" else splits[split]
        reporter.results(split, gliner_scores(model, wanted, tasks, args))
    return {"history": history, "training_examples": len(examples)}


def parse_arguments() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("classifier", "reranker", "llm", "gliner"), required=True)
    parser.add_argument("--test-set", choices=("eurlex", *HUMSET_VERSIONS), required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--run-name", help="names the output files; defaults to kind and test set")
    parser.add_argument("--with-newer-laws", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-train", type=int, help="use at most this many of the newest training documents"
    )
    parser.add_argument("--max-pairs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("fine_tune_output"))
    parser.add_argument(
        "--data-root", type=Path, help="use local copies rather than downloading the test sets"
    )
    parser.add_argument("--prepared", type=Path, help="use local copies of the reranker's prepared files")
    parser.add_argument(
        "--prepared-revision", help="the commit of --upload-to that holds the reranker's prepared files"
    )
    parser.add_argument("--upload-to", help="private Hugging Face dataset repository for the results")
    parser.add_argument(
        "--save-model", action="store_true", help="classifier only: also upload the kept epoch's weights"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="tiny sample and tiny model, to check the script"
    )
    return parser.parse_args()


def main() -> None:
    """Fine-tune one model on one test set and write its rankings."""
    args = parse_arguments()
    if args.save_model and args.kind != "classifier":
        raise ValueError(f"--save-model is for the classifier kind, not {args.kind}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.smoke:
        raise RuntimeError("no GPU found: this would run very slowly and at full cost")
    print("device:", torch.cuda.get_device_name(0) if device == "cuda" else "cpu", flush=True)
    torch.manual_seed(args.seed)
    if args.smoke:
        args.base_model = {"llm": SMOKE_LLM, "gliner": SMOKE_GLINER}.get(args.kind, SMOKE_MODEL)
        args.epochs, args.batch, args.eval_batch, args.max_length = 1, 8, 8, 64
    run = args.run_name or f"{args.kind}_{args.test_set}"
    reporter = Reporter(args.upload_to, run, args.output)
    splits = load_splits(args.test_set, args.data_root, SMOKE_ITEMS if args.smoke else None)
    print({name: len(split.ids) for name, split in splits.items()}, flush=True)
    train = {"reranker": train_reranker, "gliner": train_gliner}.get(args.kind, train_label_model)
    summary = train(splits, args, reporter, device)
    reporter.results(
        "summary", {**summary, "base_model": args.base_model, "with_newer_laws": args.with_newer_laws}
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
