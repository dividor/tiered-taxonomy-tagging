#!/usr/bin/env python3
"""Prepare what the fine-tuned reranker needs, before its runs are launched. `scripts/gpu_jobs.py` runs this.

1. The local models (Omikuji, the embedding classifier, similar documents and label search) are trained on
   each test set named, and on MultiEURLEX also with the earlier half of the development laws, by date, added
   to training, saved as the `eurlex_newer` test set. Finished steps are not repeated
   (`scripts/train_local_models.py` has usually done them already).
2. The reranker reads a text and a label's name together, so it needs candidate labels to read. This writes
   each label's name and candidate lists for training, development and test items, from the local models, and
   uploads them to the private dataset the jobs read from.

The November 2026 runs read training candidates made before ties between equally common labels were broken
by label. Ties then fell in set order, which changes from one process to the next, so those lists cannot be
rebuilt exactly: rebuilding them changes about 0.2% (HumSet) to 0.5% (MultiEURLEX) of training candidate
labels. The development and test lists, which decide what the reranker scores, are rebuilt exactly.

    uv run python scripts/hf_jobs/prepare_inputs.py --output <folder> --upload-to <you>/<dataset>

Leave out `--upload-to` to write the files without uploading them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy import datasets as ds
from tiered_taxonomy import experiment as ex
from tiered_taxonomy import local_models as lm
from tiered_taxonomy import runs
from tiered_taxonomy import signals as sg
from tiered_taxonomy.config import data_root, load_config

LOCAL = ["omikuji", "embedding_classifier", "knn", "label_search"]
# How many labels per source to try, to show the share of gold labels the candidate lists hold.
TRIED_PER_SOURCE = (10, 15, 20, 30, 50)


def report_coverage(name: str, sources: dict[str, sg.Rankings], items: list[ds.Item]) -> None:
    """Print the share of gold labels the candidate lists hold, for each list length tried."""
    total = sum(len(item.labels) for item in items)
    for per in TRIED_PER_SOURCE:
        lists = sg.candidate_lists(sources, [item.id for item in items], per)
        found = sum(len(set(item.labels) & set(lists.get(item.id, []))) for item in items)
        mean = np.mean([len(value) for value in lists.values()])
        print(f"{name}: {per} per source holds {found / total:.3f} of gold labels, {mean:.0f} labels a list")


def reranker_inputs(root: Path, config: dict, name: str) -> dict[str, object]:
    """Return the reranker's label names and candidate lists for one test set. On MultiEURLEX, development
    laws in the earlier half get candidates from models trained without them, and the later half from models
    trained with the newer laws."""
    settings = config["reranker"]
    data = ex.load_eurlex(root) if name == ex.EURLEX else ex.load_humset(root, name)
    bench = runs.make_bench(data, root, config)
    per = settings["per_source"][ex.HUMSET if ex.is_humset(name) else name]
    if name == ex.EURLEX:
        early, late = bm.split_by_date(data.dev)
        newer = sg.NEWER_LAWS_SUFFIX.format(name)
        before = sg.load_sources(root, name, LOCAL, "dev")
        after = sg.load_sources(root, newer, LOCAL, "dev")
        test = sg.load_sources(root, newer, LOCAL, "test")
        report_coverage(name, after, late)
        dev_lists = {
            **sg.candidate_lists(before, [item.id for item in early], per),
            **sg.candidate_lists(after, [item.id for item in late], per),
        }
    else:
        dev = sg.load_sources(root, name, LOCAL, "dev")
        test = sg.load_sources(root, name, LOCAL, "test")
        report_coverage(name, dev, data.dev[: settings["coverage_items"]])
        dev_lists = sg.candidate_lists(dev, [item.id for item in data.dev], per)
    train_lists = sg.training_candidates(
        data, root, bench.index, settings["train_items"], settings["train_per_source"], config["seed"]
    )
    return {
        f"{name}_label_names.json": sg.label_names(data.taxonomy),
        f"{name}_candidates_train.json": train_lists,
        f"{name}_candidates_dev.json": dev_lists,
        f"{name}_candidates_test.json": sg.candidate_lists(test, [item.id for item in data.test], per),
    }


def main() -> int:
    """Prepare the newer-laws models, then write the reranker's inputs and upload them if asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload-to")
    parser.add_argument(
        "--test-sets", nargs="+", choices=(ex.EURLEX, *ex.HUMSET_VERSIONS), default=[ex.EURLEX, ex.HUMSET]
    )
    args = parser.parse_args()
    root = data_root()
    config = load_config()
    lm.train(root, config, tuple(args.test_sets))
    args.output.mkdir(parents=True, exist_ok=True)
    api = HfApi() if args.upload_to else None
    for name in args.test_sets:
        for filename, content in reranker_inputs(root, config, name).items():
            path = args.output / filename
            path.write_text(json.dumps(content), encoding="utf-8")
            print("wrote", path, flush=True)
            if api is not None:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"prepared/{filename}",
                    repo_id=args.upload_to,
                    repo_type="dataset",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
