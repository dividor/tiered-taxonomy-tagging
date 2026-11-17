#!/usr/bin/env python3
"""Write the report's label trees and example records to outputs/results/label_browser.json.

    uv run python scripts/export_label_browser.py

It reads the data folder: both taxonomies, the training counts and the same test samples every notebook
uses, from which it takes three records each, and the training count at or below which a label is rare.
Run it again only if the data or the samples change; the report reads the file it writes, so the report
itself needs no data folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tiered_taxonomy import experiment as ex
from tiered_taxonomy import label_browser as lb
from tiered_taxonomy.config import data_root, load_config

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = 3
# Laws run to thousands of words, so they are cut; the humanitarian excerpts are short and shown whole.
LAW_WORDS, EXCERPT_WORDS = 120, 400


def main() -> int:
    """Write the label trees and example records for both test sets."""
    root, config = data_root(), load_config()
    seed, samples = config["seed"], config["samples"]
    eurlex, humset = ex.load_eurlex(root), ex.load_humset(root)
    laws = ex.sample_eurlex(
        eurlex,
        samples["eurlex"]["normal_documents"],
        samples["eurlex"]["long_documents"],
        samples["eurlex"]["long_threshold_words"],
        seed,
    )
    excerpts = ex.sample_humset(humset, samples["humset"]["per_language"], seed)
    body = {
        "eurlex": {
            "tree": lb.label_tree(eurlex.taxonomy, eurlex.train_counts),
            "train_documents": len(eurlex.train),
            "examples": [
                lb.example_record(item, LAW_WORDS, lb.EURLEX_LINK)
                for item in lb.pick_examples(laws, EXAMPLES, (4, 7))
            ],
            "rare_at_most": ex.rare_cutoff(eurlex.train_counts),
        },
        "humset": {
            "tree": lb.label_tree(humset.taxonomy, humset.train_counts),
            "train_documents": len(humset.train),
            "examples": [
                lb.example_record(item, EXCERPT_WORDS)
                for item in lb.pick_examples(excerpts, EXAMPLES, (2, 4))
            ],
            "rare_at_most": ex.rare_cutoff(humset.train_counts),
        },
    }
    out = REPO / "outputs" / "results" / "label_browser.json"
    out.write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
