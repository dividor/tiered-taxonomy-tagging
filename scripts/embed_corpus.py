#!/usr/bin/env python3
"""Embed every split of both test sets with BGE-M3 on this machine. Rerunning resumes where it stopped.

uv run python scripts/embed_corpus.py                   # every test set
uv run python scripts/embed_corpus.py humset_v2         # only the test sets named
"""

from __future__ import annotations

import sys

from sentence_transformers import SentenceTransformer

from tiered_taxonomy import experiment as ex
from tiered_taxonomy.config import data_root, load_config
from tiered_taxonomy.corpus_embeddings import encode_split


def main() -> int:
    """Embed the splits, smallest first, so that early results arrive soon."""
    root = data_root()
    config = load_config()
    words = config["text"]["input_words"]
    encoder = SentenceTransformer(config["embeddings"]["model"])
    encoder.max_seq_length = 512
    wanted = sys.argv[1:] or [ex.HUMSET, ex.HUMSET_V2, ex.EURLEX]
    unknown = set(wanted) - {ex.EURLEX, *ex.HUMSET_VERSIONS}
    if unknown:
        raise SystemExit(f"unknown test sets: {sorted(unknown)}")
    loaded = [
        (ex.load_eurlex(root), 16) if name == ex.EURLEX else (ex.load_humset(root, name), 32)
        for name in wanted
    ]
    for data, batch in loaded:
        for split, items in (("test", data.test), ("dev", data.dev), ("train", data.train)):
            path = root / "corpus_embeddings" / data.name / f"{split}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            texts = [ex.input_text(item, words) for item in items]
            vectors = encode_split(encoder, texts, path, batch)
            print(data.name, split, vectors.shape, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
