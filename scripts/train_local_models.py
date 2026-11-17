#!/usr/bin/env python3
"""Train the models that run on this machine on every test set, before the GPU jobs, which read them.

    uv run python scripts/train_local_models.py

Omikuji, the embedding classifier, similar documents and label search, on MultiEURLEX (also with the
newer laws), HumSet 1.0.0 and HumSet 2.0.0. It needs the corpus embeddings (scripts/embed_corpus.py).
Finished steps are noted in the data folder and not repeated, so a stopped run resumes.
"""

from __future__ import annotations

from tiered_taxonomy import experiment as ex
from tiered_taxonomy import local_models as lm
from tiered_taxonomy.config import data_root, load_config


def main() -> int:
    details = lm.train(data_root(), load_config(), (ex.EURLEX, ex.HUMSET, ex.HUMSET_V2))
    for name, steps in details.items():
        print(name, "trained:", ", ".join(sorted(steps)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
