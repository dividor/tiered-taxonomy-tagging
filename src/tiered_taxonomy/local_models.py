"""Train the models that run on this machine (Omikuji, the embedding classifier, similar documents and label
search) on every test set, before the GPU jobs, because the fine-tuned reranker reads their suggestions.

Finished steps are noted in the data folder and not repeated, so the notebooks that later ask for these
models find them already trained.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from tiered_taxonomy import experiment as ex
from tiered_taxonomy import runs
from tiered_taxonomy import signals as sg

Loader = Callable[[Path, str], ex.TestSet]


def device() -> str:
    """Return the fastest device this machine has for the embedding classifier."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def newer_laws_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the local models' settings for MultiEURLEX with the newer laws: the same, except that the
    embedding classifier uses the learning rate chosen on training laws alone in notebook 08, rather than
    trying every rate again."""
    trained = config["trained"]
    rate = config["reranker"]["newer_laws_learning_rate"]
    return {**trained, "embedding_classifier": {**trained["embedding_classifier"], "learning_rates": [rate]}}


def load(root: Path, name: str) -> ex.TestSet:
    """Load a test set by name."""
    return ex.load_eurlex(root) if name == ex.EURLEX else ex.load_humset(root, name)


def train(
    root: Path,
    config: Mapping[str, Any],
    names: tuple[str, ...],
    loader: Loader = load,
    on: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Train the local models on each named test set, then on MultiEURLEX with the newer laws if it is
    named, and return each set's training details."""
    chosen = on or device()
    seed, words = config["seed"], config["text"]["input_words"]
    details = {}
    for name in names:
        data = loader(root, name)
        bench = runs.make_bench(data, root, config)
        details[name] = sg.collect_signals(data, root, config["trained"], seed, words, chosen, bench.index)
        if name == ex.EURLEX:
            view = sg.newer_laws_view(data, root)
            details[view.name] = sg.collect_signals(
                view, root, newer_laws_settings(config), seed, words, chosen, bench.index
            )
    return details
