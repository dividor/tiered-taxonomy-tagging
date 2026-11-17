"""Tests for training the local models before the GPU jobs: every named test set, and MultiEURLEX also with
the newer laws at the learning rate chosen earlier."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tiered_taxonomy import experiment as ex
from tiered_taxonomy import local_models as lm

CONFIG: dict[str, Any] = {
    "seed": 7,
    "text": {"input_words": 300},
    "trained": {"tuning_items": 10, "embedding_classifier": {"epochs": 2, "learning_rates": [0.1, 0.01]}},
    "reranker": {"newer_laws_learning_rate": 0.01},
}


def test_newer_laws_keep_every_setting_but_try_only_the_chosen_rate() -> None:
    settings = lm.newer_laws_settings(CONFIG)
    assert settings["embedding_classifier"] == {"epochs": 2, "learning_rates": [0.01]}
    assert settings["tuning_items"] == 10
    assert CONFIG["trained"]["embedding_classifier"]["learning_rates"] == [0.1, 0.01]


def test_train_covers_each_test_set_and_the_newer_laws_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trained: list[tuple[str, list[float], str]] = []

    def collect(data: Any, _root: Path, settings: Any, *_rest: Any) -> dict[str, Any]:
        trained.append((data.name, settings["embedding_classifier"]["learning_rates"], _rest[-2]))
        return {"omikuji": {}}

    monkeypatch.setattr(lm.sg, "collect_signals", collect)
    monkeypatch.setattr(
        lm.sg, "newer_laws_view", lambda data, _root: SimpleNamespace(name=f"{data.name}_newer")
    )
    monkeypatch.setattr(lm.runs, "make_bench", lambda *_: SimpleNamespace(index=None))
    details = lm.train(
        tmp_path,
        CONFIG,
        (ex.EURLEX, ex.HUMSET),
        loader=lambda _root, name: SimpleNamespace(name=name),
        on="cpu",
    )
    assert list(details) == [ex.EURLEX, f"{ex.EURLEX}_newer", ex.HUMSET]
    assert trained == [
        (ex.EURLEX, [0.1, 0.01], "cpu"),
        (f"{ex.EURLEX}_newer", [0.01], "cpu"),
        (ex.HUMSET, [0.1, 0.01], "cpu"),
    ]
    assert lm.device() in {"mps", "cuda", "cpu"}
