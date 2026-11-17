"""Tests for the data folder setting."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tiered_taxonomy.config import data_root


def test_data_root_returns_the_folder_named_in_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAXONOMY_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path


def test_data_root_raises_when_the_setting_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAXONOMY_DATA_ROOT", "")
    with pytest.raises(RuntimeError, match=re.escape("copy .env.example to .env")):
        data_root()


def test_data_root_says_how_to_create_a_missing_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "not_yet"
    monkeypatch.setenv("TAXONOMY_DATA_ROOT", str(missing))
    with pytest.raises(FileNotFoundError, match=re.escape(f"mkdir -p {missing}")):
        data_root()
