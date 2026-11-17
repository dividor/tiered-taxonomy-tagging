"""Paths and settings. A missing setting raises rather than falling back to a default."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "outputs" / "results"
FIGURES = PROJECT_ROOT / "outputs" / "figures"


def data_root() -> Path:
    """Return the folder holding the downloaded test sets, which stays outside the repository."""
    load_dotenv(PROJECT_ROOT / ".env")
    value = os.environ.get("TAXONOMY_DATA_ROOT", "")
    if not value:
        raise RuntimeError("TAXONOMY_DATA_ROOT is not set; copy .env.example to .env and fill it in")
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(
            f"data folder not found: {path}; create it first, for example: mkdir -p {path}"
        )
    return path


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the experiment configuration."""
    with (path or PROJECT_ROOT / "configs" / "experiment.yaml").open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    return config
