"""Load the two test sets as items with text, gold labels and the group an item belongs to."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tiered_taxonomy.taxonomy import Taxonomy, from_paths

HUMSET_LABEL_COLUMNS = ("sectors", "subpillars_1d", "subpillars_2d")
# HumSet 2.0.0 tags some excerpts with a pillar but none of its subpillars (1.0.0 never does), so the pillar
# columns are read too: a pillar that no subpillar of the excerpt implies is a label of its own.
HUMSET_PILLAR_COLUMNS = ("pillars_1d", "pillars_2d")
EURLEX_LEVELS = {1: "level_1", 2: "level_2", 3: "level_3"}
EURLEX_LANGUAGES = ("en", "fr", "es")


@dataclass(frozen=True)
class Item:
    """One piece of text to label."""

    id: str
    text: str
    language: str
    labels: frozenset[str]
    group: str = ""
    labels_by_tier: Mapping[int, frozenset[str]] = field(default_factory=dict)
    project: str = ""


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words."""
    return len(text.split())


def first_words(text: str, limit: int) -> str:
    """Return the first `limit` words of a text."""
    return " ".join(text.split()[:limit])


def chunk_words(text: str, size: int) -> list[str]:
    """Split a text into consecutive chunks of at most `size` words. An empty text has no chunks."""
    words = text.split()
    return [" ".join(words[start : start + size]) for start in range(0, len(words), size)]


def humset_labels(row: Mapping[Hashable, Any]) -> frozenset[str]:
    """Return an entry's labels. Sectors are roots; subpillars are written `Pillar->Subpillar`; a pillar
    tagged without any of its subpillars is a label on its own."""
    labels = {str(label) for column in HUMSET_LABEL_COLUMNS for label in row[column]}
    implied = {label.split("->")[0] for label in labels if "->" in label}
    pillars = {str(label) for column in HUMSET_PILLAR_COLUMNS for label in row[column]}
    return frozenset(labels | (pillars - implied))


def humset_taxonomy(rows: pd.DataFrame) -> Taxonomy:
    """Build HumSet's taxonomy from the labels in the data: pillars over subpillars, and flat sectors."""
    labels = {label for row in rows.to_dict("records") for label in humset_labels(row)}
    return from_paths(sorted(labels))


def load_humset_rows(root: Path, split: str, version: str = "1.0.0") -> pd.DataFrame:
    """Read one split of a HumSet version. Version 2.0.0 has no project id, so its project title stands in."""
    rows = pd.read_parquet(root / "humset" / version / f"{split}.parquet")
    if "project_id" not in rows.columns:
        if version == "1.0.0":
            raise ValueError("HumSet 1.0.0 should have a project_id column")
        rows = rows.assign(project_id=rows["project_title"])
    return rows


def humset_items(rows: pd.DataFrame, taxonomy: Taxonomy) -> list[Item]:
    """Turn HumSet rows into items, grouped by source document, with gold labels for tiers 1 and 2."""
    items = []
    for row in rows.to_dict("records"):
        labels = humset_labels(row)
        by_tier = {tier: frozenset(taxonomy.roll_up(labels, tier)) for tier in (1, 2)}
        items.append(
            Item(
                str(row["entry_id"]),
                str(row["excerpt"]),
                str(row["lang"]),
                labels,
                str(row["lead_id"]),
                by_tier,
                str(row["project_id"]),
            )
        )
    return items


def stream_eurlex(archive: Path, split: str) -> Iterator[dict[str, Any]]:
    """Yield the raw records of one MultiEURLEX split from the archive, without unpacking it to disk."""
    wanted = f"{split}.jsonl"
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name == wanted:
                handle = tar.extractfile(member)
                if handle is None:
                    raise ValueError(f"{wanted} in {archive} is not a regular file")
                for line in handle:
                    yield json.loads(line)
                return
    raise FileNotFoundError(f"{wanted} not found in {archive}")


def eurlex_rows(records: Iterable[Mapping[str, Any]], languages: Iterable[str]) -> pd.DataFrame:
    """Keep only the named languages and label levels from raw MultiEURLEX records."""
    rows = []
    wanted = tuple(languages)
    for record in records:
        concepts = record["eurovoc_concepts"]
        row: dict[str, Any] = {"celex_id": record["celex_id"], "date": record["publication_date"]}
        row.update({f"text_{language}": record["text"].get(language) or "" for language in wanted})
        row.update({name: list(concepts[name]) for name in (*EURLEX_LEVELS.values(), "all_levels")})
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_eurlex(root: Path) -> dict[str, Path]:
    """Write compact Parquet files for English in every split, plus French and Spanish for dev and test."""
    folder = root / "multi_eurlex"
    paths = {}
    for split, languages in (("train", ("en",)), ("dev", EURLEX_LANGUAGES), ("test", EURLEX_LANGUAGES)):
        path = folder / f"{split}.parquet"
        if not path.exists():
            eurlex_rows(stream_eurlex(folder / "multi_eurlex.tar.gz", split), languages).to_parquet(path)
        paths[split] = path
    return paths


def load_eurlex_rows(root: Path, split: str) -> pd.DataFrame:
    """Read one prepared MultiEURLEX split."""
    return pd.read_parquet(root / "multi_eurlex" / f"{split}.parquet")


def eurlex_items(rows: pd.DataFrame, language: str = "en") -> list[Item]:
    """Turn MultiEURLEX rows into items, with the dataset's own level labels by tier. `experiment.load_eurlex`
    replaces those with labels rolled up through the EuroVoc tree, which the scores use."""
    items = []
    for row in rows.to_dict("records"):
        by_tier = {tier: frozenset(str(label) for label in row[name]) for tier, name in EURLEX_LEVELS.items()}
        items.append(
            Item(
                str(row["celex_id"]),
                str(row[f"text_{language}"]),
                language,
                frozenset(str(label) for label in row["all_levels"]),
                str(row["celex_id"]),
                by_tier,
            )
        )
    return items


def sample_by_length(
    lengths: np.ndarray, long_threshold: int, normal_count: int, seed: int
) -> tuple[list[int], list[int]]:
    """Return (long, normal) item positions: every long item, and a repeatable sample of the rest."""
    long_items = np.flatnonzero(lengths >= long_threshold)
    others = np.flatnonzero(lengths < long_threshold)
    rng = np.random.default_rng(seed)
    picked = rng.choice(others, size=min(normal_count, len(others)), replace=False)
    return [int(i) for i in long_items], sorted(int(i) for i in picked)


def sample_by_language(languages: list[str], per_language: int, seed: int) -> list[int]:
    """Return a repeatable sample of item positions with up to `per_language` for each language."""
    rng = np.random.default_rng(seed)
    array = np.array(languages)
    picked: list[int] = []
    for language in sorted(set(languages)):
        positions = np.flatnonzero(array == language)
        picked.extend(
            int(i) for i in rng.choice(positions, size=min(per_language, len(positions)), replace=False)
        )
    return sorted(picked)
