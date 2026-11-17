"""Tests for loading the test sets, using small invented records."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tiered_taxonomy.datasets import (
    chunk_words,
    eurlex_items,
    eurlex_rows,
    first_words,
    humset_items,
    humset_taxonomy,
    load_humset_rows,
    sample_by_language,
    sample_by_length,
    stream_eurlex,
    word_count,
)

RECORD = {
    "celex_id": "1",
    "publication_date": "2013-01-01",
    "text": {"en": "english text", "fr": "texte", "de": "text"},
    "eurovoc_concepts": {"level_1": ["d"], "level_2": ["m"], "level_3": ["t"], "all_levels": ["t", "c"]},
}


def humset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_id": ["1"],
            "lead_id": ["L"],
            "project_id": ["P"],
            "lang": ["fr"],
            "excerpt": ["texte"],
            "sectors": [["Health"]],
            "subpillars_1d": [["Context->Economy"]],
            "subpillars_2d": [[]],
            "pillars_1d": [["Context"]],
            "pillars_2d": [[]],
        }
    )


def test_text_helpers() -> None:
    assert word_count("a b  c") == 3
    assert first_words("a b c", 2) == "a b"
    assert chunk_words("a b c", 2) == ["a b", "c"]
    assert chunk_words("", 2) == []


def test_humset_items_have_gold_by_tier_and_a_document_group() -> None:
    rows = humset_frame()
    tree = humset_taxonomy(rows)
    item = humset_items(rows, tree)[0]
    assert item.labels == {"Health", "Context->Economy"}
    assert item.group == "L"
    assert item.project == "P"
    assert item.labels_by_tier[1] == {"Health", "Context"}
    assert item.labels_by_tier[2] == {"Context->Economy"}


def test_a_pillar_tagged_without_its_subpillars_is_a_label_of_its_own() -> None:
    rows = humset_frame().assign(pillars_1d=[["Context", "Covid-19"]], pillars_2d=[["Impact"]])
    tree = humset_taxonomy(pd.concat([rows, humset_frame().assign(subpillars_1d=[["Covid-19->Testing"]])]))
    item = humset_items(rows, tree)[0]
    assert item.labels == {"Health", "Context->Economy", "Covid-19", "Impact"}
    assert item.labels_by_tier[1] == {"Health", "Context", "Covid-19", "Impact"}
    assert item.labels_by_tier[2] == {"Context->Economy"}
    assert tree.tiers["Impact"] == 1


def test_eurlex_rows_keep_only_named_languages_and_levels() -> None:
    rows = eurlex_rows([RECORD], ["en", "es"])
    assert rows.loc[0, "text_en"] == "english text"
    assert rows.loc[0, "text_es"] == ""
    assert "text_de" not in rows.columns
    item = eurlex_items(rows)[0]
    assert item.labels == {"t", "c"}
    assert item.labels_by_tier[2] == {"m"}


def test_stream_eurlex_reads_a_split_from_the_archive(tmp_path: Path) -> None:
    path = tmp_path / "a.tar.gz"
    body = (json.dumps(RECORD) + "\n").encode()
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("dev.jsonl")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    assert next(stream_eurlex(path, "dev"))["celex_id"] == "1"
    with pytest.raises(FileNotFoundError):
        list(stream_eurlex(path, "test"))


def test_sample_by_length_keeps_all_long_items_and_repeats() -> None:
    lengths = np.array([10, 9000, 20, 30, 40])
    long_items, normal = sample_by_length(lengths, 5000, 2, seed=1)
    assert long_items == [1]
    assert len(normal) == 2
    assert 1 not in normal
    assert normal == sample_by_length(lengths, 5000, 2, seed=1)[1]


def test_sample_by_language_balances_and_repeats() -> None:
    languages = ["en"] * 5 + ["fr"] * 2
    picked = sample_by_language(languages, 2, seed=1)
    assert sum(languages[i] == "en" for i in picked) == 2
    assert sum(languages[i] == "fr" for i in picked) == 2
    assert picked == sample_by_language(languages, 2, seed=1)


def test_humset_rows_use_the_project_title_only_where_a_version_has_no_project_id(tmp_path: Path) -> None:
    for version in ("1.0.0", "2.0.0"):
        (tmp_path / "humset" / version).mkdir(parents=True)
    humset_frame().to_parquet(tmp_path / "humset" / "1.0.0" / "test.parquet")
    untitled = humset_frame().drop(columns="project_id").assign(project_title=["Project A"])
    untitled.to_parquet(tmp_path / "humset" / "2.0.0" / "test.parquet")
    assert load_humset_rows(tmp_path, "test")["project_id"].tolist() == ["P"]
    assert load_humset_rows(tmp_path, "test", "2.0.0")["project_id"].tolist() == ["Project A"]
    untitled.to_parquet(tmp_path / "humset" / "1.0.0" / "test.parquet")
    with pytest.raises(ValueError, match="project_id"):
        load_humset_rows(tmp_path, "test")
