"""Tests for fetching the pinned test sets: only missing files are downloaded, and every file is checked."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tiered_taxonomy import downloads as dl


def source(target: str, content: bytes, repo: str | None = "someone/data") -> dl.Source:
    return dl.Source(target, f"remote/{target}", hashlib.sha256(content).hexdigest(), repo, "abc")


def test_missing_files_are_downloaded_checked_and_present_ones_only_checked(tmp_path: Path) -> None:
    calls: list[str] = []

    def hub(found: dl.Source, target: Path) -> None:
        calls.append(found.target)
        target.write_bytes(b"hub file")

    def url(address: str, target: Path) -> None:
        calls.append(address)
        target.write_bytes(b"web file")

    sources = [source("a/one.parquet", b"hub file"), source("two.json", b"web file", repo=None)]
    assert dl.fetch(tmp_path, sources, hub, url) == ["a/one.parquet", "two.json"]
    assert calls == ["a/one.parquet", "remote/two.json"]
    assert (tmp_path / "a/one.parquet").read_bytes() == b"hub file"
    assert dl.fetch(tmp_path, sources, hub, url) == [] and len(calls) == 2


def test_a_wrong_file_raises_and_a_bad_download_is_never_put_in_place(tmp_path: Path) -> None:
    (tmp_path / "old.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="not the pinned"):
        dl.fetch(tmp_path, [source("old.json", b"original")], lambda *_: None, lambda *_: None)

    def corrupt(_found: dl.Source, target: Path) -> None:
        target.write_bytes(b"truncated")

    with pytest.raises(ValueError, match="not the pinned"):
        dl.fetch(tmp_path, [source("new.parquet", b"whole file")], corrupt, lambda *_: None)
    assert not (tmp_path / "new.parquet").exists()


def test_every_pinned_source_names_a_version_and_a_full_checksum() -> None:
    assert len({found.target for found in dl.SOURCES}) == len(dl.SOURCES)
    for found in dl.SOURCES:
        assert len(found.sha256) == 64
        assert (found.repo is None and found.filename.startswith("https://")) or len(
            found.revision or ""
        ) == 40
