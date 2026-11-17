"""The test sets, each pinned to one version and checked against its SHA-256, so every run starts from the
same bytes. `scripts/download_data.py` fetches whatever is missing into the data folder.

EuroVoc's links between labels are not here: notebook 01 downloads them from the EU Publications Office's live
SPARQL endpoint, which serves today's EuroVoc, so a run at a later date may see a slightly different tree.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

HUMSET = "nlp-thedeep/humset"
HUMSET_REVISION = "0a9719d8cd1387d03373347edb6f53b9ea715299"
MULTI_EURLEX = "coastalcph/multi_eurlex"
MULTI_EURLEX_REVISION = "d0318b7f68f43f7d77797437ce172d899254cd0f"
DESCRIPTORS_URL = (
    "https://raw.githubusercontent.com/nlpaueb/multi-eurlex/af2da31b54ca4272d880a46aee8dbbf346712e90/"
    "data/eurovoc_descriptors.json"
)


@dataclass(frozen=True)
class Source:
    """One file of the test sets: where it goes in the data folder, where it comes from, and its checksum.
    `repo` is a Hugging Face dataset; without one, `filename` is a URL."""

    target: str
    filename: str
    sha256: str
    repo: str | None = None
    revision: str | None = None


def _humset(version: str, split: str, sha256: str) -> Source:
    return Source(
        f"humset/{version}/{split}.parquet",
        f"{version}/{split}/0000.parquet",
        sha256,
        HUMSET,
        HUMSET_REVISION,
    )


SOURCES = (
    _humset("1.0.0", "train", "63dedaa165dfe8b8a18132046bb0efcaf4533efcc40e08cab8969ce03a32101d"),
    _humset("1.0.0", "validation", "8eb7aadfb0b23fce407249650545022c9ebad40282d48e4ea00d55261aaa5e7d"),
    _humset("1.0.0", "test", "03018183f6c6fdd791fdb456e2aae87247cc09a1de53b6c4cba5e975fbfb6d2d"),
    _humset("2.0.0", "train", "ef7ab01aa99a8edaa9e8edc6773280b1a00b10da5c54440fd5ab07aabdc63c14"),
    _humset("2.0.0", "validation", "5366b092a53c8b6012b5bc17c93f5e46420f00be8b645ec34e157432abce679a"),
    _humset("2.0.0", "test", "f44a145ad7b4a43d53433ecff9385c516bbc4a72c3deef483b1e50d5c48c4d6d"),
    Source(
        "multi_eurlex/multi_eurlex.tar.gz",
        "data/multi_eurlex.tar.gz",
        "3a2195bc01c7aea0e302bc7af94f2f55f0c15cdeabfe4e086c083196069a2e83",
        MULTI_EURLEX,
        MULTI_EURLEX_REVISION,
    ),
    Source(
        "multi_eurlex/eurovoc_descriptors.json",
        DESCRIPTORS_URL,
        "5d42e52c09bc8eb2166edf40f19097eed7e5040e9f132818d1b09ff62f492804",
    ),
)


def sha256(path: Path) -> str:
    """Return a file's SHA-256, read in blocks so large files need little memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path: Path, expected: str) -> None:
    """Raise if a file is not the pinned version."""
    found = sha256(path)
    if found != expected:
        raise ValueError(
            f"{path} has SHA-256 {found}, not the pinned {expected}: delete it and download again"
        )


def download_url(url: str, target: Path) -> None:
    """Save a URL to a file, raising on any HTTP error."""
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)


def download_hub(source: Source, target: Path) -> None:
    """Save one pinned file from a Hugging Face dataset."""
    if source.repo is None:
        raise ValueError(f"{source.target} has no Hugging Face dataset")
    cached = hf_hub_download(source.repo, source.filename, repo_type="dataset", revision=source.revision)
    shutil.copyfile(cached, target)


def fetch(
    root: Path,
    sources: Iterable[Source] = SOURCES,
    hub: Callable[[Source, Path], None] = download_hub,
    url: Callable[[str, Path], None] = download_url,
) -> list[str]:
    """Download every missing file into the data folder and check every file, old or new. Returns the files
    downloaded. A file that is present but not the pinned version raises rather than being replaced."""
    fetched = []
    for source in sources:
        target = root / source.target
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            if source.repo is None:
                url(source.filename, partial)
            else:
                hub(source, partial)
            check(partial, source.sha256)
            partial.rename(target)
            fetched.append(source.target)
        else:
            check(target, source.sha256)
    return fetched
