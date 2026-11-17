#!/usr/bin/env python3
"""Download the test sets into the data folder, each pinned to one version and checked by its SHA-256.

    uv run python scripts/download_data.py

About 2.9 GB, almost all of it MultiEURLEX. Files already present are checked, not downloaded again.
"""

from __future__ import annotations

from tiered_taxonomy.config import data_root
from tiered_taxonomy.downloads import SOURCES, fetch


def main() -> int:
    root = data_root()
    fetched = fetch(root)
    for name in fetched:
        print("downloaded", root / name)
    print(f"all {len(SOURCES)} files present and checked in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
