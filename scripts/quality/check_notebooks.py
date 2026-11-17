#!/usr/bin/env python3
"""Fail if a notebook still holds long console output, such as progress bars, which `scripts/trim_notebook.py`
removes after every run. Keeps committed notebooks small."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tiered_taxonomy.notebooks import long_streams


def main() -> int:
    """Check each notebook named on the command line and return 1 if any needs trimming."""
    untrimmed = [
        name for name in sys.argv[1:] if long_streams(json.loads(Path(name).read_text(encoding="utf-8")))
    ]
    for name in untrimmed:
        print(f"{name} holds long console output: run `uv run python scripts/trim_notebook.py {name}`")
    return 1 if untrimmed else 0


if __name__ == "__main__":
    sys.exit(main())
