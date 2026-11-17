#!/usr/bin/env python3
"""Remove long console output, such as progress bars, from an executed notebook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tiered_taxonomy.notebooks import trim_streams


def main() -> int:
    """Trim each notebook named on the command line."""
    for name in sys.argv[1:]:
        path = Path(name)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        removed = trim_streams(notebook)
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{name}: removed {removed} long outputs, now {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
