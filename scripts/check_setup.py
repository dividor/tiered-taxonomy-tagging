#!/usr/bin/env python3
"""Check what a run needs before it starts: the data folder, room in it, the Hugging Face login and the jobs
dataset setting. Prints nothing secret. Exits 1 if anything is missing."""

from __future__ import annotations

import shutil
import sys

from tiered_taxonomy import gpu_jobs as gj
from tiered_taxonomy.config import data_root
from tiered_taxonomy.models import hf_token

# A full run leaves about 12 GB in the data folder; keep some room beyond that.
NEEDED_GB = 15


def main() -> int:
    """Print each check and return 1 if any failed."""
    problems = []
    root = data_root()
    print(f"data folder: {root}")
    free = shutil.disk_usage(root).free / 1e9
    if free < NEEDED_GB:
        problems.append(f"only {free:.0f} GB free in the data folder; a full run needs about {NEEDED_GB} GB")
    hf_token()
    print("Hugging Face token found")
    try:
        print("GPU jobs upload to:", gj.jobs_dataset())
    except RuntimeError as error:
        problems.append(str(error))
    for problem in problems:
        print("problem:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
