#!/usr/bin/env python3
"""Build the results report page, comparing every method, from the saved results, and the README's results
section from the same rows.

    uv run python scripts/make_report.py                  # writes outputs/report/index.html and README.md
    uv run python scripts/make_report.py <output.html>    # the page there; the README is still updated

It reads only `outputs/results/*.json`, `configs/experiment.yaml` and the method descriptions in
`src/tiered_taxonomy/report_assets/methods.yaml`, so it needs no data folder. It stops if a finding
the page states no longer holds for the saved results.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tiered_taxonomy.report import load_inputs
from tiered_taxonomy.report_markdown import results_section, with_results
from tiered_taxonomy.report_page import build_page

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    """Write the report page to the path given, or to outputs/report/index.html."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "report" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(REPO), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    readme = REPO / "README.md"
    section = results_section(load_inputs(REPO))
    readme.write_text(with_results(readme.read_text(encoding="utf-8"), section), encoding="utf-8")
    print(f"wrote the results section of {readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
