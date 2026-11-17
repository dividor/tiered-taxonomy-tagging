"""Keep executed notebooks small enough to commit."""

from __future__ import annotations

from typing import Any

MAX_STREAM_CHARS = 2000


def too_long(output: dict[str, Any], limit: int = MAX_STREAM_CHARS) -> bool:
    """Return whether an output is console text over the limit, such as a progress bar."""
    text = output.get("text", "")
    size = len(text) if isinstance(text, str) else sum(len(line) for line in text)
    return output.get("output_type") == "stream" and size > limit


def long_streams(notebook: dict[str, Any], limit: int = MAX_STREAM_CHARS) -> int:
    """Return how many outputs are console text over the limit, without changing the notebook."""
    return sum(
        too_long(output, limit) for cell in notebook.get("cells", []) for output in cell.get("outputs", [])
    )


def trim_streams(notebook: dict[str, Any], limit: int = MAX_STREAM_CHARS) -> int:
    """Remove text printed to the console that is over the limit, such as progress bars. Returns how many."""
    removed = long_streams(notebook, limit)
    for cell in notebook.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = [output for output in cell["outputs"] if not too_long(output, limit)]
    return removed
