"""Tests for trimming notebook outputs."""

from __future__ import annotations

from tiered_taxonomy.notebooks import long_streams, trim_streams


def test_trim_removes_only_long_console_output() -> None:
    notebook = {
        "cells": [
            {
                "outputs": [
                    {"output_type": "stream", "text": ["x" * 3000]},
                    {"output_type": "stream", "text": "short"},
                    {"output_type": "execute_result", "text": "y" * 3000},
                ]
            },
            {"source": "a markdown cell has no outputs"},
        ]
    }
    assert trim_streams(notebook) == 1
    assert len(notebook["cells"][0]["outputs"]) == 2


def test_long_streams_counts_without_changing_the_notebook() -> None:
    notebook = {"cells": [{"outputs": [{"output_type": "stream", "text": "x" * 3000}]}, {"source": "text"}]}
    assert long_streams(notebook) == 1 and long_streams(notebook, limit=5000) == 0
    assert len(notebook["cells"][0]["outputs"]) == 1
    trim_streams(notebook)
    assert long_streams(notebook) == 0
