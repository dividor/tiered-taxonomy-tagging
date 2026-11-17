"""Cite papers in the results report the way a paper does, "(Tamagnone et al., 2023)", from the sources in
`report_assets/methods.yaml`, and hold what a paper printed, for the HumSet tables to show beside ours."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CITE = re.compile(r"\{cite:([\w,]+)\}")
# The method fields shown as text, in the tables and on the cards, that may cite a source.
CITED_FIELDS = ("sub", "summary", "differs", "scorer", "clues_intro")


def author_year(ref: Mapping[str, Any]) -> str:
    """Return a source's authors and year as a citation names them: "Kluge and Kähler, 2025" for two
    authors, "Tamagnone et al., 2023" for more. A source with no year cannot be cited and raises."""
    if not ref.get("year"):
        raise ValueError(f"{ref['title']} has no year, so it cannot be cited")
    names = re.split(r", | and ", str(ref["authors"]))
    surnames = [name.split()[-1] for name in names if name != "others"]
    if len(names) > 2:
        who = f"{surnames[0]} et al."
    elif len(names) == 2:
        who = f"{surnames[0]} and {surnames[1]}"
    else:
        who = surnames[0]
    return f"{who}, {ref['year']}"


def cite(*refs: Mapping[str, Any]) -> str:
    """Return the bracketed citation of one or more sources: "(Wang et al., 2026; Xu et al., 2026)"."""
    if not refs:
        raise ValueError("a citation needs at least one source")
    return "(" + "; ".join(author_year(ref) for ref in refs) + ")"


def with_citations(text: str, refs: Mapping[str, Mapping[str, Any]]) -> str:
    """Replace each {cite:key} or {cite:key,key} in `text` with its bracketed citation. An unknown key
    raises."""

    def one(match: re.Match[str]) -> str:
        keys = match.group(1).split(",")
        unknown = [key for key in keys if key not in refs]
        if unknown:
            raise KeyError(f"no source called {unknown} to cite")
        return cite(*(refs[key] for key in keys))

    return CITE.sub(one, text)


def cited_methods(methods: Mapping[str, Any]) -> dict[str, Any]:
    """Return the method descriptions with every citation in their text fields written out."""
    refs = methods["references"]
    return {
        **methods,
        "methods": {
            key: {
                field: with_citations(value, refs) if field in CITED_FIELDS else value
                for field, value in entry.items()
            }
            for key, entry in methods["methods"].items()
        },
    }


@dataclass(frozen=True)
class Published:
    """What a paper printed by the macro F1 measure, for one HumSet table: the tooltip on the Macro F1
    heading, and, on the row that reruns the paper's recipe, a short note under its score with a tooltip
    saying what the paper reported and how our measurement differs. The `avg_` fields do the same for the
    sectors-and-pillars column, the mean of the three tier-1 groups that the paper prints as its Avg."""

    head_tip: str
    row_key: str
    note: str
    note_tip: str
    avg_tip: str
    avg_note: str
    avg_note_tip: str
