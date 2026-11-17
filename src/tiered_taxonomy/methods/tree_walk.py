"""Walking a taxonomy from the top, with the recovery moves from recent papers.

Three settings are compared. `strict` follows one branch, as plain top-down routing does. `branching` keeps
several
branches open (Graph-Constrained Policy, arXiv 2607.11954). `anchored` also starts from retrieved entry
points rather
than the root (Constraint-Aware Search, arXiv 2607.10588) and shows likely labels under each option (ToT-ICD).
Backtracking with tools, the fourth published move, is not implemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tiered_taxonomy.llm import Client, Reply
from tiered_taxonomy.methods.base import Prediction
from tiered_taxonomy.prompts import WALK_SYSTEM, walk_message
from tiered_taxonomy.taxonomy import Taxonomy


@dataclass(frozen=True)
class WalkSettings:
    """How many branches a walk keeps open at each level, and how many model calls it may make for one
    text."""

    max_branches: int = 3
    max_calls: int = 15
    look_ahead_leaves: int = 0
    entry_tier: int = 2


def parse_walk(reply: Reply, options: Sequence[str]) -> tuple[list[str], bool, int]:
    """Read a walk reply as (chosen options, whether to stop here, invalid ids). Malformed replies choose
    nothing."""
    if reply.parsed is None:
        return [], False, 0
    selected = reply.parsed.get("selected")
    if not isinstance(selected, list):
        return [], False, 0
    allowed = set(options)
    chosen = [str(option) for option in selected if str(option) in allowed]
    return chosen, reply.parsed.get("stop_here") is True, len(selected) - len(chosen)


def strongest_under(
    node_id: str, taxonomy: Taxonomy, retrieved: Mapping[str, float], count: int
) -> list[str]:
    """Return the retrieved labels below a node, strongest first."""
    below = taxonomy.descendants(node_id)
    found = [label for label in retrieved if label in below]
    return sorted(found, key=lambda label: (-retrieved[label], label))[:count]


def entry_points(taxonomy: Taxonomy, retrieved: Mapping[str, float], tier: int, count: int) -> list[str]:
    """Return the nodes at a tier with the most retrieved weight beneath them, to start a walk from."""
    totals: dict[str, float] = {}
    for label, value in retrieved.items():
        for ancestor in taxonomy.roll_up([label], tier):
            totals[ancestor] = totals.get(ancestor, 0.0) + value
    return sorted(totals, key=lambda key: (-totals[key], key))[:count]


def _ask(
    text: str,
    taxonomy: Taxonomy,
    client: Client,
    current: str | None,
    options: Sequence[str],
    look_ahead: Mapping[str, list[str]] | None,
    prediction: Prediction,
) -> tuple[list[str], bool]:
    heading = None if current is None else taxonomy.path(current)
    reply = client.ask(WALK_SYSTEM, walk_message(text, taxonomy, heading, options, look_ahead))
    prediction.calls += 1
    prediction.malformed += int(reply.parsed is None)
    chosen, stop_here, invalid = parse_walk(reply, options)
    prediction.invalid += invalid
    return chosen, stop_here


def walk(
    text: str,
    taxonomy: Taxonomy,
    client: Client,
    settings: WalkSettings,
    starts: Sequence[str] | None = None,
    retrieved: Mapping[str, float] | None = None,
) -> Prediction:
    """Walk down level by level, keeping at most `max_branches` nodes open, and return where the walk ended.

    With `starts`, the first step is a choice among those entry points rather than among the roots. A chosen
    node becomes a label when it has no children, when nothing below it fits, when the model says to stop
    there, or when the call budget runs out on reaching it. Strict walking uses `max_branches` of 1.
    """
    prediction = Prediction()
    if starts:
        look_ahead = _look_ahead(starts, taxonomy, retrieved, settings.look_ahead_leaves)
        chosen, _ = _ask(text, taxonomy, client, None, starts, look_ahead, prediction)
        frontier: list[str | None] = list(chosen)
    else:
        frontier = [None]
    seen: set[str | None] = set()
    while frontier:
        following: list[str] = []
        for current in frontier[: settings.max_branches]:
            if current not in seen:
                seen.add(current)
                following.extend(_visit(current, text, taxonomy, client, settings, retrieved, prediction))
        frontier = list(dict.fromkeys(following))
    return prediction


def _visit(
    current: str | None,
    text: str,
    taxonomy: Taxonomy,
    client: Client,
    settings: WalkSettings,
    retrieved: Mapping[str, float] | None,
    prediction: Prediction,
) -> list[str]:
    """Open one node: record it as a label where the walk ends there, and return the nodes chosen below it."""
    options = taxonomy.roots if current is None else taxonomy.children.get(current, [])
    if not options or prediction.calls >= settings.max_calls:
        if current is not None:
            prediction.labels[current] = 1.0
        return []
    look_ahead = _look_ahead(options, taxonomy, retrieved, settings.look_ahead_leaves)
    chosen, stop_here = _ask(text, taxonomy, client, current, options, look_ahead, prediction)
    if current is not None and (stop_here or not chosen):
        prediction.labels[current] = 1.0
    return list(chosen)


def _look_ahead(
    options: Sequence[str], taxonomy: Taxonomy, retrieved: Mapping[str, float] | None, count: int
) -> dict[str, list[str]] | None:
    if not retrieved or count <= 0:
        return None
    return {option: strongest_under(option, taxonomy, retrieved, count) for option in options}
