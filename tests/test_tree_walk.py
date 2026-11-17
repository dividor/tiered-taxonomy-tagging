"""Tests for walking the tree. Nothing is sent or paid for."""

from __future__ import annotations

from typing import Any

from method_fakes import Scripted, reply, tree
from tiered_taxonomy.methods.tree_walk import WalkSettings, entry_points, parse_walk, strongest_under, walk
from tiered_taxonomy.prompts import walk_message
from tiered_taxonomy.taxonomy import from_paths


def test_parse_walk_reads_choices_and_counts_invalid_ids() -> None:
    assert parse_walk(reply('{"selected": ["a", "z"], "stop_here": true}'), ["a", "b"]) == (["a"], True, 1)
    assert parse_walk(reply("nope"), ["a"]) == ([], False, 0)
    assert parse_walk(reply('{"selected": "a"}'), ["a"]) == ([], False, 0)


def test_strict_walk_follows_one_branch_to_a_leaf() -> None:
    client: Any = Scripted('{"selected": ["Health", "Context"]}', '{"selected": ["Health->Water"]}')
    result = walk("water", tree(), client, WalkSettings(max_branches=1))
    assert set(result.labels) == {"Health->Water"}
    assert result.calls == 2


def test_branching_walk_reaches_several_branches() -> None:
    client: Any = Scripted(
        '{"selected": ["Context", "Health"]}',
        '{"selected": ["Context->Economy"]}',
        '{"selected": ["Health->Water"]}',
    )
    result = walk("text", tree(), client, WalkSettings(max_branches=2))
    assert set(result.labels) == {"Context->Economy", "Health->Water"}


def test_walk_can_stop_on_an_inner_node() -> None:
    client: Any = Scripted('{"selected": ["Health"]}', '{"selected": [], "stop_here": true}')
    assert set(walk("text", tree(), client, WalkSettings()).labels) == {"Health"}


def test_walk_labels_a_chosen_node_when_nothing_below_it_fits() -> None:
    client: Any = Scripted('{"selected": ["Health"]}', '{"selected": []}')
    assert set(walk("text", tree(), client, WalkSettings()).labels) == {"Health"}


def test_walk_labels_what_it_reached_when_the_budget_runs_out() -> None:
    client: Any = Scripted('{"selected": ["Context", "Health"]}')
    result = walk("text", tree(), client, WalkSettings(max_branches=2, max_calls=1))
    assert result.calls == 1
    assert set(result.labels) == {"Context", "Health"}


def test_walk_keeps_only_so_many_branches_open_at_a_level() -> None:
    client: Any = Scripted('{"selected": ["Context", "Health"]}', '{"selected": ["Context->Economy"]}')
    result = walk("text", tree(), client, WalkSettings(max_branches=1))
    assert set(result.labels) == {"Context->Economy"}
    assert result.calls == 2


def test_walk_from_entry_points_asks_the_model_to_choose_among_them_first() -> None:
    client: Any = Scripted('{"selected": ["Health"]}', '{"selected": ["Health->Water"]}')
    result = walk("text", tree(), client, WalkSettings(), starts=["Health", "Context"])
    assert set(result.labels) == {"Health->Water"}
    assert result.calls == 2
    assert "Context" in client.asked[0]


def test_entry_points_the_model_rejects_are_never_labelled() -> None:
    flat = from_paths(["Health", "Water"])
    client: Any = Scripted('{"selected": []}')
    assert walk("text", flat, client, WalkSettings(), starts=["Health", "Water"]).labels == {}


def test_look_ahead_shows_likely_labels_under_each_option() -> None:
    retrieved = {"Health->Water": 0.9, "Context->Economy": 0.1}
    assert strongest_under("Health", tree(), retrieved, 1) == ["Health->Water"]
    assert entry_points(tree(), retrieved, 1, 1) == ["Health"]
    message = walk_message("t", tree(), None, ["Health"], {"Health": ["Health->Water"]})
    assert "such as: Water" in message
    client: Any = Scripted('{"selected": []}')
    walk("t", tree(), client, WalkSettings(look_ahead_leaves=1), retrieved=retrieved)
    assert "such as: Water" in client.asked[0]
