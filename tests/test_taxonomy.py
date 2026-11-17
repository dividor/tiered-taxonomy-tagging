"""Tests for the taxonomy tree, using small invented trees."""

from __future__ import annotations

import pytest

from tiered_taxonomy.eurovoc import build_taxonomy, concept_id, unreachable
from tiered_taxonomy.taxonomy import Node, Taxonomy, from_paths, with_definitions, without_implied_parents


def small_tree() -> Taxonomy:
    return Taxonomy(
        {
            "a": Node("a", "Agri"),
            "b": Node("b", "Trade"),
            "a1": Node("a1", "Plant", ("a",)),
            "a2": Node("a2", "Animal", ("a",)),
            "fruit": Node("fruit", "Fruit", ("a1",)),
            "export": Node("export", "Export", ("a2", "b")),
        }
    )


def test_a_parent_given_with_its_own_child_is_dropped_and_one_given_alone_is_kept() -> None:
    given = {"Covid-19", "Covid-19->Testing", "Displacement", "Health"}
    assert without_implied_parents(given) == {"Covid-19->Testing", "Displacement", "Health"}
    assert without_implied_parents(["100149", "100160"]) == {"100149", "100160"}
    assert without_implied_parents([]) == set()


def test_tiers_are_the_shortest_distance_from_a_root() -> None:
    tree = small_tree()
    assert tree.roots == ["a", "b"]
    assert tree.tiers["a"] == 1
    assert tree.tiers["a1"] == 2
    assert tree.tiers["fruit"] == 3
    assert tree.tiers["export"] == 2
    assert tree.at_tier(1) == ["a", "b"]


def test_roll_up_follows_every_parent_and_keeps_labels_at_the_tier() -> None:
    tree = small_tree()
    assert tree.roll_up(["fruit"], 1) == {"a"}
    assert tree.roll_up(["export"], 1) == {"a", "b"}
    assert tree.roll_up(["a1"], 2) == {"a1"}
    assert tree.roll_up(["a"], 2) == set()


def test_ancestors_descendants_and_leaves() -> None:
    tree = small_tree()
    assert tree.ancestors("fruit") == {"a1", "a"}
    assert tree.descendants("a") == {"a1", "a2", "fruit", "export"}
    assert tree.leaves_under("a") == {"fruit", "export"}
    assert tree.leaves_under("fruit") == {"fruit"}


def test_path_reads_from_the_root() -> None:
    assert small_tree().path("fruit") == "Agri > Plant > Fruit"


def test_unknown_parent_raises() -> None:
    with pytest.raises(ValueError, match="unknown parent"):
        Taxonomy({"x": Node("x", "X", ("missing",))})


def test_a_cycle_with_no_root_raises() -> None:
    with pytest.raises(ValueError, match="cannot be reached"):
        Taxonomy({"r": Node("r", "R"), "x": Node("x", "X", ("y",)), "y": Node("y", "Y", ("x",))})


def test_from_paths_builds_parents_and_flat_roots() -> None:
    tree = from_paths(["Context->Economy", "Context->Politics", "Health"])
    assert tree.roots == ["Context", "Health"]
    assert tree.children["Context"] == ["Context->Economy", "Context->Politics"]
    assert tree.nodes["Context->Economy"].name == "Economy"
    assert tree.tiers["Context->Politics"] == 2


def test_with_definitions_keeps_structure() -> None:
    tree = with_definitions(small_tree(), {"fruit": "Edible fruit."})
    assert tree.nodes["fruit"].definition == "Edible fruit."
    assert tree.roll_up(["fruit"], 1) == {"a"}


def test_concept_id_takes_the_end_of_an_address() -> None:
    assert concept_id("http://eurovoc.europa.eu/1115") == "1115"


def test_build_taxonomy_drops_concepts_the_dataset_does_not_name() -> None:
    names = {key: {"en": key.upper()} for key in ("d", "m", "c")}
    edges = [("d", "m"), ("m", "c"), ("m", "unknown"), ("unknown", "x"), ("nobody", "alone")]
    tree = build_taxonomy(edges, names, ["d"])
    assert set(tree.nodes) == {"d", "m", "c"}
    assert tree.nodes["c"].name == "C"
    assert tree.tiers["c"] == 3
    assert unreachable(tree, ["c", "alone"]) == {"alone"}
