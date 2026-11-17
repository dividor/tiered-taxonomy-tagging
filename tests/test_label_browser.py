"""Tests for the report's label trees and example records, on a small made-up taxonomy."""

from __future__ import annotations

from collections import Counter

import pytest

from tiered_taxonomy import label_browser as lb
from tiered_taxonomy.datasets import Item
from tiered_taxonomy.taxonomy import Node, Taxonomy

TAXONOMY = Taxonomy(
    {
        "t": Node("t", "trade"),
        "a": Node("a", "agriculture"),
        "m": Node("m", "marketing", ("t",)),
        "l": Node("l", "labelling", ("m",)),
        "p": Node("p", "Packaging", ("m",)),
    }
)


def item(item_id: str, labels: set[str], language: str = "en", text: str = "one two three") -> Item:
    return Item(item_id, text, language, frozenset(labels))


def test_label_tree_lists_tiers_children_by_name_and_training_counts() -> None:
    tree = lb.label_tree(TAXONOMY, Counter({"l": 4}))
    assert tree["roots"] == ["a", "t"]
    assert tree["tiers"] == {"1": 2, "2": 1, "3": 2}
    assert tree["nodes"]["m"] == ["marketing", 2, ["l", "p"], 0]
    assert tree["nodes"]["l"] == ["labelling", 3, [], 4]


def test_excerpt_keeps_line_breaks_and_says_when_it_cut() -> None:
    assert lb.excerpt("A B\n\nC D E\nF", 4) == ("A B\nC D", True)
    assert lb.excerpt("A B\nC", 5) == ("A B\nC", False)
    assert lb.excerpt("A B C D", 4) == ("A B C D", False)


def test_example_record_links_the_record_and_sorts_its_labels() -> None:
    record = lb.example_record(
        item("32014R0653", {"p", "l"}, text="REGULATION\nof 15 May 2014 on cattle"), 3, lb.EURLEX_LINK
    )
    assert record["labels"] == ["l", "p"]
    assert record["text"] == "REGULATION\nof 15" and record["cut"]
    assert record["words"] == 7
    assert record["link"].endswith("CELEX:32014R0653")
    assert lb.example_record(item("9", {"l"}), 10)["link"] == ""


def test_pick_examples_prefers_one_per_language_then_raises_when_short() -> None:
    items = [
        item("1", {"l"}),
        item("2", {"l", "p"}),
        item("3", {"l", "p"}, "fr"),
        item("4", {"l", "p"}),
        item("5", {"l", "p"}, "es"),
    ]
    assert [i.id for i in lb.pick_examples(items, 3, (2, 3))] == ["2", "3", "5"]
    assert [i.id for i in lb.pick_examples(items, 4, (2, 3))] == ["2", "3", "5", "4"]
    with pytest.raises(ValueError, match="only 4 items"):
        lb.pick_examples(items, 5, (2, 3))
