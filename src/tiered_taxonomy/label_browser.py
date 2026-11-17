"""What the report's label browser and example records show: each taxonomy as a tree, and a few test records
with the labels people gave them.

The trees hold label names, tiers and how often each label was used in training. The examples hold a few
records' text, cut short for the laws, by the user's choice: they are the one place document text enters the
repository, so that a reader can see what is being labelled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tiered_taxonomy import datasets as ds
from tiered_taxonomy.taxonomy import Taxonomy

EURLEX_LINK = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{id}"


def label_tree(taxonomy: Taxonomy, train_counts: Mapping[str, int]) -> dict[str, Any]:
    """Return a taxonomy as roots and nodes, each node as [name, tier, children, times used in training]."""
    tiers: dict[str, int] = {}
    for tier in taxonomy.tiers.values():
        tiers[str(tier)] = tiers.get(str(tier), 0) + 1
    return {
        "roots": sorted(taxonomy.roots, key=lambda node: taxonomy.nodes[node].name.lower()),
        "tiers": dict(sorted(tiers.items(), key=lambda pair: int(pair[0]))),
        "nodes": {
            node_id: [
                node.name,
                taxonomy.tiers[node_id],
                sorted(taxonomy.children.get(node_id, []), key=lambda kid: taxonomy.nodes[kid].name.lower()),
                int(train_counts.get(node_id, 0)),
            ]
            for node_id, node in taxonomy.nodes.items()
        },
    }


def excerpt(text: str, words: int) -> tuple[str, bool]:
    """Return a text's first `words` words, keeping its line breaks, and whether it was cut."""
    kept: list[str] = []
    count = 0
    for line in text.splitlines():
        line_words = line.split()
        if not line_words:
            continue
        room = words - count
        kept.append(" ".join(line_words[:room]))
        count += min(len(line_words), room)
        if count >= words:
            return "\n".join(kept), count < len(text.split())
    return "\n".join(kept), False


def example_record(item: ds.Item, words: int, link: str = "") -> dict[str, Any]:
    """Return what the report shows for one test record: its text, cut to `words` words, and its labels."""
    text, cut = excerpt(item.text, words)
    return {
        "id": item.id,
        "language": item.language,
        "words": ds.word_count(item.text),
        "text": text,
        "cut": cut,
        "labels": sorted(item.labels),
        "link": link.format(id=item.id) if link else "",
    }


def pick_examples(items: Sequence[ds.Item], count: int, labels: tuple[int, int]) -> list[ds.Item]:
    """Return the first `count` items with a typical number of labels, one per language while they last."""
    fewest, most = labels
    fitting = [item for item in items if fewest <= len(item.labels) <= most]
    chosen: list[ds.Item] = []
    for item in fitting:
        if item.language not in {c.language for c in chosen}:
            chosen.append(item)
    chosen += [item for item in fitting if item not in chosen]
    if len(chosen) < count:
        raise ValueError(f"only {len(chosen)} items have {fewest} to {most} labels; {count} were asked for")
    return chosen[:count]
