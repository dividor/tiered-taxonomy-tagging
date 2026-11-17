"""A tiered taxonomy: named nodes with parents, where a node may have more than one parent (EuroVoc does)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Node:
    """One taxonomy concept. `definition` holds a label card's text when one has been generated."""

    id: str
    name: str
    parents: tuple[str, ...] = ()
    definition: str = ""


@dataclass
class Taxonomy:
    """Nodes with their parents. The tier of a node is its shortest distance from a root, and roots are tier
    1."""

    nodes: dict[str, Node]
    children: dict[str, list[str]] = field(init=False)
    roots: list[str] = field(init=False)
    tiers: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        children: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes.values():
            for parent in node.parents:
                if parent not in self.nodes:
                    raise ValueError(f"node {node.id} has unknown parent {parent}")
                children[parent].append(node.id)
        self.children = {key: sorted(value) for key, value in children.items()}
        self.roots = sorted(node.id for node in self.nodes.values() if not node.parents)
        self.tiers = self._compute_tiers()

    def _compute_tiers(self) -> dict[str, int]:
        tiers = dict.fromkeys(self.roots, 1)
        frontier = list(self.roots)
        while frontier:
            following: list[str] = []
            for node_id in frontier:
                for child in self.children.get(node_id, []):
                    if child not in tiers:
                        tiers[child] = tiers[node_id] + 1
                        following.append(child)
            frontier = following
        unreachable = set(self.nodes) - set(tiers)
        if unreachable:
            raise ValueError(
                f"{len(unreachable)} nodes cannot be reached from a root, for example {min(unreachable)}"
            )
        return tiers

    def ancestors(self, node_id: str) -> set[str]:
        """Return every node above this one, through all of its parents."""
        found: set[str] = set()
        pending = list(self.nodes[node_id].parents)
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(self.nodes[current].parents)
        return found

    def roll_up(self, labels: Iterable[str], tier: int) -> set[str]:
        """Map labels to their ancestors at a tier. A label already at that tier is kept; shallower ones are
        not."""
        rolled: set[str] = set()
        for label in labels:
            if self.tiers[label] == tier:
                rolled.add(label)
            rolled.update(a for a in self.ancestors(label) if self.tiers[a] == tier)
        return rolled

    def at_tier(self, tier: int) -> list[str]:
        """Return the node ids at a tier."""
        return sorted(node_id for node_id, found in self.tiers.items() if found == tier)

    def leaves_under(self, node_id: str) -> set[str]:
        """Return the nodes without children at or below a node."""
        kids = self.children.get(node_id, [])
        if not kids:
            return {node_id}
        return set().union(*(self.leaves_under(kid) for kid in kids))

    def descendants(self, node_id: str) -> set[str]:
        """Return every node below this one."""
        found: set[str] = set()
        pending = list(self.children.get(node_id, []))
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(self.children.get(current, []))
        return found

    def path(self, node_id: str) -> str:
        """Return one readable path from a root, following the first listed parent at each step."""
        names = [self.nodes[node_id].name]
        current = node_id
        while self.nodes[current].parents:
            current = self.nodes[current].parents[0]
            names.append(self.nodes[current].name)
        return " > ".join(reversed(names))


def without_implied_parents(labels: Iterable[str], separator: str = "->") -> set[str]:
    """Return labels without any whose own child, written `Parent->Child`, is also given. A child names its
    parent, so a HumSet pillar given with one of its subpillars adds nothing, while a pillar given alone is
    kept, as HumSet's gold keeps it. Labels with no separator, such as EuroVoc's, are never dropped."""
    given = set(labels)
    return given - {label.split(separator)[0] for label in given if separator in label}


def from_paths(paths: Iterable[str], separator: str = "->") -> Taxonomy:
    """Build a taxonomy from labels written as `Parent->Child`. A label with no separator becomes a root."""
    nodes: dict[str, Node] = {}
    for path in paths:
        parts = [part.strip() for part in path.split(separator)]
        parent: tuple[str, ...] = ()
        built = ""
        for part in parts:
            built = f"{built}{separator}{part}" if built else part
            nodes.setdefault(built, Node(built, part, parent))
            parent = (built,)
    return Taxonomy(nodes)


def with_definitions(taxonomy: Taxonomy, definitions: Mapping[str, str]) -> Taxonomy:
    """Return a copy of a taxonomy whose nodes carry label-card definitions."""
    return Taxonomy(
        {
            node_id: Node(node.id, node.name, node.parents, definitions.get(node_id, node.definition))
            for node_id, node in taxonomy.nodes.items()
        }
    )
