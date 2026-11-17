"""The EuroVoc tree, from the EU Publications Office's SPARQL endpoint, with the dataset's own label names."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import requests

from tiered_taxonomy.taxonomy import Node, Taxonomy

ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
SCHEME = "http://eurovoc.europa.eu/100141"
_PREFIX = "PREFIX skos:<http://www.w3.org/2004/02/skos/core#> "
# Concept to its broader concept, within the thesaurus.
BROADER_QUERY = (
    _PREFIX + f"SELECT ?child ?parent WHERE {{ ?child skos:inScheme <{SCHEME}> ; skos:broader ?parent . }}"
)
# Microthesaurus to its top concepts.
TOP_QUERY = _PREFIX + (
    "SELECT ?parent ?child WHERE { ?parent skos:hasTopConcept ?child . "
    f"?child skos:inScheme <{SCHEME}> . FILTER(?parent != <{SCHEME}>) }}"
)
# Domain to its microthesauri.
PART_QUERY = (
    "PREFIX ev:<http://eurovoc.europa.eu/schema#> "
    "SELECT ?parent ?child WHERE { ?child a ev:MicroThesaurus ; ev:domain ?parent . }"
)


def concept_id(uri: str) -> str:
    """Return the last part of a EuroVoc address, which is the id the dataset uses."""
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _select(query: str) -> list[dict[str, str]]:
    response = requests.get(
        ENDPOINT, params={"query": query}, headers={"Accept": "application/sparql-results+json"}, timeout=180
    )
    response.raise_for_status()
    rows = response.json()["results"]["bindings"]
    return [{key: value["value"] for key, value in row.items()} for row in rows]


def fetch_edges(path: Path) -> list[tuple[str, str]]:
    """Return (parent, child) links, downloading them once to a file in the data folder."""
    if not path.exists():
        rows = [row for query in (BROADER_QUERY, TOP_QUERY, PART_QUERY) for row in _select(query)]
        edges = sorted({(concept_id(row["parent"]), concept_id(row["child"])) for row in rows})
        path.write_text(json.dumps(edges), encoding="utf-8")
    return [(parent, child) for parent, child in json.loads(path.read_text(encoding="utf-8"))]


def build_taxonomy(
    edges: Iterable[tuple[str, str]],
    names: Mapping[str, Mapping[str, str]],
    domains: Iterable[str],
    language: str = "en",
) -> Taxonomy:
    """Build a tree from links, keeping nodes the dataset names that can be reached from the given domains.

    The endpoint serves a newer EuroVoc than the dataset, so links involving concepts the dataset does not
    know are
    dropped, and a named concept with no remaining link to a domain is left out and counted by `unreachable`.
    """
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)
    for parent, child in edges:
        if parent in names and child in names:
            parents[child].append(parent)
            children[parent].append(child)
    reached = _reachable(domains, children)
    nodes = {}
    for node_id in reached:
        kept = tuple(sorted(parent for parent in parents[node_id] if parent in reached))
        nodes[node_id] = Node(node_id, names[node_id][language], () if node_id in set(domains) else kept)
    return Taxonomy(nodes)


def _reachable(starts: Iterable[str], children: Mapping[str, list[str]]) -> set[str]:
    found = set(starts)
    pending = list(found)
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def unreachable(taxonomy: Taxonomy, labels: Iterable[str]) -> set[str]:
    """Return the labels that have no place in a taxonomy."""
    return set(labels) - set(taxonomy.nodes)
