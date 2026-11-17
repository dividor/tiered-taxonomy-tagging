"""The instructions each method gives a model. Replies are always one JSON object."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from tiered_taxonomy.taxonomy import Taxonomy

_LANGUAGE_NOTE = (
    "The text may be in English, French or Spanish. The labels are in English. "
    "Each listed line starts with its id: copy ids exactly as written."
)

SELECT_SYSTEM = (
    "You assign taxonomy labels to a piece of text. Choose every listed label the text is substantially "
    "about, and "
    f"no others. {_LANGUAGE_NOTE} Use only ids from the list. Reply with one JSON object: "
    '{"labels": [{"id": "<id>", "confidence": <0 to 1>, "quote": "<short exact quote from the text>"}]}. '
    'If none fit, reply {"labels": []}.'
)

KEYWORDS_SYSTEM = (
    "You are a subject indexer. Write the topics a piece of text is about as short keywords, in English, "
    "as a library "
    f"cataloguer would. {_LANGUAGE_NOTE} Examples of good keywords: 'school feeding', 'customs duties', "
    f"'refugee "
    'camps\', \'fruit imports\'. Reply with one JSON object: {"keywords": ["<keyword>", ...]}.'
)

RATE_SYSTEM = (
    "You judge how well candidate subject labels describe a piece of text. Rate each from 0 (irrelevant) "
    "to 10 "
    f"(a main subject of the text). {_LANGUAGE_NOTE} Reply with one JSON object: "
    '{"ratings": [{"id": "<id>", "score": <0 to 10>}]} and rate every candidate.'
)

WALK_SYSTEM = (
    "You place a piece of text in a taxonomy, one level at a time. From the listed options, choose those "
    "the text is "
    f"substantially about. {_LANGUAGE_NOTE} Reply with one JSON object: "
    '{"selected": ["<id>", ...], "stop_here": <true or false>}. '
    "Set stop_here to true if the current heading itself "
    'describes the text and nothing more specific below it fits. If nothing fits, reply {"selected": [], '
    '"stop_here": false}.'
)

CHECK_SYSTEM = (
    "You check whether labels assigned to a text are justified. For each label, find a short exact quote "
    "in the text "
    "that shows the text is substantially about that label, using the label's description. Do not guess: "
    "if no passage "
    f"justifies it, say so. {_LANGUAGE_NOTE} Reply with one JSON object: "
    '{"checks": [{"id": "<id>", "supported": <true or false>, "quote": "<exact quote or empty>"}]}.'
)

CARD_SYSTEM = (
    "You write short definitions for taxonomy labels, working only from each label's place in the "
    "taxonomy. For each "
    "label, write one sentence saying what it covers and, where it could be confused with a sibling, what "
    "it does not "
    'cover. Reply with one JSON object: {"cards": [{"id": "<id>", "definition": "<sentence>"}]}.'
)


def describe_label(taxonomy: Taxonomy, node_id: str) -> str:
    """Return one line for a label: its id, its path if the id does not already say it, and any definition.

    Showing a label two ways led models to reply with the wrong one, so an id that spells out its path
    stands alone.
    """
    node = taxonomy.nodes[node_id]
    definition = f" | {node.definition}" if node.definition else ""
    if not node_id.isdigit():
        return f"{node_id}{definition}"
    return f"{node_id} | {taxonomy.path(node_id)}{definition}"


def text_message(text: str) -> str:
    """Return the text to label, marked off from the instructions."""
    return f"TEXT:\n{text}"


def select_message(text: str, taxonomy: Taxonomy, candidates: Iterable[str]) -> str:
    """Return a request to choose labels from candidates."""
    lines = "\n".join(describe_label(taxonomy, node_id) for node_id in candidates)
    return f"LABELS:\n{lines}\n\n{text_message(text)}"


def rate_message(text: str, taxonomy: Taxonomy, candidates: Iterable[str]) -> str:
    """Return a request to rate candidate labels."""
    lines = "\n".join(describe_label(taxonomy, node_id) for node_id in candidates)
    return f"CANDIDATES:\n{lines}\n\n{text_message(text)}"


def check_message(text: str, taxonomy: Taxonomy, labels: Iterable[str]) -> str:
    """Return a request to justify assigned labels."""
    lines = "\n".join(describe_label(taxonomy, node_id) for node_id in labels)
    return f"ASSIGNED LABELS:\n{lines}\n\n{text_message(text)}"


def walk_message(
    text: str,
    taxonomy: Taxonomy,
    heading: str | None,
    options: Sequence[str],
    look_ahead: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Return a request to choose among one node's children, optionally showing the likely labels under
    each."""
    lines = []
    for node_id in options:
        hint = ""
        if look_ahead and look_ahead.get(node_id):
            names = ", ".join(taxonomy.nodes[leaf].name for leaf in look_ahead[node_id])
            hint = f" | leads to labels such as: {names}"
        lines.append(f"{describe_label(taxonomy, node_id)}{hint}")
    place = f"CURRENT HEADING: {heading}\n" if heading else "CURRENT HEADING: top of the taxonomy\n"
    return f"{place}OPTIONS:\n" + "\n".join(lines) + f"\n\n{text_message(text)}"


def card_message(taxonomy: Taxonomy, labels: Iterable[str]) -> str:
    """Return a request to write definitions for labels."""
    lines = "\n".join(f"{node_id} | {taxonomy.path(node_id)}" for node_id in labels)
    return f"LABELS:\n{lines}"
