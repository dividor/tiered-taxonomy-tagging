"""The LLM methods: a reduced Kähler pipeline, the funnel, label cards, and the evidence check.

The Kähler pipeline here is a reduced form of LLM-Ensemble (arXiv 2607.14882): keywords come from two
models with one prompt, not several models with several prompts. The shape, generate then map then rate then
combine, is kept. The funnel is a reduced form of MATCH (arXiv 2610.25948): one view of a text rather than
eight, and no pruning by tier, because a short second view and pruning both lost correct labels on the
development split, in exploratory work before this repository.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tiered_taxonomy.datasets import first_words
from tiered_taxonomy.llm import Client, Reply, run_all
from tiered_taxonomy.methods.base import Prediction, clamp, parse_selection, quote_found
from tiered_taxonomy.prompts import (
    CARD_SYSTEM,
    CHECK_SYSTEM,
    KEYWORDS_SYSTEM,
    RATE_SYSTEM,
    SELECT_SYSTEM,
    card_message,
    check_message,
    rate_message,
    select_message,
    text_message,
)
from tiered_taxonomy.retrieval import EmbeddingStore, LabelIndex, merge_views, prune_by_tier
from tiered_taxonomy.taxonomy import Taxonomy


@dataclass(frozen=True)
class FunnelSettings:
    """How many candidates the funnel retrieves and keeps, and how it prunes by tier."""

    retrieve_per_view: int = 100
    keep_candidates: int = 100
    prune_tier: int = 2
    prune_keep_share: float = 1.0
    title_words: int = 0


def funnel_candidates(
    text: str, index: LabelIndex, store: EmbeddingStore, settings: FunnelSettings
) -> dict[str, float]:
    """Retrieve across all labels from two views of a text, merge, prune by tier and keep the strongest."""
    views = [first_words(text, settings.title_words), text] if settings.title_words else [text]
    hits = index.search_many(store.embed(views), settings.retrieve_per_view)
    merged = merge_views(hits)
    pruned = prune_by_tier(merged, index.taxonomy, settings.prune_tier, settings.prune_keep_share)
    ordered = sorted(pruned, key=lambda key: (-pruned[key], key))[: settings.keep_candidates]
    return {label: pruned[label] for label in ordered}


def funnel_predict(
    text: str, index: LabelIndex, store: EmbeddingStore, client: Client, settings: FunnelSettings
) -> Prediction:
    """Run the funnel for one text: retrieve, prune, then have a model choose among what is left."""
    candidates = funnel_candidates(text, index, store, settings)
    reply = client.ask(SELECT_SYSTEM, select_message(text, index.taxonomy, candidates))
    return parse_selection(reply, candidates)


def hybrid_candidates(trained: Mapping[str, float], retrieved: Mapping[str, float], each: int) -> list[str]:
    """Return the union of the strongest labels from a trained classifier and from retrieval."""
    strongest = [
        sorted(source, key=lambda key: (-source[key], key))[:each] for source in (trained, retrieved)
    ]
    return sorted(set(strongest[0]) | set(strongest[1]))


def select_from_candidates(
    text: str, taxonomy: Taxonomy, labels: Sequence[str], client: Client
) -> Prediction:
    """Have a model choose among given candidate labels."""
    return parse_selection(client.ask(SELECT_SYSTEM, select_message(text, taxonomy, labels)), set(labels))


def select_from_all(text: str, taxonomy: Taxonomy, labels: Sequence[str], client: Client) -> Prediction:
    """Show a model every label and let it choose. This only works for a small taxonomy."""
    return parse_selection(client.ask(SELECT_SYSTEM, select_message(text, taxonomy, labels)), set(labels))


def parse_keywords(reply: Reply) -> list[str]:
    """Read a `keywords` reply. A malformed reply gives no keywords."""
    if reply.parsed is None or not isinstance(reply.parsed.get("keywords"), list):
        return []
    return [str(keyword).strip() for keyword in reply.parsed["keywords"] if str(keyword).strip()]


def keyword_candidates(
    keywords: Sequence[str], index: LabelIndex, store: EmbeddingStore, per_keyword: int
) -> dict[str, float]:
    """Map keywords to nearby labels, summing similarity over keywords and scaling the result to 0 to 1."""
    if not keywords:
        return {}
    totals: dict[str, float] = {}
    for hits in index.search_many(store.embed(keywords), per_keyword):
        for label, value in hits.items():
            totals[label] = totals.get(label, 0.0) + value
    top = max(totals.values())
    return {label: value / top for label, value in totals.items()}


def parse_ratings(reply: Reply, allowed: Mapping[str, float]) -> dict[str, float]:
    """Read a `ratings` reply as relevance from 0 to 1. A label the model did not rate counts as zero."""
    ratings = dict.fromkeys(allowed, 0.0)
    if reply.parsed is None or not isinstance(reply.parsed.get("ratings"), list):
        return ratings
    for entry in reply.parsed["ratings"]:
        if isinstance(entry, Mapping) and str(entry.get("id")) in allowed:
            ratings[str(entry.get("id"))] = clamp(entry.get("score"), 0.0, 10.0, 0.0) / 10.0
    return ratings


def rate_labels(text: str, taxonomy: Taxonomy, labels: Sequence[str], rater: Client) -> Prediction:
    """Have a model rate given labels against a text. The prediction's scores are its ratings, from 0 to 1."""
    prediction = Prediction()
    if not labels:
        return prediction
    reply = rater.ask(RATE_SYSTEM, rate_message(text, taxonomy, labels))
    prediction.calls = 1
    prediction.malformed = int(reply.parsed is None)
    prediction.labels = parse_ratings(reply, dict.fromkeys(labels, 0.0))
    return prediction


def pipeline_predict(
    text: str,
    index: LabelIndex,
    store: EmbeddingStore,
    generators: Sequence[Client],
    rater: Client,
    per_keyword: int,
    weights: tuple[float, float],
    keep_score: float,
) -> Prediction:
    """Run the reduced Kähler pipeline for one text: generate keywords, map them, rate them, combine the
    scores."""
    prediction = Prediction()
    keywords: list[str] = []
    for client in generators:
        reply = client.ask(KEYWORDS_SYSTEM, text_message(text))
        prediction.calls += 1
        prediction.malformed += int(reply.parsed is None)
        keywords.extend(parse_keywords(reply))
    ensemble = keyword_candidates(sorted(set(keywords)), index, store, per_keyword)
    if not ensemble:
        return prediction
    reply = rater.ask(RATE_SYSTEM, rate_message(text, index.taxonomy, sorted(ensemble)))
    prediction.calls += 1
    prediction.malformed += int(reply.parsed is None)
    relevance = parse_ratings(reply, ensemble)
    for label, value in ensemble.items():
        final = weights[0] * value + weights[1] * relevance[label]
        if final >= keep_score:
            prediction.labels[label] = final
    return prediction


def parse_checks(reply: Reply, text: str, assigned: Sequence[str]) -> dict[str, bool]:
    """Return which labels passed: the model said yes, and its quote really is in the text."""
    passed = dict.fromkeys(assigned, False)
    if reply.parsed is None or not isinstance(reply.parsed.get("checks"), list):
        return passed
    for entry in reply.parsed["checks"]:
        if isinstance(entry, Mapping) and str(entry.get("id")) in passed:
            passed[str(entry.get("id"))] = entry.get("supported") is True and quote_found(
                str(entry.get("quote") or ""), text
            )
    return passed


def evidence_check(text: str, prediction: Prediction, taxonomy: Taxonomy, client: Client) -> Prediction:
    """Keep only the labels a second reading can justify with a quote that really is in the text."""
    assigned = sorted(prediction.labels)
    if not assigned:
        return Prediction(calls=prediction.calls, malformed=prediction.malformed, invalid=prediction.invalid)
    reply = client.ask(CHECK_SYSTEM, check_message(text, taxonomy, assigned))
    passed = parse_checks(reply, text, assigned)
    return Prediction(
        labels={label: prediction.labels[label] for label in assigned if passed[label]},
        quotes={label: prediction.quotes.get(label, "") for label in assigned if passed[label]},
        calls=prediction.calls + 1,
        malformed=prediction.malformed + int(reply.parsed is None),
        invalid=prediction.invalid,
    )


def quote_filter(text: str, prediction: Prediction) -> Prediction:
    """Keep only the labels whose own quote really appears in the text. This needs no extra model call."""
    kept = [label for label in prediction.labels if quote_found(prediction.quotes.get(label, ""), text)]
    return Prediction(
        labels={label: prediction.labels[label] for label in kept},
        quotes={label: prediction.quotes[label] for label in kept},
        calls=prediction.calls,
        malformed=prediction.malformed,
        invalid=prediction.invalid,
    )


def parse_cards(reply: Reply, allowed: Sequence[str]) -> dict[str, str]:
    """Read a `cards` reply, keeping only labels that were asked about."""
    if reply.parsed is None or not isinstance(reply.parsed.get("cards"), list):
        return {}
    wanted = set(allowed)
    cards = {}
    for entry in reply.parsed["cards"]:
        if isinstance(entry, Mapping) and str(entry.get("id")) in wanted and entry.get("definition"):
            cards[str(entry.get("id"))] = str(entry["definition"]).strip()
    return cards


def make_label_cards(
    taxonomy: Taxonomy, client: Client, batch_size: int = 40, workers: int = 8
) -> tuple[dict[str, str], int]:
    """Write a definition for every label, from the taxonomy alone. Returns the cards and how many are
    missing.

    Siblings go together, so that definitions can say how they differ. Labels whose batch failed, for example
    because the reply was cut off, are tried once more in much smaller batches.
    """
    ordered = sorted(sorted(taxonomy.nodes), key=taxonomy.path)

    def write(group: list[str]) -> dict[str, str]:
        return parse_cards(client.ask(CARD_SYSTEM, card_message(taxonomy, group)), group)

    cards: dict[str, str] = {}
    for size in (batch_size, max(1, batch_size // 4)):
        pending = [label for label in ordered if label not in cards]
        groups = [pending[start : start + size] for start in range(0, len(pending), size)]
        for result in run_all(write, groups, workers):
            cards.update(result)
    return cards, len(ordered) - len(cards)


def retrieval_only(queries: np.ndarray, index: LabelIndex, k: int) -> list[dict[str, float]]:
    """Return the nearest labels for each query, with no model involved. This is the baseline for the
    funnel."""
    return index.search_many(queries, k)
