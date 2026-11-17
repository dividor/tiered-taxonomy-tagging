"""The two LLM recipes from Kähler et al. (arXiv 2607.14882), as closely as these test sets and models allow.

LLM-Ensemble (Kluge and Kähler, 2025) has five stages: several models complete few-shot prompts without
seeing the vocabulary; each keyword is mapped to labels by hybrid vector and BM25 search; similarities are
summed across models and prompts; one more model rates each label from 0 to 10; and the final score is 0.3
times the ensemble score plus 0.7 times relevance. KI-FSPrompt (Kähler et al., 2025) replaces the fixed
examples with training documents similar to the text at hand.

What cannot be matched is stated here, so that it is stated wherever results appear. Of the paper's models,
only the ranker, Llama-3.1-8B-Instruct, is on Hugging Face's router, so the others are replaced by the open
models named in the configuration. The original ensemble was selected from 135 combinations in a much larger
study, while here the models are fixed in advance. The paper's few-shot examples are in German from its own
data, so examples here come from each test set's training split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from tiered_taxonomy.datasets import first_words
from tiered_taxonomy.llm import Client
from tiered_taxonomy.methods.base import Prediction
from tiered_taxonomy.methods.llm_methods import parse_keywords, parse_ratings
from tiered_taxonomy.prompts import RATE_SYSTEM, rate_message
from tiered_taxonomy.retrieval import EmbeddingStore, LabelIndex

COMPLETE_SYSTEM = (
    "You are a subject indexer. Read each text and list the subjects a library cataloguer would assign, as "
    "short "
    'terms in English. Reply with one JSON object: {"keywords": ["<term>", ...]}.'
)
EXAMPLE_WORDS = 120


@dataclass(frozen=True)
class Example:
    """A few-shot example: a text and the names of its gold labels."""

    text: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RecipeSettings:
    """The scoring settings shared by both recipes, from the paper."""

    ensemble_weight: float = 0.3
    relevance_weight: float = 0.7
    candidates_per_keyword: int = 3
    keep_score: float = 0.0
    vector_share: float = 0.5
    max_rated: int = 60


def few_shot_message(examples: Sequence[Example], text: str) -> str:
    """Return a few-shot prompt: worked examples, then the text to index."""
    parts = []
    for example in examples:
        keywords = ", ".join(f'"{keyword}"' for keyword in example.keywords)
        parts.append(f'TEXT:\n{first_words(example.text, EXAMPLE_WORDS)}\nJSON: {{"keywords": [{keywords}]}}')
    parts.append(f"TEXT:\n{text}\nJSON:")
    return "\n\n".join(parts)


class HybridLabelSearch:
    """Maps a keyword to labels with vector search and BM25 together, as the paper's Weaviate setup does."""

    def __init__(self, index: LabelIndex, store: EmbeddingStore, vector_share: float = 0.5) -> None:
        self._index = index
        self._store = store
        self._share = vector_share
        names = [index.taxonomy.nodes[node_id].name.lower().split() for node_id in index.ids]
        self._bm25 = BM25Okapi(names)

    def search(self, keywords: Sequence[str], k: int) -> list[dict[str, float]]:
        """Return each keyword's top labels. Both kinds of score are scaled to 0 to 1 and then blended."""
        if not keywords:
            return []
        vectors = self._store.embed(list(keywords))
        results = []
        for keyword, vector in zip(keywords, vectors, strict=True):
            dense = np.clip(self._index.vectors @ vector, 0, None)
            sparse = np.asarray(self._bm25.get_scores(keyword.lower().split()), dtype=float)
            top_sparse = sparse.max() or 1.0
            blended = self._share * dense + (1 - self._share) * sparse / top_sparse
            picks = np.argsort(-blended)[:k]
            results.append({self._index.ids[int(i)]: float(blended[i]) for i in picks})
        return results


def ensemble_scores(hits_per_keyword: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Sum each label's similarity across all keywords from all models and prompts, scaled so the best is
    1."""
    totals: dict[str, float] = {}
    for hits in hits_per_keyword:
        for label, value in hits.items():
            totals[label] = totals.get(label, 0.0) + value
    if not totals:
        return {}
    top = max(totals.values()) or 1.0
    return {label: value / top for label, value in totals.items()}


def complete_all(
    text: str, generators: Sequence[Client], prompts: Sequence[Sequence[Example]], prediction: Prediction
) -> list[str]:
    """Run every model on every few-shot prompt, returning all the keywords."""
    keywords: list[str] = []
    for client in generators:
        for examples in prompts:
            reply = client.ask(COMPLETE_SYSTEM, few_shot_message(examples, text))
            prediction.calls += 1
            prediction.malformed += int(reply.parsed is None)
            keywords.extend(parse_keywords(reply))
    return keywords


def rate_and_combine(
    text: str,
    keywords: Sequence[str],
    search: HybridLabelSearch,
    index: LabelIndex,
    ranker: Client,
    settings: RecipeSettings,
    prediction: Prediction,
) -> Prediction:
    """Map keywords to labels, have the ranker rate the strongest, and combine the two scores."""
    hits = search.search(sorted(set(keywords)), settings.candidates_per_keyword)
    ensemble = ensemble_scores(hits)
    if not ensemble:
        return prediction
    strongest = sorted(ensemble, key=lambda label: (-ensemble[label], label))[: settings.max_rated]
    reply = ranker.ask(RATE_SYSTEM, rate_message(text, index.taxonomy, strongest))
    prediction.calls += 1
    prediction.malformed += int(reply.parsed is None)
    relevance = parse_ratings(reply, {label: ensemble[label] for label in strongest})
    for label in strongest:
        final = settings.ensemble_weight * ensemble[label] + settings.relevance_weight * relevance[label]
        if final >= settings.keep_score:
            prediction.labels[label] = final
    return prediction


def llm_ensemble(
    text: str,
    generators: Sequence[Client],
    prompts: Sequence[Sequence[Example]],
    search: HybridLabelSearch,
    index: LabelIndex,
    ranker: Client,
    settings: RecipeSettings,
) -> Prediction:
    """Run the LLM-Ensemble recipe for one text, with fixed few-shot prompts."""
    prediction = Prediction()
    keywords = complete_all(text, generators, prompts, prediction)
    return rate_and_combine(text, keywords, search, index, ranker, settings, prediction)


def ki_fsprompt(
    text: str,
    similar: Sequence[Example],
    generators: Sequence[Client],
    search: HybridLabelSearch,
    index: LabelIndex,
    ranker: Client,
    settings: RecipeSettings,
) -> Prediction:
    """Run the KI-FSPrompt recipe for one text, with examples retrieved for this text."""
    prediction = Prediction()
    keywords = complete_all(text, generators, [similar], prediction)
    return rate_and_combine(text, keywords, search, index, ranker, settings, prediction)


def similar_positions(query: np.ndarray, documents: np.ndarray, count: int) -> list[int]:
    """Return the positions of the documents most similar to a query, most similar first."""
    scores = documents @ query
    picks = np.argpartition(-scores, min(count, len(scores) - 1))[:count]
    return sorted((int(i) for i in picks), key=lambda i: (-float(scores[i]), i))
