"""Tests for the paper recipes and the combined method, on invented data and with no model calls."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from tiered_taxonomy.llm import Reply, parse_json_object
from tiered_taxonomy.methods import combined as cb
from tiered_taxonomy.methods import kahler as kh
from tiered_taxonomy.retrieval import EmbeddingStore, LabelIndex
from tiered_taxonomy.taxonomy import from_paths

TREE = from_paths(["Health->Water", "Health->Clinics", "Trade->Export"])


class WordEmbeddings(Embeddings):
    """Embeds by counting a few words."""

    WORDS = ("water", "clinics", "export", "health", "trade")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(text.lower().count(word)) + 0.01 for word in self.WORDS] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class Scripted:
    """A client that gives prepared replies in order and remembers what it was asked."""

    def __init__(self, *contents: str) -> None:
        self.replies = deque(contents)
        self.asked: list[str] = []

    def ask(self, _system: str, user: str) -> Reply:
        self.asked.append(user)
        content = self.replies.popleft() if self.replies else "{}"
        return Reply(content, parse_json_object(content), 1, 1, 0.0, False)


@pytest.fixture
def index(tmp_path: Path) -> tuple[LabelIndex, EmbeddingStore]:
    store = EmbeddingStore(WordEmbeddings(), "fake", tmp_path / "v.sqlite")
    return LabelIndex(TREE, store), store


def test_few_shot_message_shows_examples_then_the_text() -> None:
    message = kh.few_shot_message([kh.Example("a law on water", ("water",))], "new text")
    assert message.index("a law on water") < message.index("new text")
    assert '"water"' in message


def test_hybrid_search_finds_labels_by_words_and_meaning(index: Any) -> None:
    labels, store = index
    search = kh.HybridLabelSearch(labels, store)
    hits = search.search(["water"], 2)[0]
    assert next(iter(hits)) == "Health->Water"
    assert search.search([], 2) == []


def test_ensemble_scores_sum_across_keywords_and_scale_to_one() -> None:
    scores = kh.ensemble_scores([{"a": 0.5, "b": 0.2}, {"a": 0.5}])
    assert scores == {"a": 1.0, "b": pytest.approx(0.2)}
    assert kh.ensemble_scores([]) == {}


def test_llm_ensemble_uses_every_model_and_prompt_and_the_paper_weights(index: Any) -> None:
    labels, store = index
    first: Any = Scripted('{"keywords": ["water"]}', '{"keywords": ["water"]}')
    second: Any = Scripted('{"keywords": ["water"]}', '{"keywords": ["clinics"]}')
    ranker: Any = Scripted('{"ratings": [{"id": "Health->Water", "score": 10}]}')
    prompts = [[kh.Example("e1", ("x",))], [kh.Example("e2", ("y",))]]
    settings = kh.RecipeSettings(candidates_per_keyword=1)
    result = kh.llm_ensemble(
        "text", [first, second], prompts, kh.HybridLabelSearch(labels, store), labels, ranker, settings
    )
    assert result.calls == 5
    assert result.labels["Health->Water"] == pytest.approx(0.3 * 1.0 + 0.7 * 1.0)


def test_ki_fsprompt_uses_the_examples_it_is_given(index: Any) -> None:
    labels, store = index
    generator: Any = Scripted('{"keywords": ["water"]}')
    ranker: Any = Scripted('{"ratings": []}')
    similar = [kh.Example("a similar law", ("water",))]
    result = kh.ki_fsprompt(
        "text", similar, [generator], kh.HybridLabelSearch(labels, store), labels, ranker, kh.RecipeSettings()
    )
    assert "a similar law" in generator.asked[0]
    assert result.calls == 2


def test_no_keywords_means_no_ranker_call(index: Any) -> None:
    labels, store = index
    generator: Any = Scripted("nonsense")
    ranker: Any = Scripted()
    result = kh.ki_fsprompt(
        "t", [], [generator], kh.HybridLabelSearch(labels, store), labels, ranker, kh.RecipeSettings()
    )
    assert result.malformed == 1
    assert ranker.asked == []


def test_similar_positions_are_most_similar_first() -> None:
    documents = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])
    assert kh.similar_positions(np.array([1.0, 0.0]), documents, 2) == [1, 2]


def graph() -> cb.LabelGraph:
    return cb.LabelGraph(
        TREE, [["Health->Water", "Trade->Export"], ["Health->Water", "Trade->Export"], ["Health"]]
    )


def test_graph_links_labels_that_appear_together() -> None:
    links = graph()
    assert links.counts["Health->Water"] == 2
    assert links.co_occurrence_support("Health->Water", {"Trade->Export": 0.5}) == pytest.approx(0.5)
    assert links.co_occurrence_support("Health->Water", {}) == 0.0


def test_graph_supports_a_label_through_its_parents_and_children() -> None:
    links = graph()
    assert links.family_support("Health->Water", {"Health": 0.9}) == (0.9, 0.0)
    assert links.family_support("Health", {"Health->Water": 0.8}) == (0.0, 0.8)
    assert links.family_support("unknown", {}) == (0.0, 0.0)


def test_candidates_are_the_union_of_each_sources_strongest() -> None:
    sources = {"a": {"x": 0.9, "y": 0.1}, "b": {"z": 0.5}}
    assert cb.candidates_for(sources, 1) == ["x", "z"]


def test_name_in_text_matches_whole_words_only() -> None:
    assert cb.name_in_text("Water", "clean water here") == 1.0
    assert cb.name_in_text("Water", "waterproof") == 0.0


def test_label_features_match_their_names() -> None:
    sources = {"trained": {"Health->Water": 0.9}, "search": {"Health": 0.4}}
    pooled = cb.pooled_scores(sources)
    row = cb.label_features("Health->Water", sources, ["trained", "search"], "clean water", graph(), pooled)
    named = dict(zip(cb.feature_names(["trained", "search"]), row, strict=True))
    assert named["trained_score"] == pytest.approx(0.9)
    assert named["search_found"] == 0.0
    assert named["sources_agreeing"] == 1.0
    assert named["name_in_text"] == 1.0
    assert named["parent_score"] == pytest.approx(0.4)
    assert named["tier"] == 2.0


def test_the_combiner_learns_to_trust_the_better_signal() -> None:
    rng = np.random.default_rng(1)
    documents = []
    gold = []
    for _ in range(300):
        good, bad = (
            ("Health->Water", "Trade->Export") if rng.random() < 0.5 else ("Trade->Export", "Health->Water")
        )
        documents.append({"reliable": {good: 0.9}, "noisy": {bad: 0.9}})
        gold.append({good})
    links = cb.LabelGraph(TREE, [])
    rows, keys = cb.build_rows(documents, ["text"] * 300, links, ["reliable", "noisy"], 5)
    assert rows.shape[1] == len(cb.feature_names(["reliable", "noisy"]))
    settings = cb.CombinedSettings(max_iterations=30)
    model = cb.train_combiner(rows, keys, gold, settings)
    scored = cb.scores_by_document(model.predict_proba(rows)[:, 1], keys, 300)
    for ranking, labels in zip(scored, gold, strict=True):
        assert max(ranking, key=ranking.__getitem__) in labels


def test_build_rows_with_no_candidates_keeps_its_shape() -> None:
    rows, keys = cb.build_rows([{"a": {}}], ["text"], cb.LabelGraph(TREE, []), ["a"], 5)
    assert rows.shape == (0, len(cb.feature_names(["a"])))
    assert keys == []


def test_choose_threshold_prefers_the_cut_that_gives_the_best_f1() -> None:
    threshold, score = cb.choose_threshold([{"a"}], [{"a": 0.9, "b": 0.5}], [0.1, 0.7])
    assert threshold == 0.7
    assert score == 1.0


def test_needs_llm_picks_unsure_and_rare_labels() -> None:
    ranking = {"sure": 0.95, "unsure": 0.4, "rare": 0.02, "dismissed": 0.01}
    counts = {"sure": 100, "unsure": 100, "rare": 1, "dismissed": 100}
    assert cb.needs_llm(ranking, (0.2, 0.7), counts, 10) == ["rare", "unsure"]


def test_needs_llm_keeps_the_strongest_when_limited() -> None:
    ranking = {"a": 0.5, "b": 0.6, "c": 0.3}
    counts = dict.fromkeys(ranking, 100)
    assert cb.needs_llm(ranking, (0.2, 0.7), counts, 10, most=2) == ["a", "b"]


def test_pooled_scores_take_each_labels_best() -> None:
    assert cb.pooled_scores({"x": {"a": 0.2}, "y": {"a": 0.5, "b": 0.1}}) == {"a": 0.5, "b": 0.1}


def test_second_stage_rows_mark_which_labels_the_llm_saw() -> None:
    rows, keys = cb.second_stage_rows(
        [{"Health->Water": 0.4, "Health": 0.9}], [{"Health->Water": 0.8}], graph()
    )
    named = dict(zip(keys, rows.tolist(), strict=True))
    assert named[(0, "Health->Water")][:3] == pytest.approx([0.4, 1.0, 0.8])
    assert named[(0, "Health")][:3] == pytest.approx([0.9, 0.0, 0.0])
    assert rows.shape[1] == len(cb.SECOND_STAGE_FEATURES)
    assert cb.second_stage_rows([{}], [{}], graph())[0].shape == (0, len(cb.SECOND_STAGE_FEATURES))


DATED = [
    (2001, ["Health->Water"]),
    (2002, ["Health->Water", "Trade->Export"]),
    (2003, ["Trade->Export"]),
    (2004, ["Trade->Export"]),
]


def test_timed_priors_count_recent_use_trend_and_years_since_last_use() -> None:
    priors = cb.TimedPriors.from_documents(DATED, recent_documents=2)
    export_count, export_trend, export_gap = priors.features("Trade->Export")
    water_count, water_trend, water_gap = priors.features("Health->Water")
    assert export_count == pytest.approx(np.log1p(2))
    assert water_count == 0.0
    assert export_trend > 0 > water_trend
    assert (export_gap, water_gap) == (0.0, 2.0)


def test_timed_priors_leave_trend_and_gap_missing_for_a_label_never_used() -> None:
    count, trend, gap = cb.TimedPriors.from_documents(DATED, recent_documents=2).features("Health->Clinics")
    assert count == 0.0
    assert np.isnan(trend)
    assert np.isnan(gap)


def test_timed_priors_refuse_documents_out_of_date_order() -> None:
    with pytest.raises(ValueError, match="date order"):
        cb.TimedPriors.from_documents(list(reversed(DATED)), recent_documents=2)


def test_timed_priors_refuse_a_window_as_large_as_the_whole() -> None:
    with pytest.raises(ValueError, match="must be smaller"):
        cb.TimedPriors.from_documents(DATED, recent_documents=4)


def test_a_graph_with_priors_adds_the_time_features_to_each_row() -> None:
    priors = cb.TimedPriors.from_documents(DATED, recent_documents=2)
    links = cb.LabelGraph(TREE, [labels for _, labels in DATED], priors=priors)
    rows, keys = cb.build_rows([{"a": {"Trade->Export": 0.8}}], ["text"], links, ["a"], 5)
    names = cb.feature_names(["a"], timed=True)
    assert keys == [(0, "Trade->Export")]
    assert rows.shape == (1, len(names))
    named = dict(zip(names, rows[0], strict=True))
    assert named["years_since_used"] == 0.0
    assert named["log_recent_count"] == pytest.approx(np.log1p(2))
    empty, _ = cb.build_rows([{"a": {}}], ["text"], links, ["a"], 5)
    assert empty.shape == (0, len(names))
