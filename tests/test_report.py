"""Tests for the results report, built from the results committed in this repository."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from tiered_taxonomy import report as rp
from tiered_taxonomy.report_page import benchmark_rows, build_page, published_measure

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs() -> rp.Inputs:
    return rp.load_inputs(REPO)


def row(**changes: object) -> rp.Row:
    base = rp.Row(
        "k", "omikuji", "Omikuji", "", "Trained", {"exact": {"f1": 0.5}, "tiers": {"1": {"f1": 0.6}}}
    )
    return replace(base, **changes)


def test_every_method_and_reference_in_the_descriptions_is_complete(inputs: rp.Inputs) -> None:
    kinds = {k["kind"] for k in inputs.methods["kinds"]}
    for name, entry in inputs.methods["methods"].items():
        assert entry["kind"] in kinds, name
        assert entry["summary"] and len(entry["steps"]) >= 3, name
        assert set(entry["refs"]) <= set(inputs.methods["references"]), name
    for name, ref in inputs.methods["references"].items():
        assert ref["link"].startswith("https://"), name
        assert ref["title"] and ref["authors"] and ref["where"], name


def test_every_notebook_a_card_names_is_in_the_repository(inputs: rp.Inputs) -> None:
    existing = {path.name[:2] for path in (REPO / "notebooks").glob("[0-9][0-9]_*.ipynb")}
    for name, entry in inputs.methods["methods"].items():
        numbers = re.findall(r"\b\d\d\b", entry["notebook"])
        numbers += re.findall(r"notebooks? (\d\d)\b", entry.get("differs", ""))
        assert numbers and set(numbers) <= existing, (name, numbers)
    for test_set in ("eurlex", "humset", rp.HUMSET_V2):
        for method_row in rp.method_rows(inputs, test_set):
            assert set(re.findall(r"\b\d\d\b", method_row.notebook)) <= existing, method_row.name


TREE = {
    "nodes": {
        "t": ["trade", 1, ["m"], 0],
        "m": ["marketing", 2, ["p", "x"], 0],
        "p": ["preparation for market", 3, ["l", "k"], 5],
        "l": ["labelling", 4, [], 9],
        "k": ["packaging", 4, [], 3],
        "x": ["exports", 3, [], 2],
    }
}


def test_label_path_runs_from_tier_one_down_to_the_label() -> None:
    assert rp.label_path(TREE, "labelling") == [
        (1, "trade"),
        (2, "marketing"),
        (3, "preparation for market"),
        (4, "labelling"),
    ]
    with pytest.raises(ValueError, match="one label"):
        rp.label_path(TREE, "missing")
    two_parents = {"nodes": {**TREE["nodes"], "x": ["exports", 3, ["l"], 2]}}
    with pytest.raises(ValueError, match="more than one parent"):
        rp.label_path(two_parents, "labelling")


def test_near_miss_paths_needs_two_labels_under_one_parent() -> None:
    assert rp.near_miss_paths(TREE, "labelling", "packaging")[-1] == (4, "labelling")
    with pytest.raises(ValueError, match="same parent"):
        rp.near_miss_paths(TREE, "labelling", "exports")


def test_near_miss_panel_names_the_tier_of_every_label(inputs: rp.Inputs) -> None:
    page = build_page(REPO)
    panel = " ".join(
        page[page.index("<figure class=nearmiss>") : page.index("</figure>", page.index("nearmiss"))].split()
    )
    assert re.findall(r"<span class=t>([^<]+)</span>", panel) == ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
    assert "the tier 4 label <q>labelling</q>" in panel
    assert "Exact label" not in panel
    assert "It is not quite a tree: 244 labels have more than one parent." in panel
    assert "through every parent" in panel
    assert (
        "our copy of EuroVoc has 7,372 labels, only 693 of them in tiers 1 to 3; "
        "the MultiEURLEX paper counts 7,390" in panel
    )
    assert "ours match them for 99.4% of test laws at tier 1, 84.8%" in " ".join(page.split())


def test_every_subtitle_starts_with_a_capital(inputs: rp.Inputs) -> None:
    subs = {
        f"{t} {r.name}": r.sub for t in ("eurlex", "humset", rp.HUMSET_V2) for r in rp.method_rows(inputs, t)
    }
    laws, excerpts = benchmark_rows(inputs)
    subs |= {f"benchmark {name}": sub for name, sub, _, _ in [*laws, *excerpts]}
    for name, sub in subs.items():
        assert not sub or sub[0].isupper(), (name, sub)


def test_first_combined_attempt_says_what_it_weighs(inputs: rp.Inputs) -> None:
    _, sub = rp.combined_name(inputs, "combined_first:newer", "eurlex")
    assert sub == (
        "Weighs fine-tuned Legal-BERT, similar documents, label search, fine-tuned small LLM and fine-tuned "
        "reranker, chosen on development laws"
    )


def test_rows_are_best_first_with_distinct_names(inputs: rp.Inputs) -> None:
    for test_set in ("eurlex", "humset"):
        rows = rp.method_rows(inputs, test_set)
        assert [r.f1 for r in rows] == sorted((r.f1 for r in rows), reverse=True)
        assert len({r.name for r in rows}) == len(rows)
    keys = {r.key for r in rp.method_rows(inputs, "eurlex")}
    assert "label_search:train_only" in keys
    assert "label_search:newer" not in keys


def test_combined_rows_are_named_by_attempt_and_training_data(inputs: rp.Inputs) -> None:
    names = {r.key: (r.name, r.sub) for r in rp.method_rows(inputs, "eurlex") if r.method == "combined"}
    assert names["combined_first:newer"][0] == "Combined method, first attempt, with newer laws"
    assert names["combined_second:train_only"][0] == "Combined method, second attempt, training laws only"
    assert names["combined_second:newer"][1] == "Adds GLiNER2.5 and each label's recent history"
    assert names["combined_second:train_only"][1] == "Adds each label's recent history"


def test_duplicate_names_raise(inputs: rp.Inputs) -> None:
    methods = json.loads(json.dumps(inputs.methods))
    methods["methods"]["walk_strict"]["name"] = methods["methods"]["walk_branching"]["name"]
    with pytest.raises(ValueError, match="share a name"):
        rp.method_rows(replace(inputs, methods=methods), "eurlex")


def test_cost_text_covers_each_kind_of_cost() -> None:
    assert rp.cost_text(row(llm_cost=0.4), "laws") == ("$0.40 per 1,000 laws", 0.4)
    assert rp.cost_text(row(laptop=True)) == ("laptop", 0.0)
    assert rp.cost_text(row(gpu_cost=1.5)) == ("$1.50 GPU, once", 1.5)
    assert rp.cost_text(row(gpu_cost=1.5, gpu_partial=True))[0] == "$1.50 and more GPU, once"
    assert rp.cost_text(row())[0] == "GPU, cost not recorded"


def test_gpu_cost_adds_runs_and_flags_unrecorded_ones() -> None:
    jobs = {"prices": {"a": 2.0}, "runs": {"classifier_humset": {"hours": 1.5, "flavor": "a"}}}
    assert rp.gpu_cost(jobs, "humset", ["classifier", "omikuji"]) == (3.0, False)
    assert rp.gpu_cost(jobs, "humset", ["xr_transformer"]) == (0.0, True)


def test_ref_html_links_the_title_and_escapes_text() -> None:
    made = rp.ref_html(
        {
            "authors": "A <b> and C",
            "year": 2026,
            "title": "T & U",
            "where": "arXiv",
            "link": "https://x.org/1",
        }
    )
    assert '<a href="https://x.org/1"' in made
    assert "T &amp; U</a>" in made
    assert "A &lt;b&gt; and C (2026). arXiv." in made
    assert "()" not in rp.ref_html(
        {"authors": "A", "year": None, "title": "T", "where": "W", "link": "https://x"}
    )


def test_interval_reports_size_and_ordered_bounds() -> None:
    assert rp.interval([-0.02, -0.03, -0.01]).startswith("0.02 <span class=ci>(95% interval 0.01 to 0.03)")


def test_check_findings_raises_when_a_fact_changes(inputs: rp.Inputs) -> None:
    laws, excerpts = rp.method_rows(inputs, "eurlex"), rp.method_rows(inputs, rp.HUMSET_V2)
    rp.check_findings(laws, excerpts, inputs.results)
    demoted = [
        replace(r, scores={**r.scores, "exact": {"f1": 0.0}}) if r.kind == "Combined" else r for r in laws
    ]
    demoted.sort(key=lambda r: -r.f1)
    with pytest.raises(ValueError, match="the combined method is best"):
        rp.check_findings(demoted, excerpts, inputs.results)


def test_page_has_a_card_for_every_row_and_every_source(inputs: rp.Inputs) -> None:
    page = build_page(REPO)
    data = re.search(r'<script type="application/json" id="cards-data">(.*?)</script>', page, re.S)
    assert data is not None
    cards = json.loads(data.group(1))
    keys = set(re.findall(r'data-key="([^"]+)"', page))
    assert keys == set(cards)
    for ref in inputs.methods["references"].values():
        assert ref["link"] in page
    assert "<title>Taxonomy Tagger Results</title>" in page
    assert "{" not in re.sub(r"<(style|script)[^>]*>.*?</\1>", "", page, flags=re.S)


def test_tier_mix_explains_the_gap_and_raises_when_it_no_longer_holds(inputs: rp.Inputs) -> None:
    laws, excerpts = rp.method_rows(inputs, "eurlex"), rp.method_rows(inputs, rp.HUMSET_V2)
    mix = rp.tier_mix(inputs, laws, excerpts)
    assert mix.law_top_three < mix.law_nodes
    assert 0.5 < mix.law_below_three <= 1 and 0 < mix.excerpt_tier_one < 1
    raised = [replace(laws[0], scores={**laws[0].scores, "exact": {"f1": 0.99}}), *laws[1:]]
    with pytest.raises(ValueError, match="not below every tier"):
        rp.tier_mix(inputs, raised, excerpts)
    like = dict(inputs.results["14_like_for_like"])
    gold = like["eurlex"]["gold"]
    like["eurlex"] = {**like["eurlex"], "gold": {**gold, "by_tier": {**gold["by_tier"], "2": 1}}}
    grouped = rp.Inputs({**inputs.results, "14_like_for_like": like}, inputs.config, inputs.methods)
    with pytest.raises(ValueError, match="labels at tier"):
        rp.tier_mix(grouped, laws, excerpts)


def test_the_humanitarian_combined_row_says_what_it_weighs(inputs: rp.Inputs) -> None:
    (row,) = [r for r in rp.method_rows(inputs, "humset") if r.kind == "Combined"]
    assert row.sub == (
        "Weighs fine-tuned HumBERT, label search and XR-Transformer; the same in both attempts"
    )
    assert rp.plain_case("XR-Transformer") == "XR-Transformer"
    assert rp.plain_case("Omikuji, with newer laws") == "Omikuji, with newer laws"
    assert rp.plain_case("Similar documents") == "similar documents"
    second = dict(inputs.results["11_second_attempt"])
    second["humset"] = {**second["humset"], "sources": ["classifier"]}
    changed = rp.Inputs({**inputs.results, "11_second_attempt": second}, inputs.config, inputs.methods)
    assert rp.combined_name(changed, "combined")[1].endswith("the second attempt's choice")


def test_page_has_label_and_example_buttons_for_both_datasets(inputs: rp.Inputs) -> None:
    page = build_page(REPO)
    for test_set in ("eurlex", "humset"):
        assert f'data-browse=labels data-set="{test_set}"' in page
        assert f'data-browse=examples data-set="{test_set}"' in page
    data = re.search(r'<script type="application/json" id="labels-data">(.*?)</script>', page, re.S)
    assert data is not None
    browser = json.loads(data.group(1))
    assert len(browser["eurlex"]["tree"]["nodes"]) == 7372
    for test_set in ("eurlex", "humset"):
        nodes = browser[test_set]["tree"]["nodes"]
        assert all(label in nodes for record in browser[test_set]["examples"] for label in record["labels"])
    assert "Browse all 7,372 labels" in page and "<dialog id=labels" in page and "<dialog id=examples" in page
    assert "<div class=freq>" in page
    assert browser["eurlex"]["train_documents"] == 55000 and browser["humset"]["train_documents"] > 0


def test_both_tables_show_rare_label_recall_as_decimals(inputs: rp.Inputs) -> None:
    kind_css = {k["kind"]: k["css"] for k in inputs.methods["kinds"]}
    laws = rp.results_table(rp.method_rows(inputs, "eurlex"), "eurlex", "law", kind_css)
    rows = rp.method_rows(inputs, rp.HUMSET_V2)
    excerpts = rp.results_table(rows, rp.HUMSET_V2, "excerpt", kind_css, published_measure(inputs, rows))
    for table in (laws, excerpts):
        assert "Recall, higher is better" in table and "Rarest quarter" in table
        assert "Never seen" in table and "%</td>" not in table
    assert excerpts.count('class="num none"') == len(rows)
    assert 'class="num none"' not in laws
    best = rp.method_rows(inputs, "eurlex")[0].scores["rarity"]["rare"]["recall"]
    assert f">{best:.2f}</td>" in laws


def test_tier_sentence_names_each_tier_a_combined_row_does_not_lead(inputs: rp.Inputs) -> None:
    from tiered_taxonomy.report_page import tier_sentence

    laws, excerpts = rp.method_rows(inputs, "eurlex"), rp.method_rows(inputs, "humset")
    combined_only = [r for r in excerpts if r.kind == "Combined"] + [
        replace(r, scores={**r.scores, "tiers": {t: {"f1": 0.0} for t in r.scores["tiers"]}})
        for r in excerpts
        if r.kind != "Combined"
    ]
    assert tier_sentence(laws, combined_only).endswith("at every tier.")
    found = rp.tiers_not_led_by_combined(excerpts)
    sentence = tier_sentence(laws, excerpts)
    for tier, row, lead in found:
        assert f"tier {tier} on the excerpts" in sentence and f"{lead:.3f} ahead" in sentence
        assert row.kind != "Combined" and lead > 0
