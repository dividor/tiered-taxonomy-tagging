"""Tests for the report's comparison with HumBERT and the HumSet tables' headings, built from the results
committed in this repository."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from tiered_taxonomy import report as rp
from tiered_taxonomy.report_page import (
    as_share,
    build_page,
    check_v2_macro_lead,
    eurlex_source,
    exact_lead,
    first_version_line,
    header,
    humset_source,
    published_measure,
    with_links,
)
from tiered_taxonomy.report_papers import Published

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs() -> rp.Inputs:
    return rp.load_inputs(REPO)


def row(**changes: object) -> rp.Row:
    base = rp.Row(
        "k", "omikuji", "Omikuji", "", "Trained", {"exact": {"f1": 0.5}, "tiers": {"1": {"f1": 0.6}}}
    )
    return replace(base, **changes)


def test_humbert_rerun_sets_both_runs_beside_print_and_ours(inputs: rp.Inputs) -> None:
    rerun = rp.humbert_rerun(inputs.results)
    assert set(rerun) == {
        "printed",
        "paper",
        "ours",
        "combined",
        "combined_v2",
        "combined_v2_intervals",
        "paper_gap",
        "trainer_table",
        "trainer_misses",
    }
    assert rerun["paper_gap"] == max(abs(rerun["paper"][g] - rerun["printed"][g]) for g in rp.HUMSET_GROUPS)
    page = build_page(REPO)
    assert "On HumSet 1.0.0, the test set" not in page and "We first ran every method on" in page
    assert page.count("HumBERT recipe, rerun") >= 3


def test_humset_v2_rows_come_from_notebook_17_on_the_whole_test_set(inputs: rp.Inputs) -> None:
    rows = rp.method_rows(inputs, rp.HUMSET_V2)
    assert {row.notebook for row in rows} == {"17"}
    assert {row.key for row in rows} >= {"combined", "classifier", "ki_fsprompt", "humbert_recipe"}
    full = inputs.results["17_humset_full_test"][rp.HUMSET_V2]
    assert rp.like_for_like(inputs, rp.HUMSET_V2) == full
    assert full["sizes"]["test"] == inputs.results["13_humbert_reproduction"]["2.0.0"]["run"]["sizes"]["test"]
    assert rp.like_for_like(inputs, "humset") == inputs.results["14_like_for_like"]["humset"]
    (combined,) = [row for row in rows if row.kind == "Combined"]
    assert combined.sub.endswith(", chosen on development excerpts")
    page = build_page(REPO)
    assert 'data-key="humset_v2|combined"' in page
    assert "Humanitarian excerpts (HumSet 2.0.0)" in page


def test_groups_clear_of_print_need_the_whole_interval_above_it() -> None:
    rerun = {
        "printed": dict.fromkeys(rp.HUMSET_GROUPS, 0.5),
        "combined_v2_intervals": {g: [0.49 if g == "Sectors" else 0.51, 0.6] for g in rp.HUMSET_GROUPS},
    }
    assert rp.groups_clear_of_print(rerun) == [g for g in rp.HUMSET_GROUPS if g != "Sectors"]


def test_check_findings_raises_when_the_trainers_averaging_matches_print(inputs: rp.Inputs) -> None:
    laws, excerpts = rp.method_rows(inputs, "eurlex"), rp.method_rows(inputs, rp.HUMSET_V2)
    rerun = json.loads(json.dumps(inputs.results["13_humbert_reproduction"]))
    rerun["2.0.0"]["matches_printed"] = dict.fromkeys(rerun["2.0.0"]["matches_printed"], True)
    with pytest.raises(ValueError, match="trainer's own averaging"):
        rp.check_findings(laws, excerpts, {**inputs.results, "13_humbert_reproduction": rerun})


def test_only_humset_rows_carry_the_papers_measure_from_notebooks_20_and_21(inputs: rp.Inputs) -> None:
    assert all(row.paper is None for row in rp.method_rows(inputs, "eurlex"))
    sources = {
        "humset": inputs.results["16_paper_measure"]["humset"],
        rp.HUMSET_V2: inputs.results["17_humset_full_test"][rp.HUMSET_V2]["paper"],
    }
    for test_set, saved in sources.items():
        for found in rp.method_rows(inputs, test_set):
            family = (
                "combined" if found.kind == "Combined" else "llm" if found.llm_cost is not None else "ranked"
            )
            assert found.paper == saved[family][found.key]


def test_the_papers_measure_is_a_column_of_the_humset_table_only() -> None:
    page = build_page(REPO)
    assert page.count('data-sort="num" data-col="6" class="num">Macro F1<button type=button class=info') == 1
    assert (
        page.count(
            'data-sort="num" data-col="2" class="num">Macro F1, sectors and pillars (paper&#x27;s Avg)<button'
        )
        == 1
    )
    assert "Paper's macro F1" not in page
    assert "Whether that rerun reproduces the published figures depends on how subpillar scores" in page
    assert "the published result holds" not in page


def test_results_table_refuses_rows_with_and_without_the_papers_measure() -> None:
    rows = [row(paper={"Mean": 0.5}), row(key="other")]
    with pytest.raises(ValueError, match="paper's measure"):
        rp.results_table(rows, "humset", "excerpt", {"Trained": "t"})


def test_info_puts_an_escaped_tooltip_on_a_button_after_the_heading() -> None:
    made = rp.info("Tier <1>", 'Says "why" & how')
    assert made.startswith("Tier &lt;1&gt;<button type=button class=info ")
    assert 'data-tip="Says &quot;why&quot; &amp; how"' in made
    assert 'aria-label="What Tier &lt;1&gt; means"' in made


def test_every_results_table_heading_has_a_tooltip() -> None:
    head = rp._table_head(["1", "2"], "excerpt", published_stub("k"))
    for label in ("Method", "Kind", "Exact labels", "Tier 1", "Tier 2", "Macro F1", "Rarest quarter"):
        assert f'aria-label="What {label} means"' in head, label
    assert 'aria-label="What Labels per excerpt means"' in head
    assert "each excerpt, on average" in head
    assert 'data-tip="Per label."' in head
    assert (
        'aria-label="What Macro F1, sectors and pillars (paper&#x27;s Avg) means"' in head
        and 'data-tip="Tier one."' in head
    )
    assert "colspan=5 class=group>F1 score" in head
    assert "Macro F1" not in rp._table_head(["1"], "law", None)
    assert "sectors and pillars" not in rp._table_head(["1"], "law", None)


def test_tier_one_average_is_the_mean_of_sectors_and_both_pillar_groups() -> None:
    scores = {
        "Sectors": 0.78,
        "Pillars 1D": 0.70,
        "Subpillars 1D": 0.0,
        "Pillars 2D": 0.71,
        "Subpillars 2D": 0.0,
    }
    assert rp.tier_one_average(scores) == pytest.approx(0.73)


def test_the_benchmark_table_shows_the_papers_own_tier_one_average() -> None:
    page = build_page(REPO)
    assert 'aria-label="What Tier-1 average means"' in page
    assert 'Its own "Avg" column is the tier-1 average in the table' in " ".join(page.split())


def test_check_findings_raises_when_the_combined_tier_one_average_is_not_above_the_papers(
    inputs: rp.Inputs,
) -> None:
    laws, excerpts = rp.method_rows(inputs, "eurlex"), rp.method_rows(inputs, rp.HUMSET_V2)
    v2 = dict(inputs.results["15_humset_v2"])
    full = dict(v2["full_test"])
    full["scores"] = {**full["scores"], "Sectors": 0.5}
    v2["full_test"] = full
    with pytest.raises(ValueError, match="tier-1 average"):
        rp.check_findings(laws, excerpts, {**inputs.results, "15_humset_v2": v2})


def test_the_recipe_row_says_what_the_paper_reported_and_how_ours_differs(inputs: rp.Inputs) -> None:
    printed = inputs.results["12_published_benchmarks"]["printed"]["humset"]["HumBERT, best"]["Mean"]
    rows = rp.method_rows(inputs, rp.HUMSET_V2)
    published = published_measure(inputs, rows)
    assert published.row_key == "humbert_recipe"
    assert published.note == f"paper {printed:.2f}"
    full = inputs.results["17_humset_full_test"][rp.HUMSET_V2]
    assert f"The HumBERT paper (Tamagnone et al., 2023) reports {printed:.3f}" in published.note_tip
    assert f"on these same {full['sizes']['test']:,} test excerpts of HumSet 2.0.0" in published.note_tip
    assert f"it scores {full['recipe_cut_on_all_dev']['Mean']:.3f}" in published.note_tip
    assert "sampled" not in published.note_tip
    assert "0.78 for Sectors" in published.head_tip and "Avg, 0.73" in published.head_tip
    kind_css = {k["kind"]: k["css"] for k in inputs.methods["kinds"]}
    table = rp.results_table(rows, rp.HUMSET_V2, "excerpt", kind_css, published)
    assert table.count(f"<small>paper {printed:.2f}<button type=button class=info") == 1


def test_results_table_needs_what_the_paper_printed_for_a_macro_f1_column() -> None:
    rows = [row(paper={"Mean": 0.5})]
    with pytest.raises(ValueError, match="needs what the paper printed"):
        rp.results_table(rows, "humset", "excerpt", {"Trained": "t"})
    with pytest.raises(ValueError, match="no row missing"):
        rp.results_table(rows, "humset", "excerpt", {"Trained": "t"}, published_stub("missing"))


def test_the_page_cites_each_benchmark_paper_where_it_names_it() -> None:
    page = re.sub(r"<a [^>]*>|</a>", "", re.sub(r"\s+", " ", build_page(REPO)))
    assert "published BERT benchmark (Chalkidis et al., 2021)" in page
    assert "the HumBERT paper (Tamagnone et al., 2023) reports" in page
    assert "arXiv 2305.16756" not in page
    assert "{cite:" not in page


def test_check_v2_macro_lead_raises_unless_the_combined_method_leads_by_macro_f1() -> None:
    xr = row(key="xr", name="XR-Transformer", scores={"exact": {"f1": 0.72}}, paper={"Mean": 0.51})
    combined = row(key="c", kind="Combined", scores={"exact": {"f1": 0.71}}, paper={"Mean": 0.66})
    check_v2_macro_lead([xr, combined])
    with pytest.raises(ValueError, match="leads by macro F1"):
        check_v2_macro_lead([xr, replace(combined, paper={"Mean": 0.5})])
    with pytest.raises(ValueError, match="xr has no macro F1"):
        check_v2_macro_lead([replace(xr, paper=None), combined])


def test_the_v2_table_has_no_note_repeating_its_scores(inputs: rp.Inputs) -> None:
    page = build_page(REPO)
    assert (
        "has the top score"
        not in page[page.index("Humanitarian excerpts (HumSet 2.0.0)") :].split("<section")[0]
    )
    assert "On that average the combined method scores" not in page


def published_stub(row_key: str) -> Published:
    return Published("Per label.", row_key, "paper 0.65", "Why.", "Tier one.", "paper 0.73", "Avg why.")


def test_the_papers_avg_column_is_the_tier_one_mean_with_the_printed_avg_on_the_recipe_row(
    inputs: rp.Inputs,
) -> None:
    printed = inputs.results["12_published_benchmarks"]["printed"]["humset"]["HumBERT, best"]
    rows = rp.method_rows(inputs, rp.HUMSET_V2)
    published = published_measure(inputs, rows)
    assert published.avg_note == f"paper {rp.tier_one_average(printed):.2f}" == "paper 0.73"
    assert "prints as its Avg: 0.73 on all 12,896 of HumSet 2.0.0's test excerpts" in published.avg_tip
    assert "0.78 for Sectors, 0.70 for Pillars 1D and 0.71 for Pillars 2D" in published.avg_tip
    full = inputs.results["17_humset_full_test"][rp.HUMSET_V2]
    recipe = full["paper"]["ranked"]["humbert_recipe"]
    assert f"Here the recipe scores {rp.tier_one_average(recipe):.3f}" in published.avg_note_tip
    assert f"it scores {rp.tier_one_average(full['recipe_cut_on_all_dev']):.3f}" in published.avg_note_tip
    kind_css = {k["kind"]: k["css"] for k in inputs.methods["kinds"]}
    table = rp.results_table(rows, rp.HUMSET_V2, "excerpt", kind_css, published)
    assert table.count("<small>paper 0.73<button type=button class=info") == 1
    for found in rows:
        assert found.paper is not None
        avg = rp.tier_one_average(found.paper)
        assert f'<td class=num data-v="{found.paper["Mean"]}">{found.paper["Mean"]:.2f}' in table
        assert f'<td class=num data-v="{avg}">{avg:.2f}' in table


def test_paper_cells_put_the_notes_on_the_recipe_row_only() -> None:
    paper = {"Mean": 0.5, "Sectors": 1.0, "Pillars 1D": 0.5, "Pillars 2D": 0.75}
    macro, avg = rp._paper_cells("k", paper, published_stub("k"))
    assert macro[0].startswith('<td class=num data-v="0.5">0.50<small>paper 0.65<button') and len(macro) == 1
    assert avg[0].startswith('<td class=num data-v="0.75">0.75<small>paper 0.73<button') and len(avg) == 1
    macro, avg = rp._paper_cells("other", paper, published_stub("k"))
    assert "<small>" not in "".join(macro + avg)
    assert rp._paper_cells("k", None, None) == ([], [])


def test_the_papers_avg_column_comes_before_exact_labels(inputs: rp.Inputs) -> None:
    rows = rp.method_rows(inputs, rp.HUMSET_V2)
    kind_css = {k["kind"]: k["css"] for k in inputs.methods["kinds"]}
    table = rp.results_table(rows, rp.HUMSET_V2, "excerpt", kind_css, published_measure(inputs, rows))
    head = table[: table.index("</thead>")]
    assert head.index("sectors and pillars") < head.index(">Exact labels<") < head.index(">Macro F1<")
    top = rows[0]
    assert top.paper is not None
    first = table[table.index("<tbody>") : table.index("</tr>", table.index("<tbody>"))]
    avg = f'data-v="{rp.tier_one_average(top.paper)}"'
    assert first.index(avg) < first.index("class=barcell") < first.index(f'data-v="{top.paper["Mean"]}"')


def test_as_share_turns_a_percentage_into_a_share_from_zero_to_one() -> None:
    assert as_share(64.0) == pytest.approx(0.64) and as_share(100.0) == 1.0


def test_every_score_on_the_page_is_a_decimal_from_zero_to_one(inputs: rp.Inputs) -> None:
    page = build_page(REPO)
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style)\b.*?</\1>", "", page, flags=re.S))
    law = inputs.results["11_second_attempt"]["eurlex"]["scores"]["all"] / 100
    like_for_like = inputs.results["11_second_attempt"]["eurlex_train_only"]["scores"]["all"] / 100
    bert = inputs.results["12_published_benchmarks"]["eurlex_mrp"]["BERT (printed)"]["all"] / 100
    assert re.search(rf"{like_for_like:.3f}\s+vs\s+{bert:.3f}", text)
    card = " ".join(text[text.index(f"{like_for_like:.3f}") :].split())[:400]
    assert "trained on the training laws only, as BERT was" in card and f"it scores {law:.3f}" in card
    laws = rp.method_rows(inputs, "eurlex")
    best, llm = rp.leaders(laws).best.f1, rp.leaders(laws).llm.f1
    assert re.search(rf"{llm:.3f}\s+vs\s+{best:.3f}", text)
    assert f"<b>{law:.3f}</b>" in page and f"<b>{bert:.3f}</b>" in page
    assert "1.00 is perfect. The MultiEURLEX paper prints it as a percentage: 48.4 there is 0.484" in text
    assert " points" not in text
    benchmark = text[
        text.index("Against the published benchmarks") : text.index("Humanitarian excerpts: mean")
    ]
    assert not re.findall(r"\b\d{2}\.\d\b", benchmark.replace("48.4 there", ""))


def test_every_headline_comparison_says_what_it_is_compared_with(inputs: rp.Inputs) -> None:
    laws, excerpts = rp.method_rows(inputs, "eurlex"), rp.method_rows(inputs, rp.HUMSET_V2)
    cards = header(inputs, laws, excerpts)
    compared = re.findall(r"<span class=vs>vs <b>[0-9.]+</b> ([^<]+)</span>", cards)
    assert compared == [
        "published BERT benchmark",
        "published HumBERT benchmark",
        "our combined method",
    ]
    assert cards.count("<strong") == 3 and "Every score runs from 0 to 1, and higher is better." in cards
    rerun = rp.humbert_rerun(inputs.results)
    assert f"{rp.tier_one_average(rerun['combined_v2']):.3f} against 0.73." in " ".join(cards.split())


def test_the_opening_and_findings_name_who_leads_humset_by_exact_labels(inputs: rp.Inputs) -> None:
    excerpts = rp.method_rows(inputs, rp.HUMSET_V2)
    top = excerpts[0]
    combined = next(r for r in excerpts if r.kind == "Combined")
    assert top.kind != "Combined", "the results changed: rewrite this test for a combined lead"
    clause = f"though {top.name} edges it by exact labels, {top.f1:.3f} to {combined.f1:.3f}"
    assert exact_lead(excerpts) == clause
    assert exact_lead([combined, top]) == "and by exact labels"
    assert exact_lead(excerpts, scores=False) == f"though {top.name} edges it by exact labels"
    page = " ".join(build_page(REPO).split())
    assert page.count(clause) == 1
    lede = page[page.index("<p class=lede>") : page.index("</p>", page.index("<p class=lede>"))]
    assert f"though {top.name} edges it by exact labels." in lede
    assert not re.search(r"\d\.\d{3}", lede)
    assert (
        "several trained models each suggest labels, and a scorer trained on documents they never saw" in page
    )


def test_the_findings_name_the_fine_tuned_models_the_best_llm_method_beats(inputs: rp.Inputs) -> None:
    excerpts = rp.method_rows(inputs, rp.HUMSET_V2)
    llm = rp.leaders(excerpts).llm
    behind = rp.fine_tuned_behind(excerpts, llm)
    assert behind and all(r.kind == "Fine-tuned" and r.f1 < llm.f1 for r in behind)
    page = " ".join(build_page(REPO).split())
    for row in behind:
        assert f"{rp.plain_case(row.name)} ({row.f1:.3f})" in page
    laws = rp.method_rows(inputs, "eurlex")
    lowered = [
        replace(r, scores={**r.scores, "exact": {"f1": 0.0}}) if r.kind == "Trained" else r for r in excerpts
    ]
    lowered.sort(key=lambda r: -r.f1)
    with pytest.raises(ValueError, match="every trained model without an LLM"):
        rp.check_findings(laws, lowered, inputs.results)


def test_the_humset_1_line_gives_its_scores_and_raises_if_the_combined_method_did_not_lead(
    inputs: rp.Inputs,
) -> None:
    v1 = rp.method_rows(inputs, "humset")
    line = first_version_line(inputs, v1)
    rerun = rp.humbert_rerun(inputs.results)
    assert (
        f"{v1[0].f1:.3f} against" in line
        and f"{rerun['combined']['Mean']:.3f} to {rerun['ours']['Mean']:.3f}" in line
    )
    with pytest.raises(ValueError, match=re.escape("on HumSet 1.0.0")):
        first_version_line(inputs, v1[1:])


def test_each_dataset_section_links_its_sources_and_says_who_labelled_it() -> None:
    laws = eurlex_source("(Chalkidis et al., 2021)")
    assert 'href="https://huggingface.co/datasets/coastalcph/multi_eurlex"' in laws
    assert 'href="https://op.europa.eu/en/web/eu-vocabularies/eurovoc"' in laws
    assert "The EU Publications Office tagged each law" in " ".join(laws.split())
    excerpts = " ".join(humset_source("(Fekih et al., 2022)").split())
    assert 'href="https://huggingface.co/datasets/nlp-thedeep/humset"' in excerpts
    assert 'href="https://blog.thedeep.io/humset/"' in excerpts
    assert "it is not a standard published on its own" in excerpts
    assert (
        '(Chalkidis et al., 2021; <a href="https://huggingface.co/datasets/coastalcph/multi_eurlex"' in laws
    )
    assert '>Dataset</a>; <a href="https://blog.thedeep.io/humset/"' in excerpts
    assert "get the dataset" not in laws + excerpts
    assert with_links("(A, 2020)", "x", "y") == "(A, 2020; x; y)"
    with pytest.raises(ValueError, match="bracketed"):
        with_links("A, 2020", "x")
    page = " ".join(build_page(REPO).split())
    assert page.count("<strong>Where the labels come from.</strong>") == 2
