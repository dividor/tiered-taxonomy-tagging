"""Assemble the results report page from its sections. The numbers, rows and checks come from `report`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tiered_taxonomy import benchmarks as bm
from tiered_taxonomy import report_browser as rb
from tiered_taxonomy.report import (
    ASSETS,
    HUMSET_GROUPS,
    HUMSET_V2,
    TIER_ONE_GROUPS,
    Inputs,
    Row,
    TierMix,
    best_scores,
    card,
    check_findings,
    combined_name,
    esc,
    fine_tuned_behind,
    groups_clear_of_print,
    humbert_rerun,
    info,
    interval,
    leaders,
    load_inputs,
    method_name,
    method_rows,
    near_miss_paths,
    plain_case,
    ranked_bars,
    ref_html,
    results_table,
    tier_mix,
    tier_one_average,
    tiers_not_led_by_combined,
)
from tiered_taxonomy.report_papers import Published, cite

FONTS = (
    "https://fonts.googleapis.com/css2?family=Poppins:wght@600;700"
    "&family=Open+Sans:ital,wght@0,400;0,600;1,400&display=swap"
)

CARD_DIALOG = """<dialog id=card aria-labelledby=card-title>
  <div class=card>
    <button type=button class=close aria-label="Close" id=card-close>&times;</button>
    <header>
      <span id=card-kind class=chip></span>
      <h2 id=card-title></h2>
      <p id=card-summary class=summary></p>
    </header>
    <ol id=card-steps class=steps aria-label="How it works"></ol>
    <div id=card-combines-wrap class=combines><h3>Models it combines</h3><ul id=card-combines></ul></div>
    <div id=card-clues-wrap class=clues><h3>The clues it weighs</h3><p id=card-clues-intro></p>
      <dl id=card-clues></dl><p id=card-scorer class=scorer></p></div>
    <div><h3>F1 score here <span class=hint>bar: this method · tick: best in the table</span></h3>
      <div id=card-scores class=scores></div>
      <p class=scorenote>Broad to narrow. A tier counts a label in the right branch; exact needs the very
      label a person chose.</p></div>
    <dl id=card-tiles class=tiles></dl>
    <p id=card-settings class=settings></p>
    <div id=card-differs-wrap class=differs><h3>Ours against the source</h3><p id=card-differs></p></div>
    <div id=card-refs-wrap><h3>Sources</h3><ul id=card-refs class=refs></ul></div>
    <p class=nb>Notebook <span id=card-nb></span></p>
  </div>
</dialog>
"""

# The page's own script: it opens a method's card and sorts the tables, beside the markup it drives.
SCRIPT = """const CARDS = JSON.parse(document.getElementById("cards-data").textContent);
const dlg = document.getElementById("card");
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
function showCard(key) {
  const c = CARDS[key];
  $("card-title").textContent = c.title;
  $("card-summary").textContent = c.summary;
  $("card-kind").textContent = c.kind;
  $("card-kind").className = "chip " + c.kindClass;
  $("card-steps").replaceChildren(...c.steps.flatMap((step, i) => {
    const last = i === c.steps.length - 1;
    const box = el("li", i === 0 ? "end" : last ? "end out" : "", step);
    if (i === 0) return [box];
    const arrow = el("li", "arrow", "→");
    arrow.setAttribute("aria-hidden", "true");
    return [arrow, box];
  }));
  $("card-steps").style.setProperty("--cols", c.steps.map(() => "minmax(0, 1fr)").join(" auto "));
  $("card-combines").replaceChildren(...c.combines.map((name) => el("li", "chip k-none", name)));
  $("card-combines-wrap").hidden = !c.combines.length;
  $("card-clues-intro").textContent = c.cluesIntro;
  $("card-clues").replaceChildren(...c.clues.flatMap(([k, v]) => [el("dt", "", k), el("dd", "", v)]));
  $("card-scorer").textContent = c.scorer;
  $("card-clues-wrap").hidden = !c.clues.length;
  $("card-scores").replaceChildren(...c.scores.map(([label, value, best]) => {
    const line = el("div", "score");
    const track = el("span", "track");
    const fill = el("i"); fill.style.width = (100 * value) + "%";
    const tick = el("b"); tick.style.left = (100 * best) + "%"; tick.title = "best: " + best.toFixed(2);
    track.append(fill, tick);
    line.append(el("span", "lbl", label), track, el("span", "val", value.toFixed(2)));
    return line;
  }));
  $("card-tiles").replaceChildren(...c.tiles.map(([k, v]) => {
    const tile = el("div");
    tile.append(el("dt", "", k), el("dd", "", v));
    return tile;
  }));
  $("card-settings").textContent = c.settings; $("card-settings").hidden = !c.settings;
  $("card-differs").textContent = c.differs; $("card-differs-wrap").hidden = !c.differs;
  $("card-refs").innerHTML = c.refs; $("card-refs-wrap").hidden = !c.refs;
  $("card-nb").textContent = c.notebook;
  dlg.showModal();
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("button.tech");
  if (b) showCard(b.dataset.key);
});
$("card-close").addEventListener("click", () => dlg.close());
dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
for (const table of document.querySelectorAll("table.sortable")) {
  const heads = [...table.tHead.querySelectorAll("th[data-sort]")];
  heads.forEach((th) => {
    th.tabIndex = 0;
    const sort = () => {
      const col = Number(th.dataset.col);
      const sorted = th.classList.contains("sorted");
      const asc = sorted ? !th.classList.contains("asc") : th.dataset.sort === "text";
      heads.forEach((h) => h.classList.remove("sorted", "asc"));
      th.classList.add("sorted"); if (asc) th.classList.add("asc");
      const rows = [...table.tBodies[0].rows];
      const val = (r) => r.cells[col].dataset.v ?? r.cells[col].textContent;
      rows.sort((a, b) => {
        const x = val(a), y = val(b);
        const number = th.dataset.sort === "num";
        const c = number ? (parseFloat(x) || 0) - (parseFloat(y) || 0) : x.localeCompare(y);
        return asc ? c : -c;
      });
      table.tBodies[0].append(...rows);
    };
    th.addEventListener("click", sort);
    th.addEventListener("keydown", (e) => {
      if (e.target !== th) return;
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); }
    });
  });
}
// One tooltip for every heading's i button, fixed to the window so a table's scroll frame cannot clip it.
const tip = el("div", "tip");
tip.id = "tip"; tip.setAttribute("role", "tooltip"); tip.hidden = true;
document.body.append(tip);
let tipFor = null;
function showTip(button) {
  if (tipFor && tipFor !== button) tipFor.removeAttribute("aria-describedby");
  tip.textContent = button.dataset.tip;
  tip.hidden = false;
  tipFor = button;
  button.setAttribute("aria-describedby", "tip");
  const box = button.getBoundingClientRect();
  const width = tip.offsetWidth, height = tip.offsetHeight;
  const left = Math.min(Math.max(16, box.left + box.width / 2 - width / 2), innerWidth - 16 - width);
  const fitsBelow = box.bottom + 8 + height <= innerHeight - 8;
  tip.style.left = left + "px";
  tip.style.top = (fitsBelow ? box.bottom + 8 : Math.max(8, box.top - 8 - height)) + "px";
}
function hideTip() {
  if (tipFor) tipFor.removeAttribute("aria-describedby");
  tip.hidden = true;
  tipFor = null;
}
const infoButton = (e) => e.target.closest && e.target.closest("button.info");
document.addEventListener("pointerover", (e) => {
  if (e.pointerType === "mouse" && infoButton(e)) showTip(infoButton(e));
});
document.addEventListener("pointerout", (e) => {
  const button = infoButton(e);
  const leaving = button && !button.contains(e.relatedTarget) && document.activeElement !== button;
  if (e.pointerType === "mouse" && leaving) hideTip();
});
document.addEventListener("focusin", (e) => { if (infoButton(e)) showTip(infoButton(e)); });
document.addEventListener("focusout", (e) => { if (infoButton(e)) hideTip(); });
document.addEventListener("click", (e) => {
  const button = infoButton(e);
  if (button) { e.stopPropagation(); showTip(button); } else if (tipFor) hideTip();
}, true);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });
addEventListener("scroll", () => { if (tipFor) hideTip(); }, true);
addEventListener("resize", () => { if (tipFor) hideTip(); });
"""


RECIPE = "humbert_recipe"


def paper_cites(inputs: Inputs) -> dict[str, str]:
    """Return the bracketed citations of the papers whose benchmarks the report sets ours against."""
    refs = inputs.methods["references"]
    return {key: cite(refs[key]) for key in ("humbert", "humset", "multieurlex")}


def published_measure(inputs: Inputs, rows: Sequence[Row]) -> Published:
    """Return what the HumBERT paper printed by macro F1, for the HumSet 2.0.0 table: the heading's tooltip,
    and the note on the recipe's row saying what the paper reported and how our measurement differs."""
    r = inputs.results
    rerun, humbert = humbert_rerun(r), paper_cites(inputs)["humbert"]
    printed = rerun["printed"]
    test_size = r["13_humbert_reproduction"]["2.0.0"]["run"]["sizes"]["test"]
    recipe = next(row for row in rows if row.key == RECIPE)
    if recipe.paper is None:
        raise ValueError("the recipe's row has no macro F1 to set beside the paper's")
    groups = _and([f"{printed[g]:.2f} for {g}" for g in HUMSET_GROUPS])
    head_tip = (
        "F1 worked out for each label and then averaged, so a rare label counts as much as a common one: "
        "first within each of HumSet's five published groups of labels, then over the groups. This is the "
        f"measure the HumBERT paper {humbert} reports. On all {test_size:,} of HumSet 2.0.0's "
        f"test excerpts it prints {groups}. Their mean, {printed['Mean']:.3f}, is worked out by us: the "
        f"paper prints only its own Avg, {tier_one_average(printed):.2f}, the mean of the three tier-1 "
        "groups, which the first score column gives for every row. Trained models get a cut-off for each "
        "label, chosen on development excerpts; LLM methods, which give no scores, count every label they "
        "return."
    )
    avg = tier_one_average(printed)
    avg_tip = (
        "Macro F1 averaged over Sectors, Pillars 1D and Pillars 2D only, leaving out both subpillar groups. "
        f"This is the number the HumBERT paper {humbert} prints as its Avg: {avg:.2f} on all "
        f"{test_size:,} of HumSet 2.0.0's test excerpts, the mean of its "
        f"{_and([f'{printed[g]:.2f} for {g}' for g in TIER_ONE_GROUPS])}. Read a row here against that "
        "printed figure directly. Cut-offs as for the Macro F1 column."
    )
    exact = (
        " Its exact-labels score is another measure again: every label pooled, so common labels weigh most, "
        "with one cut-off rule for all labels."
    )
    full = r["17_humset_full_test"][HUMSET_V2]
    sizes, all_dev = full["sizes"], full["recipe_cut_on_all_dev"]
    note_tip = (
        f"The HumBERT paper {humbert} reports {printed['Mean']:.3f} by this measure, the mean of its "
        f"five printed groups, on these same {sizes['test']:,} test excerpts of HumSet 2.0.0. Here the "
        f"recipe scores {recipe.paper['Mean']:.3f}, with each label's cut-off chosen on "
        f"{sizes['tune']:,} development excerpts, as for every method in this table, because the "
        "combined method trains on the others. With the cut-offs chosen the same way on all "
        f"{sizes['dev']:,} development excerpts it scores {all_dev['Mean']:.3f}; rerun with the authors' "
        "own code, whose trainer chooses its cut-offs its own way, it scores "
        f"{rerun['paper']['Mean']:.3f} (notebooks 13 and 17).{exact}"
    )
    avg_note_tip = (
        f"The HumBERT paper {humbert} prints {avg:.2f} as its Avg on these same {sizes['test']:,} test "
        f"excerpts of HumSet 2.0.0. Here the recipe scores {tier_one_average(recipe.paper):.3f}, with "
        f"each label's cut-off chosen on {sizes['tune']:,} development excerpts; with the cut-offs "
        f"chosen on all {sizes['dev']:,} it scores {tier_one_average(all_dev):.3f}, and rerun with the "
        f"authors' own code {tier_one_average(rerun['paper']):.3f} (notebooks 13 and 17)."
    )
    return Published(
        head_tip,
        RECIPE,
        f"paper {printed['Mean']:.2f}",
        note_tip,
        avg_tip,
        f"paper {avg:.2f}",
        avg_note_tip,
    )


def as_share(points: float) -> float:
    """Return a mean R-Precision saved as a percentage, as the MultiEURLEX paper prints it, as a share from 0
    to 1, so that every score on the page reads the same way."""
    return points / 100


def exact_lead(excerpts: Sequence[Row]) -> str:
    """Return the clause on who leads HumSet 2.0.0 by exact labels, naming the leader when it is not the
    combined method."""
    top = excerpts[0]
    if top.kind == "Combined":
        return "and by exact labels"
    combined = next(r for r in excerpts if r.kind == "Combined")
    return f"though {esc(top.name)} edges it by exact labels, {top.f1:.3f} to {combined.f1:.3f}"


def header(inputs: Inputs, laws: Sequence[Row], excerpts: Sequence[Row]) -> str:
    """Return the title, the one-paragraph answer and the three headline numbers. `excerpts` are the
    HumSet 2.0.0 rows."""
    r = inputs.results
    count = len({row.method for row in [*laws, *excerpts]})
    law_score = as_share(r["11_second_attempt"]["eurlex"]["scores"]["all"])
    bert = as_share(r["12_published_benchmarks"]["eurlex_mrp"]["BERT (printed)"]["all"])
    rerun = humbert_rerun(r)
    v2, humbert = rerun["combined_v2"]["Mean"], rerun["printed"]["Mean"]
    e = leaders(laws)
    c = paper_cites(inputs)
    return f"""<header class=col>
  <span class=eyebrow>Tiered taxonomy experiments · results to 14 November 2026</span>
  <h1>Which taxonomy taggers work best</h1>
  <p class=lede>We tested {count} ways of tagging documents with labels from a tiered taxonomy, on EU laws
  and on humanitarian excerpts. The most accurate was our combined method: several trained models each
  suggest labels, and a scorer trained on documents they never saw decides which to keep. Which models it
  draws on is chosen for each dataset. It led on the EU laws, and on HumSet 2.0.0 by the HumBERT paper's
  measure, {exact_lead(excerpts)}. Methods that prompt an LLM came well behind, though on the EU laws they
  found more of the rare labels.</p>
</header>
<section class=col aria-label="The short answer"><div class=answer>
  <div><strong class=pass>{law_score:.3f}</strong>
    <span class=vs>vs <b>{bert:.3f}</b> published BERT benchmark</span>
    <span class=eyebrow>Our combined method · EU laws</span><p>Mean R-Precision on the full test set. BERT's
    score is from the MultiEURLEX paper {c["multieurlex"]}.</p></div>
  <div><strong class=pass>{v2:.3f}</strong>
    <span class=vs>vs <b>{humbert:.3f}</b> published HumBERT benchmark</span>
    <span class=eyebrow>Our combined method · HumSet 2.0.0</span><p>Mean macro F1 over five label groups on
    the full test set; HumBERT's is the mean of the five in its paper {c["humbert"]}. On that paper's own
    average it is {tier_one_average(rerun["combined_v2"]):.3f} against
    {tier_one_average(rerun["printed"]):.2f}.</p></div>
  <div class=llm><strong>{e.llm.f1:.3f}</strong>
    <span class=vs>vs <b>{e.best.f1:.3f}</b> our combined method</span>
    <span class=eyebrow>Best LLM method · EU laws</span><p>{e.llm.name}, which prompts an LLM, by
    exact-label F1 on the EU laws.</p></div>
</div>
<p class="note answer-note">Every score runs from 0 to 1, and higher is better.</p></section>
"""


def near_miss(
    mix: TierMix,
    tree: Mapping[str, Any],
    paper: str,
    person: str = "labelling",
    method: str = "packaging",
) -> str:
    """Return a worked example of a near miss, and where the assigned labels sit, to explain why the exact
    score is lower than the tier scores. The example's path is read from EuroVoc's tree in `tree`, and `paper`
    is the citation of the MultiEURLEX paper."""
    path = near_miss_paths(tree, person, method)
    *above, (tier, _) = path
    parent = above[-1][1]
    boxes = "".join(
        f"<li><span class=t>Tier {t}</span><span class=lab>{esc(label)}</span>"
        "<span class=ok>both match</span></li>"
        for t, label in above
    )
    shared = ", ".join(str(t) for t, _ in above[:-1]) + f" and {above[-1][0]}"
    return f"""<figure class=nearmiss>
    <figcaption><h3>Why the exact score is lower than the tier scores</h3>
    <p>Every box below is a label in our copy of EuroVoc, which arranges its labels in {mix.law_tiers}
    tiers. It is not quite a tree: {mix.law_several_parents:,} labels have more than one parent. Tiers 1
    and 2 only group the labels below them. Laws are tagged from tier 3 down {paper}, and none of these test
    laws has a tier 1 or 2 label.</p>
    <p>Say a person gave a law the tier {tier} label <q>{esc(person)}</q>, and a method chose
    <q>{esc(method)}</q>, another tier {tier} label. Both sit under the same tier {tier - 1} label,
    <q>{esc(parent)}</q>.</p>
    </figcaption>
    <ol class=path>{boxes}<li class=leaf><span class=t>Tier {tier}</span><span class=lab>
      <span class=who>{esc(person)} <small>person</small></span>
      <span class=who>{esc(method)} <small>method</small></span></span>
      <span class=no>no match</span></li></ol>
    <p>A tier score replaces each label with the label or labels above it at that tier, through every parent,
    then compares those. At tiers {shared} both labels become the same label, so this near miss counts as
    correct. The exact score compares the labels themselves, at whatever tier they sit, here tier {tier}, so
    it scores nothing.</p>
    <ul>
      <li><strong>EU laws:</strong> our copy of EuroVoc has {mix.law_nodes:,} labels, only
      {mix.law_top_three:,} of them in tiers 1 to 3; the MultiEURLEX paper counts {bm.MULTI_EURLEX_LABELS:,}
      in its version. Of the labels people assigned to these test laws,
      {mix.law_below_three:.0%} sit deeper than tier 3, so every method's exact score is below all three of
      its tier scores.</li>
      <li><strong>Humanitarian excerpts:</strong> the taxonomy has only two tiers, and
      {mix.excerpt_tier_one:.0%} of the labels people assigned are tier 1 sectors and pillars with no
      subpillar. The exact score mixes the two tiers, so for {mix.excerpts_between} of the
      {mix.excerpt_rows} methods it falls between the tier 1 and tier 2 scores.</li>
    </ul>
  </figure>"""


def first_version_line(inputs: Inputs, v1_rows: Sequence[Row]) -> str:
    """Return the one line on HumSet 1.0.0, where every method was first run. Raises unless the combined
    method had the top exact score there and beat HumBERT's recipe, rerun, by macro F1, as the line says."""
    top = v1_rows[0]
    if top.kind != "Combined":
        raise ValueError(f"findings no longer hold: on HumSet 1.0.0 {top.key} has the top exact score")
    runner = next(r for r in v1_rows if r.kind != "Combined")
    rerun = humbert_rerun(inputs.results)
    if rerun["combined"]["Mean"] <= rerun["ours"]["Mean"]:
        raise ValueError(
            "findings no longer hold: on HumSet 1.0.0 the recipe rerun beats the combined method"
        )
    size = next(s["test"] for s in inputs.results["01_test_sets"]["sizes"] if s["test set"] == "humset")
    sample = top.scores["exact"]["items"]
    return (
        f"<p class=note>We first ran every method on {sample:,} sampled test excerpts of HumSet 1.0.0, an "
        "earlier version with different splits. There the combined method had the top score "
        f"by exact labels, {top.f1:.3f} against {esc(plain_case(runner.name))}'s {runner.f1:.3f}, and on "
        f"all {size:,} of that version's test excerpts it beat HumBERT's recipe, rerun by us, by macro F1, "
        f"{rerun['combined']['Mean']:.3f} to {rerun['ours']['Mean']:.3f}.</p>"
    )


def results(
    inputs: Inputs,
    mix: TierMix,
    laws_table: str,
    v2_table: str,
    v2_rows: Sequence[Row],
    v1_rows: Sequence[Row],
) -> str:
    """Return the key to the measures and kinds, and the two results tables, with which methods lead the
    HumSet one and a line on HumSet 1.0.0."""
    kinds = "".join(
        f'<dt><span class="chip {k["css"]}">{esc(k["kind"])}</span></dt><dd>{esc(k["meaning"])}</dd>'
        for k in inputs.methods["kinds"]
    )
    models = inputs.config["models"]
    cutoff = inputs.results["label_browser"]["eurlex"]["rare_at_most"]
    v2 = inputs.results["15_humset_v2"]
    v2_test = inputs.results["17_humset_full_test"][HUMSET_V2]["sizes"]["test"]
    c = paper_cites(inputs)
    return f"""<section>
  <div class=col>
    <span class=eyebrow>Every method on the same documents</span>
    <h2>The results</h2>
    <p>Every method was run on the same 600 EU laws and on all {v2_test:,} test excerpts of HumSet 2.0.0,
    the HumBERT paper's test set, and scored the same way. None
    of them saw these documents during training or tuning. Open a method's card, by its name, to read what it
    does and the papers behind it. Sort the table by any column heading.</p>
  </div>
  <div class=key>
    <div><h3>What the numbers mean</h3><dl>
      <dt>F1 score</dt><dd>Balances finding the correct labels against adding wrong ones. 1.00 is
      perfect.</dd>
      <dt>Exact labels</dt><dd>Credit only for the very label a person assigned. The main measure, and the
      hardest. On HumSet, a pillar given together with one of its own subpillars is dropped first, since the
      subpillar already names it; a pillar given alone stays.</dd>
      <dt>Tier 1, 2, 3</dt><dd>Credit for landing in the right branch at that level of the taxonomy, even
      with the wrong label inside it. Tier 1 is the broadest, so the easiest.</dd>
      <dt>Macro F1</dt><dd>HumSet only. The measure the HumBERT paper {c["humbert"]} reports: F1 worked out
      for each
      label, so rare labels count as much as common ones, averaged in each of five groups of labels and then
      over the groups. Trained models get a threshold per label, chosen on development excerpts; LLM methods,
      which give labels without scores, count every label they return.</dd>
      <dt>Rare labels</dt><dd>Recall, from 0 to 1: the share each method found of the correct labels in the
      least-used quarter of labels in training, and of those never given in training.</dd>
      <dt>Cost</dt><dd>LLM methods: what the calls cost per 1,000 documents. Trained methods: one-off GPU
      time, or
      free on a laptop.</dd>
    </dl></div>
    <div><h3>Kinds of method</h3><dl>{kinds}</dl></div>
  </div>
  {near_miss(mix, inputs.results["label_browser"]["eurlex"]["tree"], c["multieurlex"])}
  <h3>EU laws (MultiEURLEX): 600 test laws, EuroVoc's top three tiers</h3>
  {rb.data_links("eurlex", inputs.results["label_browser"])}
  <p class=note>The laws are split by date, so the test laws are newer than the training laws. Rows marked
  "with
  newer laws" were also trained on newer laws from the development split, which brings them closer in time
  to the
  test laws. The published BERT benchmark {c["multieurlex"]} used training laws only. A rare label here was
  given to at most
  {cutoff:,} of the training laws.</p>
  <p class=note>Tiers 1 to 3 here come from moving each label up our copy of EuroVoc, which is newer than the
  dataset (notebook 01). MultiEURLEX publishes its own labels for its levels 1 to 3 {c["multieurlex"]}, and
  ours match them for {mix.level_agreement[0]:.1%} of test laws at tier 1, {mix.level_agreement[1]:.1%} at
  tier 2 and {mix.level_agreement[2]:.1%} at tier 3. So these tier scores cannot be set beside the paper's
  results by level.</p>
  {laws_table}
  {v2_section(v2, v2_test, v2_table + v2_leads(v2_rows), c["humbert"], inputs.results["label_browser"])}
  {first_version_line(inputs, v1_rows)}
  <p class=note>LLM methods used <code>{esc(models["main"]["model"])}</code> and
  <code>{esc(models["small"]["model"])}</code> through Hugging Face Inference Providers, except
  LLM-Ensemble, which
  used the five open models listed in <code>configs/experiment.yaml</code>. GPU training ran on Hugging Face
  Jobs at
  November 2026 prices.</p>
</section>
"""


def v2_leads(rows: Sequence[Row]) -> str:
    """Return a note on which method leads the HumSet 2.0.0 table by each measure. Rows are best first by
    exact labels. When something other than the combined method leads by exact labels, it gives both scores
    and says that the gap has not been tested against chance. A row without a macro F1 raises."""

    def macro(row: Row) -> float:
        if row.paper is None:
            raise ValueError(f"on HumSet 2.0.0 {row.key} has no macro F1")
        return row.paper["Mean"]

    combined = next(row for row in rows if row.kind == "Combined")
    top = rows[0]
    by_macro = max(rows, key=macro)
    if top is combined and by_macro is combined:
        return "<p class=note>The combined method has the top score by exact labels and by macro F1.</p>"
    if by_macro is not combined:
        raise ValueError(f"on HumSet 2.0.0 {by_macro.key} leads by macro F1, not the combined method")
    return (
        f"<p class=note>By exact labels the {esc(plain_case(top.name))} has the top score, {top.f1:.3f}, "
        f"against the combined method's {combined.f1:.3f}; by macro F1 the combined method leads, "
        f"{macro(combined):.3f} against {macro(top):.3f}. We have not tested whether the "
        "exact-labels gap is larger than chance.</p>"
    )


def _listed(labels: Sequence[str]) -> str:
    quoted = [f"<q>{esc(label)}</q>" for label in labels]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def v2_section(v2: dict[str, Any], test: int, table: str, humbert: str, browser: Mapping[str, Any]) -> str:
    """Return the HumSet table: every method trained on HumSet 2.0.0 and scored on all `test` of its test
    excerpts. `humbert` is the HumBERT paper's citation."""
    labels = v2["labels"]
    return f"""<h3>Humanitarian excerpts (HumSet 2.0.0): all {test:,} test excerpts, the HumBERT paper's
  test set</h3>
  {rb.data_links("humset", browser)}
  <p class=note>Excerpts in English, French and Spanish. Tier 1 holds the sectors and pillars, and tier 2 the
  subpillars. A rare label here was given to at most {v2["rare_at_most"]:,} of the training excerpts. Every
  label in these test excerpts was given in training, so the never-seen column reads none. Every
  method, LLM methods included, was trained, tuned and run on this version's own splits (notebooks 15 and
  17), so the macro F1 columns score the paper's own test set by its measure {humbert} and can be set beside
  its printed figures; the note on the recipe's row says what the paper reported and how our cut-offs differ
  from its trainer's.</p>
  <details class=more><summary>How the labels differ from HumSet 1.0.0</summary>
  <p class=note>Version 2.0.0 uses {labels["in_use"]} labels. Only it has
  {_listed(labels["only_in_2.0.0"])}; only 1.0.0 has {_listed(labels["only_in_1.0.0"])}. Version 2.0.0
  also tags some excerpts with a pillar but none of its subpillars; that pillar is then a label of its own.
  The label browser and examples are from 1.0.0.</p></details>
  {table}"""


def tier_sentence(laws: Sequence[Row], excerpts: Sequence[Row]) -> str:
    """Return the sentence on which methods top each tier, naming any tier a combined row does not lead."""
    exceptions = [
        f"tier {tier} on the {where}, where the {esc(plain_case(row.name))} is {lead:.3f} ahead"
        for where, rows in (("laws", laws), ("excerpts", excerpts))
        for tier, row, lead in tiers_not_led_by_combined(rows)
    ]
    if not exceptions:
        return "A version of the combined method also has the top score at every tier."
    return (
        "A version of the combined method also has the top score at every tier but "
        + "; ".join(exceptions)
        + "."
    )


def findings(inputs: Inputs, laws: Sequence[Row], excerpts: Sequence[Row]) -> str:
    """Return the findings, with `excerpts` the HumSet 2.0.0 rows. `check_findings` has already confirmed
    every fact they state."""
    r = inputs.results
    e, h = leaders(laws), leaders(excerpts)
    combined = next(x for x in excerpts if x.kind == "Combined")
    if combined.paper is None or h.single.paper is None:
        raise ValueError("on HumSet 2.0.0 the combined method and the best single model need a macro F1")
    macro, single_macro = combined.paper["Mean"], h.single.paper["Mean"]
    behind = fine_tuned_behind(excerpts, h.llm)
    behind_text = _and([f"{esc(plain_case(x.name))} ({x.f1:.3f})" for x in behind])
    pipe = next(x for x in laws if x.key == "pipeline")
    strict = next(x for x in laws if x.key == "walk_strict")
    per_law = r["12_published_benchmarks"]["diagnosis"]["eurlex"]["gold labels per item"]
    ki = r["09_paper_recipes"]
    funnel, evid = r["04_funnel"], r["06_evidence"]["differences"]
    unseen = "unseen"
    return f"""<section class=col>
  <span class=eyebrow>What the results show</span>
  <h2>Findings</h2>
  <ul class=findings>
    <li><strong>The combined method is the most accurate.</strong> On the laws it scores {e.best.f1:.3f} by
    exact labels, against {e.single.f1:.3f} for the best single model, the fine-tuned Legal-BERT. On HumSet
    2.0.0 it leads by the HumBERT paper's macro F1, {macro:.3f}
    against {single_macro:.3f} for {esc(h.single.name)}, the best single model, {exact_lead(excerpts)}.
    {tier_sentence(laws, excerpts)}</li>
    <li><strong>Trained models beat LLM methods, with exceptions on HumSet.</strong> On the laws every
    fine-tuned model scores above every LLM method; the best LLM method, KI-FSPrompt, scores
    {e.llm.f1:.3f}. On HumSet 2.0.0 KI-FSPrompt scores {h.llm.f1:.3f}: below every trained model that has no
    LLM inside it, but above the {behind_text}. KI-FSPrompt also needs labelled documents, because it shows
    them to the LLM as examples.</li>
    <li><strong>LLM methods fall further behind on finer labels.</strong> On the laws, KI-FSPrompt trails
    the best
    single model by {e.single.tier("1") - e.llm.tier("1"):.2f} at tier 1 ({e.llm.tier("1"):.2f} against
    {e.single.tier("1"):.2f}), and by {e.single.tier("3") - e.llm.tier("3"):.2f} at tier 3
    ({e.llm.tier("3"):.2f} against {e.single.tier("3"):.2f}).</li>
    <li><strong>LLM methods find labels the training data lacks.</strong> The LLM keyword pipeline finds
    {pipe.scores["rarity"][unseen]["recall"]:.0%} of the correct labels never seen in training; the combined
    method
    finds {e.best.scores["rarity"][unseen]["recall"]:.0%}. But the pipeline suggests
    {pipe.scores["labels_per_item"]:.0f} labels per law, where a law has about {per_law:.1f}, so most of its
    suggestions are wrong.</li>
    <li><strong>Showing the LLM labelled examples is the best LLM recipe.</strong> KI-FSPrompt beats the
    plain LLM
    keyword pipeline by {interval(ki["eurlex"]["ki_fsprompt"]["minus_pipeline"])} on the laws and by
    {interval(ki["humset"]["ki_fsprompt"]["minus_pipeline"])} on the HumSet 1.0.0 sample, at about half
    the cost.
    LLM-Ensemble,
    with five LLMs, did worse than the plain pipeline and cost about eight times as much.</li>
    <li><strong>Search-then-choose needs a trained model in the shortlist.</strong> On the laws, search
    alone put
    only {funnel["reached"]["eurlex names"]:.0%} of the correct labels in its top 100, so the LLM could not
    choose
    the rest. Adding the trained model's suggestions raised that to
    {funnel["stages"]["hybrid: offered to the model"]:.0%}, and F1 by
    {interval(funnel["differences"]["eurlex"]["hybrid minus cards"])}.</li>
    <li><strong>Walking down the taxonomy one branch at a time fails.</strong> It scores {strict.f1:.2f} on
    the
    laws. Keeping three branches open helps on both datasets. Starting from search results instead of from
    the top
    helps on the laws but hurts on the excerpts.</li>
    <li><strong>Asking the LLM to double-check its labels made things worse.</strong> F1 fell by
    {interval(evid["eurlex"]["second reading minus plain"])} on the laws and by
    {interval(evid["humset"]["second reading minus plain"])} on the HumSet 1.0.0 sample, and the number of
    calls doubled.
    Keeping only labels with a supporting quote also lowered F1.</li>
  </ul>
  <p class=note>Each difference with an interval was tested on the same documents: the 600 laws, or the
  sample of HumSet 1.0.0 where every method was first run. Every one of those intervals excludes zero, so
  none of these differences is likely to be chance.</p>
</section>
"""


Bars = list[tuple[str, str, float, str]]


def benchmark_rows(inputs: Inputs) -> tuple[Bars, Bars]:
    r = inputs.results
    c = paper_cites(inputs)
    mrp = r["12_published_benchmarks"]["eurlex_mrp"]
    first, second = r["10_combined_method"], r["11_second_attempt"]
    newer, only = "With newer laws", "Training laws only"
    laws_in_points = [
        ("Combined method, second attempt", newer, second["eurlex"]["scores"]["all"], "lead"),
        ("Combined method, first attempt", newer, first["eurlex"]["scores"]["all"], "lead2"),
        ("Combined method, second attempt", only, second["eurlex_train_only"]["scores"]["all"], "lead2"),
        ("Combined method, first attempt", only, first["eurlex_train_only"]["scores"]["all"], "lead2"),
        ("Fine-tuned Legal-BERT", newer, first["eurlex"]["best_single"]["all"], ""),
        ("Fine-tuned Legal-BERT", only, first["eurlex_train_only"]["best_single"]["all"], ""),
        ("BERT, published", f"The benchmark to beat {c['multieurlex']}", mrp["BERT (printed)"]["all"], "ref"),
        ("XR-Transformer", only, mrp["xr_transformer"]["all"], ""),
        ("Omikuji", only, mrp["omikuji"]["all"], ""),
        ("Embedding classifier", only, mrp["embedding_classifier"]["all"], ""),
        ("Similar documents", only, mrp["knn"]["all"], ""),
        ("Label search", "No training", mrp["label_search"]["all"], ""),
    ]
    laws = [(name, sub, as_share(score), kind) for name, sub, score, kind in laws_in_points]
    rerun = humbert_rerun(r)
    v2 = r["15_humset_v2"]["full_test"]
    on_v2 = sorted(
        [
            ("HumBERT, published", f"The benchmark to beat {c['humbert']}", rerun["printed"]["Mean"], "ref"),
            ("HumBERT recipe, rerun", "The authors' code and settings", rerun["paper"]["Mean"], ""),
            (
                "Combined method",
                combined_name(inputs, "combined", HUMSET_V2)[1],
                v2["scores"]["Mean"],
                "lead",
            ),
            (
                method_name(inputs.methods, v2["best_single_source"], HUMSET_V2) + ", ours",
                "The best single source on development excerpts",
                v2["best_single"]["Mean"],
                "",
            ),
        ],
        key=lambda row: -row[2],
    )
    return laws, on_v2


def _and(groups: Sequence[str]) -> str:
    return groups[0] if len(groups) == 1 else f"{', '.join(groups[:-1])} and {groups[-1]}"


def _behind(behind: Sequence[str], scores: Mapping[str, float], printed: Mapping[str, float]) -> str:
    """Return a clause naming any group where the combined method is below the printed figure."""
    if not behind:
        return ""
    parts = [f"{g} ({scores[g]:.4f} against {printed[g]:.2f})" for g in behind]
    return f", and is level within rounding in {_and(parts)}"


BENCH_TIPS = {
    "group": "HumSet's labels fall into five published groups: sectors, pillars and subpillars of the 1D "
    "framework, and pillars and subpillars of the 2D framework. Subpillars sit under pillars.",
    "published": "The best HumBERT figures printed in the paper {humbert}, table 5, on all of HumSet "
    "2.0.0's test excerpts, with each label's cut-off chosen on validation. They have two decimals; the "
    "five-group mean is worked out from them, as the paper prints none.",
    "rerun": "HumBERT's recipe, run again by us with the authors' own code and settings, one run, scored "
    "with every label counted once.",
    "combined": "Our combined method, trained and tuned on this version's own splits and scored on its full "
    "test set.",
    "interval": "The range the combined method's score would fall in 95 times out of 100 if the test "
    "excerpts were drawn again: 200 resamples of the test excerpts. Worked out for each group and the "
    "five-group mean.",
}
TIER_ONE_TIP = (
    "The mean of Sectors, Pillars 1D and Pillars 2D only. This is what the HumBERT paper's own Avg column "
    "holds {humbert}, so it leaves out both subpillar groups."
)
ROLLUP_TIPS = {
    "documents": "Long documents cut into 400-word pieces: EU laws, and humanitarian reports rebuilt from "
    "their excerpts.",
    "first": "Label the opening piece only, and give the whole document its labels.",
    "share": "Keep a label if it is on at least {share:.0%} of the pieces, and always on at least one.",
    "any": "Keep every label found in any piece.",
    "certainty": "Add up how sure the model was about a label on each piece, counting doubt against it, and "
    "keep labels with a positive total (Wu et al., 2026).",
    "cost": "What the LLM calls cost for all of these documents.",
}


def benchmarks(inputs: Inputs) -> str:
    """Return the comparison with the published benchmarks on the full test sets."""
    r = inputs.results
    laws, on_v2 = benchmark_rows(inputs)
    rerun = humbert_rerun(r)
    printed, paper, v2 = rerun["printed"], rerun["paper"], rerun["combined_v2"]
    v2_ci = rerun["combined_v2_intervals"]
    clear = groups_clear_of_print(rerun)
    behind = [g for g in HUMSET_GROUPS if v2[g] < printed[g]]
    groups = [*HUMSET_GROUPS, "Mean"]
    c = paper_cites(inputs)

    def span(bounds: Sequence[float]) -> str:
        return f"{bounds[0]:.3f} to {bounds[1]:.3f}"

    def cell(bounds: Sequence[float]) -> str:
        return f"{bounds[0]:.3f}&ndash;{bounds[1]:.3f}"

    names = {"Mean": "Five-group mean"}
    rows = "".join(
        f"<tr><td>{names.get(g, g)}</td><td class=num>{printed[g]:.{3 if g == 'Mean' else 2}f}</td>"
        f"<td class=num>{paper[g]:.3f}</td>"
        f"<td class=num><b>{v2[g]:.3f}</b></td><td class=num>{cell(v2_ci[g])}</td></tr>"
        for g in groups
    )
    tier_one = [tier_one_average(scores) for scores in (printed, paper, v2)]
    rows += (
        f"<tr><td>{info('Tier-1 average', TIER_ONE_TIP.format(humbert=c['humbert']))}</td>"
        f"<td class=num>{tier_one[0]:.2f}</td>"
        f"<td class=num>{tier_one[1]:.3f}</td><td class=num><b>{tier_one[2]:.3f}</b></td>"
        f"<td class=num>&ndash;</td></tr>"
    )
    g2 = [as_share(v) for v in r["11_second_attempt"]["eurlex"]["against_first"]["gain"]["all"]]
    g2t = [as_share(v) for v in r["11_second_attempt"]["eurlex_train_only"]["against_first"]["gain"]["all"]]
    return f"""<section class=col>
  <span class=eyebrow>Full test sets</span>
  <h2>Against the published benchmarks</h2>
  <p>The published benchmarks use the full test sets: 5,000 EU laws, and
  {r["15_humset_v2"]["sizes"]["test"]:,} humanitarian excerpts in HumSet 2.0.0. Every method was run on all
  of HumSet 2.0.0's, as its table above shows; on the laws only methods without an LLM call per law were
  run on all of them.</p>
  <h3>EU laws: mean R-Precision</h3>
  <p class=note>For each law, take as many of the top-ranked labels as the law really has, and count the
  share that
  are correct. 1.00 is perfect. The MultiEURLEX paper prints it as a percentage: 48.4 there is 0.484
  here.</p>
  <div class=legend><span class=l1>Combined method, second attempt</span><span class=l2>Combined method, other
  versions</span><span>Single models</span><span class=lr>Published benchmark</span></div>
  {ranked_bars(laws, 0.7, "{:.3f}")}
  <p>The second attempt added GLiNER2.5 and each label's recent history. It raised the score by {g2[0]:.3f}
  (95% interval {g2[1]:.3f} to {g2[2]:.3f}) with newer laws, and by {g2t[0]:.3f} ({g2t[1]:.3f} to
  {g2t[2]:.3f})
  with training laws only. It was made after seeing the first attempt's test scores, so we report it as a
  second
  attempt.</p>
  <h3>Humanitarian excerpts: mean macro F1</h3>
  <p class=note>F1 is worked out for each label and then averaged, so rare labels count as much as common
  ones. This is the measure the HumBERT paper {c["humbert"]} reports, over five groups of labels, on all
  {r["15_humset_v2"]["sizes"]["test"]:,} test excerpts of HumSet 2.0.0.</p>
  {ranked_bars(on_v2, 0.7, "{:.3f}")}
  <div class=scroll><table><thead>
  <tr><th>{info("Label group", BENCH_TIPS["group"])}</th>
  <th class=num>{info("HumBERT, published", BENCH_TIPS["published"].format(humbert=c["humbert"]))}</th>
  <th class=num>{info("HumBERT recipe, rerun", BENCH_TIPS["rerun"])}</th>
  <th class=num>{info("Combined method", BENCH_TIPS["combined"])}</th>
  <th class=num>{info("95% interval", BENCH_TIPS["interval"])}</th></tr></thead>
  <tbody>{rows}</tbody></table></div>
  <p>On HumSet 2.0.0, the version HumBERT's published figures come from, the combined method scores
  {v2["Mean"]:.3f} on the mean against the published {printed["Mean"]:.3f}, and its 95% interval,
  {span(v2_ci["Mean"])}, lies above it. Its interval lies wholly above the published figure in
  {_and(clear)}{_behind(behind, v2, printed)}. It beats HumBERT's recipe, rerun with the authors' own code
  and settings, in every group.</p>
  <p>Whether that rerun reproduces the published figures depends on how subpillar scores are averaged, which
  the paper {c["humbert"]} does not say. Averaging every label once, as every score here does, it comes within
  {rerun["paper_gap"]:.3f} of every published group. Averaging subpillars within each pillar first, as the
  authors' code does, it falls short in {_shortfalls(rerun["trainer_misses"])}, so notebook 13 records it as
  not reproducing the paper. The published figures have two decimals; their mean is worked out from them.</p>
  <p>The HumBERT paper {c["humbert"]} prints no mean of the five groups. Its own "Avg" column,
  {tier_one[0]:.2f}, is the tier-1
  average: the mean of Sectors, Pillars 1D and Pillars 2D, leaving out both subpillar groups, which score
  lowest. On that average the combined method scores {tier_one[2]:.3f} on 2.0.0, and the recipe rerun
  {tier_one[1]:.3f}.</p>
  <p class=note>The recipe rerun is one run with one seed, so it has no interval.</p>
</section>
"""


def _shortfalls(misses: Mapping[str, float]) -> str:
    """Name each group where the rerun falls short of print, with by how much: "Subpillars 1D (by 0.041)"."""
    return _and([f"{group} (by {gap:.3f})" for group, gap in misses.items()])


def long_documents(inputs: Inputs) -> str:
    """Return the comparison of rules for labelling long documents in pieces."""
    rows = ""
    share = inputs.config["rollup"]["majority_share"]
    for test_set, label in (("eurlex", "120 long EU laws"), ("humset", "200 humanitarian reports")):
        r = inputs.results["07_rollup"]["results"][test_set]
        cells = "".join(
            f"<td class=num>{r[m]['exact']['f1']:.2f}<small>{r[m]['labels_per_item']:.0f} labels</small></td>"
            for m in ("first_chunk_only", "share_of_chunks", "any_chunk", "certainty_weighted")
        )
        rows += f"<tr><td>{label}</td>{cells}<td class=num>${r['cost_usd']:.2f}</td></tr>"
    return f"""<section class=col>
  <span class=eyebrow>Long documents</span>
  <h2>Labelling long documents in pieces</h2>
  <p>Long documents were cut into 400-word pieces, an LLM method labelled each piece, and four rules
  combined the
  pieces' labels into labels for the whole document. Weighting each piece by how sure the model was gave the
  same
  result as keeping any label found in any piece. Which rule worked best depended on the dataset.</p>
  <div class=scroll><table><thead><tr><th>{info("Documents", ROLLUP_TIPS["documents"])}</th>
  <th class=num>{info("First piece only", ROLLUP_TIPS["first"])}</th>
  <th class=num>{info("Label in a share of pieces", ROLLUP_TIPS["share"].format(share=share))}</th>
  <th class=num>{info("Label in any piece", ROLLUP_TIPS["any"])}</th>
  <th class=num>{info("Weighted by certainty", ROLLUP_TIPS["certainty"])}</th>
  <th class=num>{info("LLM cost", ROLLUP_TIPS["cost"])}</th></tr></thead><tbody>{rows}</tbody></table></div>
  <p class=note>Exact-label F1, with the average number of labels each rule gave a document. The
  certainty-weighted
  rule follows Wu et al. (2026), listed below.</p>
</section>
"""


def _surname(ref: dict[str, Any]) -> str:
    return str(ref["authors"]).split(",")[0].split(" and ")[0].split()[-1]


def limits_and_sources(inputs: Inputs) -> str:
    """Return the limits of the results and the list of papers and software."""
    refs = sorted(inputs.methods["references"].values(), key=lambda r: (not r.get("year"), _surname(r)))
    return f"""<section class=col>
  <span class=eyebrow>Read with care</span>
  <h2>Limits</h2>
  <ul class=findings>
    <li>Most methods are simplified from their papers, so these results say little about the originals. Each
    method's card says how ours differs.</li>
    <li>The EU laws table uses 600 laws, a sample of the test set; HumSet 2.0.0's table uses its whole test
    set. Small differences between neighbouring rows may not hold on other documents.</li>
    <li>Some findings, marked with the HumSet 1.0.0 sample, were measured only on that earlier version, where
    every method was first run.</li>
    <li>Results depend on the LLMs named above. Other LLMs could change the order of the LLM methods.</li>
    <li>The second attempt was made after seeing the first attempt's test scores. The pass marks it was judged
    against were not changed.</li>
  </ul>
</section>
<section class=col>
  <span class=eyebrow>Sources</span>
  <h2>Papers and software</h2>
  <ol class=refs>{"".join(ref_html(r) for r in refs)}</ol>
  <p class=note>Paper details checked against arXiv and the ACL Anthology on 10 November 2026. Built by
  <code>scripts/make_report.py</code> from <code>outputs/results/*.json</code>.</p>
</section>
"""


def build_page(repo: Path) -> str:
    """Return the whole report page for the repository at `repo`. Raises if a finding no longer holds."""
    inputs = load_inputs(repo)
    kind_css = {k["kind"]: k["css"] for k in inputs.methods["kinds"]}
    laws, excerpts = method_rows(inputs, "eurlex"), method_rows(inputs, "humset")
    excerpts_v2 = method_rows(inputs, HUMSET_V2)
    check_findings(laws, excerpts_v2, inputs.results)
    cards: dict[str, Any] = {}
    for test_set, rows, unit in (("eurlex", laws, "law"), (HUMSET_V2, excerpts_v2, "excerpt")):
        best = best_scores(rows)
        cards |= {f"{test_set}|{row.key}": card(inputs, row, unit, f"{unit}s", best) for row in rows}
    body = (
        header(inputs, laws, excerpts_v2)
        + results(
            inputs,
            tier_mix(inputs, laws, excerpts_v2),
            results_table(laws, "eurlex", "law", kind_css),
            results_table(
                excerpts_v2, HUMSET_V2, "excerpt", kind_css, published_measure(inputs, excerpts_v2)
            ),
            excerpts_v2,
            excerpts,
        )
        + findings(inputs, laws, excerpts_v2)
        + benchmarks(inputs)
        + long_documents(inputs)
        + limits_and_sources(inputs)
    )
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    style = (ASSETS / "report.css").read_text(encoding="utf-8")
    return (
        f'<meta charset="utf-8">\n<title>Taxonomy Tagger Results</title>\n'
        f'<meta name="claude:viewport-fit-sides" content="cover">\n'
        f'<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        f'<link rel="stylesheet" href="{FONTS}">\n'
        f"<style>\n{style}</style>\n<main>\n{body}</main>\n{CARD_DIALOG}{rb.DIALOGS}"
        f'<script type="application/json" id="cards-data">{data}</script>\n'
        f"{rb.page_data(inputs.results['label_browser'])}"
        f"<script>\n{SCRIPT}\n{rb.SCRIPT}</script>\n"
    )
