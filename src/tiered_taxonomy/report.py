"""Build the results report: one page comparing every method, from the saved results and the configuration.

Every number on the page comes from `outputs/results/*.json` or `configs/experiment.yaml`. What the page
says about each method and its sources is in `report_assets/methods.yaml`. The findings are written for
particular results, so `check_findings` raises if a rerun changes any of the facts they state.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tiered_taxonomy import gpu_jobs as gj
from tiered_taxonomy.report_papers import Published, cited_methods

ASSETS = Path(__file__).parent / "report_assets"
NOTEBOOKS = (
    "01_test_sets",
    "gpu_jobs",
    "label_browser",
    "03_bakeoff",
    "04_funnel",
    "05_tree_walk",
    "06_evidence",
    "07_rollup",
    "09_paper_recipes",
    "12_published_benchmarks",
    "10_combined_method",
    "11_second_attempt",
    "14_like_for_like",
    "13_humbert_reproduction",
    "15_humset_v2",
    "16_paper_measure",
    "17_humset_full_test",
)
FULL_TEST = "17_humset_full_test"
SCOPE = {"newer": "with newer laws", "train_only": "training laws only"}
LOCAL = frozenset({"omikuji", "embedding_classifier", "knn", "label_search"})
GPU_RUNS = {
    ("humset", "classifier"): ("classifier_humset",),
    ("humset", "reranker"): ("reranker_humset",),
    ("humset", "llm"): ("llm_humset",),
    ("humset", "gliner"): ("gliner_humset",),
    ("humset", "humbert_recipe"): ("humbert_reproduction_full_v1",),
    ("train_only", "classifier"): ("classifier_eurlex",),
    ("train_only", "xr_transformer"): ("xr_transformer_eurlex",),
    ("newer", "classifier"): ("classifier_eurlex_newer",),
    ("newer", "reranker"): ("reranker_eurlex_newer",),
    ("newer", "llm"): ("llm_eurlex_newer",),
    ("newer", "gliner"): ("gliner_eurlex_newer",),
}
HUMSET_V2 = "humset_v2"
# HumSet 2.0.0's fine-tuning runs.
GPU_RUNS |= {
    (HUMSET_V2, "classifier"): ("classifier_humset_v2",),
    (HUMSET_V2, "reranker"): ("reranker_humset_v2",),
    (HUMSET_V2, "llm"): ("llm_humset_v2",),
    (HUMSET_V2, "gliner"): ("gliner_humset_v2",),
    (HUMSET_V2, "xr_transformer"): ("xr_transformer_humset_v2",),
    (HUMSET_V2, "humbert_recipe"): ("humbert_reproduction_full_v2_rankings",),
}
LLM_KINDS = frozenset({"LLM", "LLM + examples", "LLM + trained model"})
SINGLE_KINDS = frozenset({"Trained", "Fine-tuned"})


@dataclass(frozen=True)
class Inputs:
    """The saved results by notebook name, the configuration and the method descriptions."""

    results: Mapping[str, Any]
    config: Mapping[str, Any]
    methods: Mapping[str, Any]


@dataclass
class Row:
    """One method's line in a results table."""

    key: str
    method: str
    name: str
    sub: str
    kind: str
    scores: Mapping[str, Any]
    llm_cost: float | None = None
    calls: float = 0.0
    gpu_cost: float | None = None
    gpu_partial: bool = False
    laptop: bool = False
    cut: tuple[int, float] | None = None
    scope: str = ""
    sources: list[str] = field(default_factory=list)
    priors: bool = False
    # The notebook that scored this row, when it is not the method's own (HumSet 2.0.0's rows come from 21).
    notebook: str = ""
    # HumBERT's measure on the same excerpts, per published group and their mean; HumSet only. Notebook 16
    # scores 1.0.0's sample, notebook 17 all of 2.0.0's test excerpts.
    paper: Mapping[str, float] | None = None

    @property
    def f1(self) -> float:
        return float(self.scores["exact"]["f1"])

    def tier(self, tier: str) -> float:
        return float(self.scores["tiers"][tier]["f1"])


def esc(text: object) -> str:
    """Escape text for HTML."""
    return html.escape(str(text))


def load_inputs(repo: Path) -> Inputs:
    """Read the saved results, the configuration and the method descriptions. A missing file raises."""
    results = {
        name: json.loads((repo / "outputs" / "results" / f"{name}.json").read_text(encoding="utf-8"))
        for name in NOTEBOOKS
    }
    config = yaml.safe_load((repo / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    methods = cited_methods(yaml.safe_load((ASSETS / "methods.yaml").read_text(encoding="utf-8")))
    return Inputs(results, config, methods)


def job_cost(jobs: Mapping[str, Any], runs: Iterable[str]) -> float:
    """Return what the named GPU runs cost, with any launch a run also needed."""
    return sum(gj.run_cost(jobs, run) for run in runs)


def gpu_cost(jobs: Mapping[str, Any], scope: str, sources: Sequence[str]) -> tuple[float, bool]:
    """Return the GPU cost of training the given sources, and whether any GPU run's cost was not recorded."""
    runs: list[str] = []
    unknown = False
    for source in sources:
        if source == "classifier_train_only":
            runs += GPU_RUNS[("train_only", "classifier")]
        elif (scope, source) in GPU_RUNS:
            runs += GPU_RUNS[(scope, source)]
        elif source not in LOCAL:
            unknown = True
    return job_cost(jobs, runs), unknown


def method_name(methods: Mapping[str, Any], method: str, test_set: str) -> str:
    """Return a method's display name, which may differ by dataset."""
    entry = methods["methods"][method]
    return str(entry.get("name_by_dataset", {}).get(test_set, entry["name"]))


def like_for_like(inputs: Inputs, test_set: str) -> Mapping[str, Any]:
    """Return every method's scores on the same documents: all of HumSet 2.0.0's test excerpts (notebook 17),
    else the samples of laws and of HumSet 1.0.0's excerpts (notebook 14)."""
    if test_set == HUMSET_V2:
        return dict(inputs.results["17_humset_full_test"][test_set])
    return dict(inputs.results["14_like_for_like"][test_set])


def _scored_in(test_set: str) -> str:
    """Return the notebook a row was scored in, when it is not the one its method's card names: every
    HumSet 2.0.0 row is scored on the full test set in the last notebook."""
    return FULL_TEST[:2] if test_set == HUMSET_V2 else ""


def _paper(inputs: Inputs, test_set: str, family: str, key: str) -> Mapping[str, float] | None:
    """Return a HumSet row's score by HumBERT's measure: from notebook 17 for 2.0.0, from notebook 16 for
    1.0.0's sample, and None on other datasets. A HumSet row that was not scored raises."""
    if test_set == HUMSET_V2:
        return dict(inputs.results[FULL_TEST][test_set]["paper"][family][key])
    if test_set == "humset":
        return dict(inputs.results["16_paper_measure"][test_set][family][key])
    return None


def _llm_rows(inputs: Inputs, test_set: str) -> list[Row]:
    rows = []
    for method, saved in like_for_like(inputs, test_set)["llm"].items():
        entry = inputs.methods["methods"][method]
        rows.append(
            Row(
                method,
                method,
                method_name(inputs.methods, method, test_set),
                entry["sub"],
                entry["kind"],
                saved["scores"],
                llm_cost=saved["cost_per_1000"],
                calls=saved["calls_per_item"],
                notebook=_scored_in(test_set),
                paper=_paper(inputs, test_set, "llm", method),
            )
        )
    return rows


def _ranked_rows(inputs: Inputs, test_set: str) -> list[Row]:
    rows = []
    jobs = inputs.results["gpu_jobs"]
    for key, saved in like_for_like(inputs, test_set)["ranked"].items():
        method, _, scope = key.partition(":")
        if key == "label_search:newer":
            continue  # label search trains on nothing, so both MultiEURLEX views are the same row
        entry = inputs.methods["methods"][method]
        name = method_name(inputs.methods, method, test_set)
        if scope and method != "label_search":
            name = f"{name}, {SCOPE[scope]}"
        cost, unknown = gpu_cost(jobs, scope or test_set, [method])
        rows.append(
            Row(
                key,
                method,
                name,
                entry["sub"],
                entry["kind"],
                saved["scores"],
                gpu_cost=None if method in LOCAL or unknown else cost,
                laptop=method in LOCAL,
                cut=(saved["limit"], saved["threshold"]),
                scope=scope,
                notebook=_scored_in(test_set),
                paper=_paper(inputs, test_set, "ranked", key),
            )
        )
    return rows


PROPER_NAMES = frozenset({"Omikuji"})


def plain_case(name: str) -> str:
    """Lower a name's first letter for use mid-sentence, unless it starts an acronym (XR-Transformer) or a
    proper name (Omikuji)."""
    first = name.split(",")[0].split(" ")[0]
    return name[0].lower() + name[1:] if name[1:2].islower() and first not in PROPER_NAMES else name


def combined_name(inputs: Inputs, key: str, test_set: str = "humset") -> tuple[str, str]:
    """Name a combined-method row by its attempt and training data, and say what the second attempt added."""
    if ":" not in key:
        sources = like_for_like(inputs, test_set)["combined"][key]["sources"]
        names = [plain_case(method_name(inputs.methods, source, test_set)) for source in sources]
        listed = f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]
        if test_set == HUMSET_V2:
            return "Combined method", f"Weighs {listed}, chosen on version 2.0.0's development excerpts"
        first = inputs.results["10_combined_method"]["humset"]["sources"]
        second = inputs.results["11_second_attempt"]["humset"]["sources"]
        if sorted(first) == sorted(second):
            return "Combined method", f"Weighs {listed}; the same in both attempts"
        return "Combined method", f"Weighs {listed}, the second attempt's choice"
    attempt, scope = key.split(":")
    run = "eurlex" if scope == "newer" else "eurlex_train_only"
    if attempt == "combined_first":
        sources = inputs.results["10_combined_method"][run]["sources"]
        names = [plain_case(method_name(inputs.methods, source, "eurlex")) for source in sources]
        listed = f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]
        return (
            f"Combined method, first attempt, {SCOPE[scope]}",
            f"Weighs {listed}, chosen on development laws",
        )
    added = sorted(
        set(inputs.results["11_second_attempt"][run]["sources"])
        - set(inputs.results["10_combined_method"][run]["sources"])
    )
    parts = [method_name(inputs.methods, a, "eurlex") for a in added]
    if inputs.results["11_second_attempt"][run]["priors"]:
        parts.append("each label's recent history")
    return f"Combined method, second attempt, {SCOPE[scope]}", "Adds " + " and ".join(parts)


def _combined_rows(inputs: Inputs, test_set: str) -> list[Row]:
    rows = []
    jobs = inputs.results["gpu_jobs"]
    for key, saved in like_for_like(inputs, test_set)["combined"].items():
        scope = key.partition(":")[2]
        name, sub = combined_name(inputs, key, test_set)
        cost, unknown = gpu_cost(jobs, scope or test_set, saved["sources"])
        sources = [
            method_name(inputs.methods, s.removesuffix("_train_only"), test_set)
            + (", training laws only" if s.endswith("_train_only") else "")
            for s in saved["sources"]
        ]
        rows.append(
            Row(
                key,
                "combined",
                name,
                sub,
                "Combined",
                saved["scores"],
                gpu_cost=cost,
                gpu_partial=unknown,
                cut=(saved["limit"], saved["threshold"]),
                scope=scope,
                sources=sources,
                priors=saved["priors"],
                notebook=_scored_in(test_set),
                paper=_paper(inputs, test_set, "combined", key),
            )
        )
    return rows


def method_rows(inputs: Inputs, test_set: str) -> list[Row]:
    """Return every method's row for one dataset, best exact-label F1 first."""
    rows = [*_llm_rows(inputs, test_set), *_ranked_rows(inputs, test_set), *_combined_rows(inputs, test_set)]
    names = [row.name for row in rows]
    if len(set(names)) != len(names):
        raise ValueError(
            f"two rows share a name on {test_set}: {sorted(n for n in names if names.count(n) > 1)}"
        )
    return sorted(rows, key=lambda row: -row.f1)


def cost_text(row: Row, units: str = "") -> tuple[str, float]:
    """Return a row's cost as text, and a number to sort it by."""
    if row.llm_cost is not None:
        return f"${row.llm_cost:.2f} per 1,000{' ' + units if units else ''}", row.llm_cost
    if row.laptop:
        return "laptop", 0.0
    if row.gpu_cost is None:
        return "GPU, cost not recorded", 0.0
    extra = " and more" if row.gpu_partial else ""
    return f"${row.gpu_cost:.2f}{extra} GPU, once", row.gpu_cost


def bar(value: float, top: float, kind: str = "") -> str:
    """Return a horizontal bar, scaled so that `top` fills it."""
    width = 100 * value / top if top else 0
    return f'<span class="bar {kind}"><i style="width:{width:.1f}%"></i></span>'


def info(label: str, tip: str) -> str:
    """Return a heading's text followed by an i button whose tooltip says what the heading means."""
    return (
        f'{esc(label)}<button type=button class=info data-tip="{esc(tip)}" '
        f'aria-label="What {esc(label)} means">i</button>'
    )


AVG_HEAD = "Macro F1, sectors and pillars (paper's Avg)"
F1_TIP = "F1 balances finding the right labels against adding wrong ones. 1.00 is perfect."
HEAD_TIPS = {
    "Method": "The technique. Click a name to open its card: how it works, what it cost and its sources.",
    "Kind": "How the method works: prompting an LLM, a model trained or fine-tuned on the training "
    "documents, or the combined method, which weighs several. The key above the table describes each kind.",
    "Exact labels": "F1 counting only the very label a person gave. Every label is pooled, so common labels "
    "weigh most. " + F1_TIP,
    "Tier": "F1 when a label counts as right if it falls in the right branch at tier {tier} of the taxonomy, "
    "even if it is the wrong label inside it. Tier 1 is the broadest, so the easiest. " + F1_TIP,
    "Rarest quarter": "Recall, from 0 to 1: the share of the correct labels each method found, among the "
    "least-used quarter of labels in training.",
    "Never seen": "Recall, from 0 to 1, on correct labels that no training document had. None means the test "
    "documents hold no such label.",
    "Labels per": "How many labels the method gives each {unit}, on average.",
    "Cost": "LLM methods: what the calls cost per 1,000 documents. Trained methods: one-off GPU time to "
    "train, or free on a laptop.",
}


def _sort_head(text: str, column: int, kind: str, css: str = "num", extra: str = "", tip: str = "") -> str:
    label = info(text, tip) if tip else esc(text)
    return f'<th data-sort="{kind}" data-col="{column}" class="{css}"{extra}>{label}</th>'


RARE_BANDS = ("rare", "unseen")


def _table_head(tiers: Sequence[str], unit: str, published: Published | None) -> str:
    """Return a results table's two heading rows; `published` adds the two macro F1 columns, with their
    tooltips: over the sectors and pillars only, the paper's own Avg, first of the scores, and over all five
    groups after the tiers."""
    paper = published is not None
    column = iter(range(len(tiers) + 9))
    name, kind = next(column), next(column)
    avg_column = next(column) if paper else -1
    exact = next(column)
    tier_columns = [next(column) for _ in tiers]
    paper_column = next(column) if paper else -1
    rare_columns = [next(column), next(column)]
    labels, cost = next(column), next(column)
    first = (
        "<tr>"
        + _sort_head("Method", name, "text", "", " rowspan=2", HEAD_TIPS["Method"])
        + _sort_head("Kind", kind, "text", "", " rowspan=2", HEAD_TIPS["Kind"])
        + f"<th colspan={1 + len(tiers) + 2 * paper} class=group>F1 score, higher is better</th>"
        + "<th colspan=2 class=group>Recall, higher is better</th>"
        + _sort_head(
            f"Labels per {unit}",
            labels,
            "num",
            "num",
            " rowspan=2",
            HEAD_TIPS["Labels per"].format(unit=unit),
        )
        + _sort_head("Cost", cost, "num", "num", " rowspan=2", HEAD_TIPS["Cost"])
        + "</tr>"
    )
    second = (
        "<tr>"
        + (_sort_head(AVG_HEAD, avg_column, "num", tip=published.avg_tip) if published else "")
        + _sort_head("Exact labels", exact, "num", "num sorted", tip=HEAD_TIPS["Exact labels"])
        + "".join(
            _sort_head(f"Tier {t}", c, "num", tip=HEAD_TIPS["Tier"].format(tier=t))
            for t, c in zip(tiers, tier_columns, strict=True)
        )
        + (_sort_head("Macro F1", paper_column, "num", tip=published.head_tip) if published else "")
        + _sort_head("Rarest quarter", rare_columns[0], "num", tip=HEAD_TIPS["Rarest quarter"])
        + _sort_head("Never seen", rare_columns[1], "num", tip=HEAD_TIPS["Never seen"])
        + "</tr>"
    )
    return first + second


def _table_row(
    row: Row,
    test_set: str,
    top: float,
    tiers: Sequence[str],
    lead: bool,
    kind_css: Mapping[str, str],
    published: Published | None,
) -> str:
    s = row.scores
    cost, cost_value = cost_text(row)
    macro, avg = _paper_cells(row.key, row.paper, published)
    cells = [
        f'<td class=name data-v="{esc(row.name)}"><button type=button class=tech '
        f'data-key="{test_set}|{row.key}">{esc(row.name)}</button><span class=sub>{esc(row.sub)}</span></td>',
        f'<td data-v="{esc(row.kind)}"><span class="chip {kind_css[row.kind]}">{esc(row.kind)}</span></td>',
        *avg,
        f'<td class=barcell data-v="{row.f1}">{bar(row.f1, top, "lead" if lead else "")}'
        f"<b>{row.f1:.2f}</b></td>",
        *(f'<td class=num data-v="{row.tier(t)}">{row.tier(t):.2f}</td>' for t in tiers),
        *macro,
        *(_rarity_cell(s["rarity"], band) for band in RARE_BANDS),
        f'<td class=num data-v="{s["labels_per_item"]}">{s["labels_per_item"]:.1f}</td>',
        f'<td class="num cost" data-v="{cost_value}">{esc(cost)}</td>',
    ]
    return f"<tr>{''.join(cells)}</tr>"


def _paper_cells(
    key: str, paper: Mapping[str, float] | None, published: Published | None
) -> tuple[list[str], list[str]]:
    """Return a row's two macro F1 cells, over all five groups and over the sectors and pillars only, each as
    a list that is empty in a table without the paper's measure; on the row that reruns the paper's recipe,
    a note under each says what the paper reported, with a tooltip on how our measurement differs."""
    if paper is None or published is None:
        return [], []
    recipe = key == published.row_key

    def cell(score: float, note: str, tip: str) -> str:
        small = f"<small>{info(note, tip)}</small>" if recipe else ""
        return f'<td class=num data-v="{score}">{score:.2f}{small}</td>'

    return (
        [cell(paper["Mean"], published.note, published.note_tip)],
        [cell(tier_one_average(paper), published.avg_note, published.avg_note_tip)],
    )


def _rarity_cell(rarity: Mapping[str, Mapping[str, float]], band: str) -> str:
    """Return one rare-label recall cell, or "none" when the test items hold no label in that band."""
    if band not in rarity:
        return '<td class="num none" data-v="-1">none</td>'
    recall = rarity[band]["recall"]
    return f'<td class=num data-v="{recall}">{recall:.2f}</td>'


def results_table(
    rows: Sequence[Row],
    test_set: str,
    unit: str,
    kind_css: Mapping[str, str],
    published: Published | None = None,
) -> str:
    """Return the sortable table of every method on one dataset. A table whose rows carry the paper's measure
    needs `published`, what the paper printed by it, and the row it names must be in the table."""
    top = max(row.f1 for row in rows)
    tiers = sorted(rows[0].scores["tiers"])
    paper = rows[0].paper is not None
    if any((row.paper is not None) != paper for row in rows):
        raise ValueError(f"{test_set}: some rows have a score by the paper's measure and some do not")
    if paper != (published is not None):
        raise ValueError(f"{test_set}: a macro F1 column needs what the paper printed, and only then")
    if published is not None and published.row_key not in {row.key for row in rows}:
        raise ValueError(f"{test_set}: no row {published.row_key} to show what the paper printed")
    body = "".join(_table_row(row, test_set, top, tiers, row is rows[0], kind_css, published) for row in rows)
    return (
        f"<div class=scroll><table class=sortable><thead>{_table_head(tiers, unit, published)}</thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def ref_html(ref: Mapping[str, Any]) -> str:
    """Return one source as a list item: its linked title, then its authors, year and where it appeared."""
    when = f" ({ref['year']})" if ref.get("year") else ""
    return (
        f'<li><a href="{esc(ref["link"])}" target="_blank" rel="noopener">{esc(ref["title"])}</a>'
        f"<span>{esc(ref['authors'])}{when}. {esc(ref['where'])}.</span></li>"
    )


def best_scores(rows: Sequence[Row]) -> dict[str, float]:
    """Return the best score in a table for each measure a card charts."""
    tiers = sorted(rows[0].scores["tiers"])
    best = {"Exact": max(r.f1 for r in rows)}
    best |= {f"Tier {t}": max(r.tier(t) for r in rows) for t in tiers}
    return best


def card(inputs: Inputs, row: Row, unit: str, units: str, best: Mapping[str, float]) -> dict[str, Any]:
    """Return what a method's card shows: a summary, its steps, its scores beside the table's best, four key
    facts, how ours differs from the source, and the sources."""
    entry = inputs.methods["methods"][row.method]
    measured = {f"Tier {t}": row.tier(t) for t in sorted(row.scores["tiers"])} | {"Exact": row.f1}
    cost = cost_text(row, units)[0]
    tiles = [
        ["Labelled documents", "needed" if entry["needs_labels"] else "not needed"],
        ["LLM calls", f"{row.calls:.0f} per {unit}" if row.calls else "none"],
        ["Cost", cost],
        [f"Labels per {unit}", f"{row.scores['labels_per_item']:.1f}"],
    ]
    settings = (
        f"Keeps up to {row.cut[0]} labels scoring at least {row.cut[1]:.2f}, set on development data."
        if row.cut
        else ""
    )
    refs = inputs.methods["references"]
    return {
        "title": row.name,
        "sub": row.sub,
        "kind": row.kind,
        "kindClass": {k["kind"]: k["css"] for k in inputs.methods["kinds"]}[row.kind],
        "summary": entry["summary"],
        "steps": entry["steps"],
        "scores": [[label, round(value, 3), round(best[label], 3)] for label, value in measured.items()],
        "tiles": tiles,
        "combines": row.sources,
        "cluesIntro": entry.get("clues_intro", ""),
        "clues": [
            [c["clue"], c["meaning"]]
            for c in entry.get("clues", [])
            if row.priors or not c.get("only_with_priors")
        ],
        "scorer": entry.get("scorer", ""),
        "priors": row.priors,
        "settings": settings,
        "differs": entry["differs"],
        "notebook": row.notebook or entry["notebook"],
        "refs": "".join(ref_html(refs[k]) for k in entry["refs"]),
    }


@dataclass(frozen=True)
class Leaders:
    """The rows the findings name."""

    best: Row
    single: Row
    llm: Row


def leaders(rows: Sequence[Row]) -> Leaders:
    """Return the best row, the best single model and the best LLM method. Rows are best first."""
    return Leaders(
        rows[0], next(r for r in rows if r.kind in SINGLE_KINDS), next(r for r in rows if r.kind in LLM_KINDS)
    )


@dataclass(frozen=True)
class TierMix:
    """Where the labels people assigned sit in each taxonomy, which explains the gap between the exact score
    and the tier scores."""

    law_nodes: int
    law_top_three: int
    law_below_three: float
    excerpt_tier_one: float
    excerpts_between: int
    excerpt_rows: int


def label_path(tree: Mapping[str, Any], name: str) -> list[tuple[int, str]]:
    """Return the tier and name of each label from tier 1 down to the label called `name`, in a label browser
    tree, whose nodes are [name, tier, children, uses]. Raises unless exactly one label has that name and each
    label on the way up has a single parent."""
    nodes = tree["nodes"]
    found = [key for key, node in nodes.items() if node[0] == name]
    if len(found) != 1:
        raise ValueError(f"expected one label called {name!r}, found {len(found)}")
    parents: dict[str, list[str]] = {}
    for key, node in nodes.items():
        for child in node[2]:
            parents.setdefault(child, []).append(key)
    path = [found[0]]
    while path[-1] in parents:
        if len(parents[path[-1]]) != 1:
            raise ValueError(f"{nodes[path[-1]][0]!r} has more than one parent")
        path.append(parents[path[-1]][0])
    return [(nodes[key][1], nodes[key][0]) for key in reversed(path)]


def near_miss_paths(tree: Mapping[str, Any], person: str, method: str) -> list[tuple[int, str]]:
    """Return the path to `person`'s label, for an example of a near miss with `method`'s label. Raises unless
    the two labels are siblings: the same tier, under the same parent."""
    mine, theirs = label_path(tree, person), label_path(tree, method)
    if len(mine) != len(theirs) or mine[:-1] != theirs[:-1] or mine[-1][0] != theirs[-1][0]:
        raise ValueError(f"{person!r} and {method!r} are not labels at the same tier under the same parent")
    return mine


def fine_tuned_behind(rows: Sequence[Row], llm: Row) -> list[Row]:
    """Return the fine-tuned rows that score below the LLM method `llm` by exact labels, best first."""
    return [r for r in rows if r.kind == "Fine-tuned" and r.f1 < llm.f1]


def tier_mix(inputs: Inputs, laws: Sequence[Row], excerpts: Sequence[Row]) -> TierMix:
    """Return where the assigned labels sit, with `excerpts` the HumSet 2.0.0 rows. Raises unless every law
    method's exact score is below each of its tier scores, and most excerpt methods' exact score falls between
    their tier 1 and tier 2 scores, as the page says."""
    sizes = {s["test set"]: s for s in inputs.results["01_test_sets"]["sizes"]}
    law_gold = inputs.results["14_like_for_like"]["eurlex"]["gold"]
    excerpt_gold = inputs.results["15_humset_v2"]["gold"]
    above = [r.name for r in laws if any(r.f1 >= r.tier(t) for t in r.scores["tiers"])]
    if above:
        raise ValueError(f"findings no longer hold: exact score not below every tier for {above}")
    between = sum(r.tier("2") < r.f1 < r.tier("1") for r in excerpts)
    if 2 * between <= len(excerpts):
        raise ValueError("findings no longer hold: most excerpt methods' exact score is not between tiers")
    below = sum(n for t, n in law_gold["by_tier"].items() if int(t) > 3)
    return TierMix(
        law_nodes=sizes["eurlex"]["nodes"],
        law_top_three=sum(sizes["eurlex"]["tier sizes"][t] for t in ("1", "2", "3")),
        law_below_three=below / law_gold["labels"],
        excerpt_tier_one=excerpt_gold["by_tier"]["1"] / excerpt_gold["labels"],
        excerpts_between=between,
        excerpt_rows=len(excerpts),
    )


def tiers_not_led_by_combined(rows: Sequence[Row]) -> list[tuple[str, Row, float]]:
    """Return each tier where a method other than a combined one has the top score: the tier, that method's
    row, and its lead over the best combined row."""
    found = []
    for tier in sorted(rows[0].scores["tiers"]):
        top = max(rows, key=lambda r: r.tier(tier))
        if top.kind != "Combined":
            best = max(r.tier(tier) for r in rows if r.kind == "Combined")
            found.append((tier, top, top.tier(tier) - best))
    return found


HUMSET_GROUPS = ("Sectors", "Pillars 1D", "Subpillars 1D", "Pillars 2D", "Subpillars 2D")


def humbert_rerun(results: Mapping[str, Any]) -> dict[str, Any]:
    """Return HumBERT's recipe rerun on each HumSet version, every label averaged once as in our scores,
    beside the printed figures and our combined method on each version, and the largest gap between the
    rerun on 2.0.0 and print. Also the trainer's own table on 2.0.0, which averages subpillars within each
    pillar first, and the groups where notebook 13 found it too far from print, and how far short it is."""
    rerun = results["13_humbert_reproduction"]
    printed = results["12_published_benchmarks"]["printed"]["humset"]["HumBERT, best"]
    paper = rerun["2.0.0"]["every_label_once"]
    v2 = results["15_humset_v2"]["full_test"]
    return {
        "printed": printed,
        "paper": paper,
        "ours": rerun["1.0.0"]["every_label_once"],
        "combined": results["10_combined_method"]["humset"]["scores"],
        "combined_v2": v2["scores"],
        "combined_v2_intervals": v2["intervals"],
        "paper_gap": max(abs(paper[group] - printed[group]) for group in HUMSET_GROUPS),
        "trainer_table": rerun["2.0.0"]["trainer_table"],
        "trainer_misses": {
            group: printed[group] - rerun["2.0.0"]["trainer_table"][group]
            for group, matched in rerun["2.0.0"]["matches_printed"].items()
            if not matched
        },
    }


TIER_ONE_GROUPS = ("Sectors", "Pillars 1D", "Pillars 2D")


def tier_one_average(scores: Mapping[str, float]) -> float:
    """Return the mean of the three tier-1 groups, which is what the HumBERT paper's "Avg" column holds: it
    leaves out both subpillar groups."""
    return sum(scores[group] for group in TIER_ONE_GROUPS) / len(TIER_ONE_GROUPS)


def groups_clear_of_print(rerun: Mapping[str, Any]) -> list[str]:
    """Return the label groups where the combined method's 95% interval on HumSet 2.0.0 lies wholly above the
    printed HumBERT figure."""
    low = {group: bounds[0] for group, bounds in rerun["combined_v2_intervals"].items()}
    return [group for group in HUMSET_GROUPS if low[group] > rerun["printed"][group]]


def _humbert_checks(results: Mapping[str, Any]) -> dict[str, bool]:
    """Return each fact the findings state about HumBERT's recipe and HumSet 2.0.0, with whether it holds."""
    rerun = humbert_rerun(results)
    checks: dict[str, bool] = {}
    checks["on HumSet 1.0.0 the combined method beats HumBERT's recipe on the mean"] = (
        rerun["combined"]["Mean"] > rerun["ours"]["Mean"]
    )
    checks["on HumSet 2.0.0 the recipe comes within 0.02 of every printed group"] = rerun["paper_gap"] <= 0.02
    checks["by the trainer's own averaging the rerun falls short of print, not above it"] = bool(
        rerun["trainer_misses"]
    ) and all(gap > 0 for gap in rerun["trainer_misses"].values())
    checks["on HumSet 2.0.0 the combined method's tier-1 average is above the paper's Avg"] = (
        tier_one_average(rerun["combined_v2"]) > tier_one_average(rerun["printed"])
    )
    checks["on HumSet 2.0.0 the combined method beats the recipe rerun in every group"] = all(
        rerun["combined_v2"][group] > rerun["paper"][group] for group in HUMSET_GROUPS
    )
    checks["on HumSet 2.0.0 the combined method's interval for the mean is above the printed mean"] = (
        rerun["combined_v2_intervals"]["Mean"][0] > rerun["printed"]["Mean"]
    )
    checks["on HumSet 2.0.0 the combined method is below a printed group only within its rounding"] = all(
        rerun["combined_v2"][group] > rerun["printed"][group] - 0.005 for group in HUMSET_GROUPS
    )
    return checks


def check_findings(laws: Sequence[Row], excerpts: Sequence[Row], results: Mapping[str, Any]) -> None:
    """Raise if any fact the findings state no longer holds for these results. `excerpts` are the HumSet
    2.0.0 rows."""
    e, h = leaders(laws), leaders(excerpts)
    combined = next(r for r in excerpts if r.kind == "Combined")
    walks = {name: {r.key: r.f1 for r in rows} for name, rows in (("laws", laws), ("excerpts", excerpts))}
    checks = {
        "the combined method is best on the laws": e.best.kind == "Combined",
        "on the excerpts the combined method leads at every tier": not tiers_not_led_by_combined(excerpts),
        "on the excerpts the combined method leads by macro F1": max(
            excerpts, key=lambda r: (r.paper or {}).get("Mean", 0.0)
        )
        is combined,
        "the best single models are Legal-BERT and XR-Transformer": (e.single.key, h.single.key)
        == ("classifier:newer", "xr_transformer"),
        "KI-FSPrompt is the best LLM method on both": e.llm.method == h.llm.method == "ki_fsprompt",
        "on the laws every fine-tuned model beats every LLM method": min(
            r.f1 for r in laws if r.kind == "Fine-tuned"
        )
        > e.llm.f1,
        "on the excerpts every trained model without an LLM beats every LLM method": min(
            r.f1 for r in excerpts if r.kind == "Trained"
        )
        > h.llm.f1,
        "on the excerpts the best LLM method beats some fine-tuned model": bool(
            fine_tuned_behind(excerpts, h.llm)
        ),
        "on the laws an LLM method finds the most rare labels": max(
            laws, key=lambda r: r.scores["rarity"]["rare"]["recall"]
        ).kind.startswith("LLM"),
        "keeping three branches open beats one on both": all(
            w["walk_branching"] > w["walk_strict"] for w in walks.values()
        ),
        "starting the walk from search results helps on the laws and hurts on the excerpts": walks["laws"][
            "walk_anchored"
        ]
        > walks["laws"]["walk_strict"]
        and walks["excerpts"]["walk_anchored"] < walks["excerpts"]["walk_strict"],
    }
    checks |= _humbert_checks(results)
    paper, funnel, evid = results["09_paper_recipes"], results["04_funnel"], results["06_evidence"]
    for name, (_, low, high) in {
        "KI-FSPrompt over the pipeline on laws": paper["eurlex"]["ki_fsprompt"]["minus_pipeline"],
        "KI-FSPrompt over the pipeline on excerpts": paper["humset"]["ki_fsprompt"]["minus_pipeline"],
        "the trained shortlist": funnel["differences"]["eurlex"]["hybrid minus cards"],
        "the double-check on laws": evid["differences"]["eurlex"]["second reading minus plain"],
        "the double-check on excerpts": evid["differences"]["humset"]["second reading minus plain"],
    }.items():
        checks[f"the interval for {name} excludes zero"] = low * high > 0
    failed = [name for name, held in checks.items() if not held]
    if failed:
        raise ValueError("findings no longer hold: " + "; ".join(failed))


def interval(triple: Sequence[float]) -> str:
    """Return a paired difference's size, with its 95% interval in smaller type."""
    point, low, high = triple
    a, b = sorted((abs(low), abs(high)))
    return f"{abs(point):.2f} <span class=ci>(95% interval {a:.2f} to {b:.2f})</span>"


def ranked_bars(rows: Sequence[tuple[str, str, float, str]], top: float, fmt: str) -> str:
    """Return a ranked list of labelled bars."""
    items = "".join(
        f"<li><span class=lbl>{esc(a)}<small>{esc(b)}</small></span>{bar(v, top, k)}"
        f"<b>{fmt.format(v)}</b></li>"
        for a, b, v, k in rows
    )
    return f"<ol class=ranked>{items}</ol>"
