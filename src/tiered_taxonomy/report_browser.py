"""The report's two data windows: a browser for each taxonomy's labels, and a few example records with the
labels people gave them, each shown in its place in the hierarchy.

Their content comes from `outputs/results/label_browser.json`, written by `scripts/export_label_browser.py`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tiered_taxonomy.report import esc

DATASETS = {
    "eurlex": {
        "title": "EuroVoc, the labels on EU laws",
        "unit": "law",
        "units": "laws",
        "about": "EuroVoc is the EU's multilingual thesaurus. MultiEURLEX tags each law with its concepts, "
        "most of them four or five tiers deep.",
    },
    "humset": {
        "title": "HumSet's humanitarian framework",
        "unit": "excerpt",
        "units": "excerpts",
        "about": "Analysts tag excerpts of humanitarian reports with sectors and pillars (tier 1) and with "
        "the subpillars under the pillars (tier 2).",
    },
}

DIALOGS = """<dialog id=labels aria-labelledby=labels-title>
  <div class="card browser">
    <button type=button class=close aria-label="Close" data-close>&times;</button>
    <header>
      <span class="chip c-none">The labels</span>
      <h2 id=labels-title></h2>
      <p id=labels-about class=summary></p>
    </header>
    <dl id=labels-tiers class=tiercount aria-label="Labels per tier"></dl>
    <div class=freq>
      <h3>How often each label was given in training</h3>
      <p id=labels-freq-note></p>
      <div id=labels-freq-bars class=freqbars role=table
        aria-label="Labels by share of training documents"></div>
    </div>
    <div class=finder>
      <label for=labels-find>Find a label</label>
      <input id=labels-find type=search autocomplete=off placeholder="for example: labelling">
    </div>
    <p id=labels-hits-note class=note hidden></p>
    <ol id=labels-hits class=hits hidden></ol>
    <ul id=labels-tree class=tree role=tree></ul>
    <p class=note>Open a label to see the ones under it. The share beside a label is of the training
    documents people gave it to; hover for the count. Broad labels are rarely given directly.</p>
  </div>
</dialog>
<dialog id=examples aria-labelledby=examples-title>
  <div class="card browser">
    <button type=button class=close aria-label="Close" data-close>&times;</button>
    <header>
      <span class="chip c-none">Example records</span>
      <h2 id=examples-title></h2>
      <p class=summary>Each record as it appears in the test set, and the labels people gave it, each shown
      on its path from tier 1 down.</p>
    </header>
    <p class=marks><span class="tag given"><span class=n>label</span></span> given by a person
      <span class="tag above"><span class=n>label</span></span> a branch above it</p>
    <div id=examples-list class=records></div>
  </div>
</dialog>
"""

SCRIPT = r"""const BROWSER = JSON.parse(document.getElementById("labels-data").textContent);
const LANGUAGES = {en: "English", fr: "French", es: "Spanish"};
const parents = {};
for (const [set, data] of Object.entries(BROWSER)) {
  const up = {};
  const nodes = data.tree.nodes;
  for (const [id, node] of Object.entries(nodes)) {
    for (const kid of node[2]) {
      if (!(kid in up) || nodes[id][1] < nodes[up[kid]][1]) up[kid] = id;
    }
  }
  parents[set] = up;
}
function pathTo(set, id) {
  const path = [id];
  while (path[0] in parents[set]) path.unshift(parents[set][path[0]]);
  return path;
}
function labelLine(set, id, mark) {
  const node = BROWSER[set].tree.nodes[id];
  const line = el("span", "tag" + (mark ? " " + mark : ""));
  line.append(el("span", "t", "T" + node[1]), el("span", "n", node[0]));
  return line;
}
function share(set, count) {
  const pct = 100 * count / BROWSER[set].train_documents;
  return Number(pct.toPrecision(2)).toLocaleString(undefined, {maximumSignificantDigits: 2}) + "%";
}
function usedSpan(set, count, none) {
  const span = el("span", "used", count ? share(set, count) + " of training" : none);
  if (count) {
    const total = BROWSER[set].train_documents.toLocaleString();
    span.title = `${count.toLocaleString()} of ${total} training ${DATASETS[set].units}`;
  }
  return span;
}
const BANDS = [
  ["never given", (p) => p === 0],
  ["under 0.01%", (p) => p > 0 && p < 0.01],
  ["0.01% to 0.1%", (p) => p >= 0.01 && p < 0.1],
  ["0.1% to 1%", (p) => p >= 0.1 && p < 1],
  ["1% to 10%", (p) => p >= 1 && p < 10],
  ["10% or more", (p) => p >= 10],
];
function showFrequency(set) {
  const data = BROWSER[set], counts = Object.values(data.tree.nodes).map((node) => node[3]);
  const shares = counts.map((count) => 100 * count / data.train_documents);
  const sizes = BANDS.map(([, test]) => shares.filter(test).length);
  const top = Math.max(...sizes), all = counts.length;
  $("labels-freq-bars").replaceChildren(...BANDS.map(([name], i) => {
    const row = el("div", "fbar" + (i === 0 ? " never" : ""));
    row.setAttribute("role", "row");
    const track = el("span", "track");
    const fill = el("i");
    fill.style.width = (100 * sizes[i] / top) + "%";
    if (sizes[i]) fill.style.minWidth = "2px";
    track.append(fill);
    row.title = `${sizes[i].toLocaleString()} labels given to ${name} of the training ${DATASETS[set].units}`;
    row.append(el("span", "lbl", name), track,
      el("span", "val", `${sizes[i].toLocaleString()} labels · ${Math.round(100 * sizes[i] / all)}%`));
    return row;
  }));
  const used = counts.filter((count) => count > 0).sort((a, b) => b - a);
  const given = used.reduce((sum, count) => sum + count, 0);
  const head = used.slice(0, Math.ceil(used.length / 10)).reduce((sum, count) => sum + count, 0);
  const median = used[Math.floor(used.length / 2)];
  $("labels-freq-note").textContent = `Share of the ${data.train_documents.toLocaleString()} training ` +
    `${DATASETS[set].units} each label was given to. The most-used tenth of the labels accounts for ` +
    `${Math.round(100 * head / given)}% of all the labels given; the typical label in use was given to ` +
    `${median.toLocaleString()} ${DATASETS[set].units} (${share(set, median)}).`;
}
function treeItem(set, id) {
  const node = BROWSER[set].tree.nodes[id];
  const li = el("li");
  li.setAttribute("role", "treeitem");
  const row = el(node[2].length ? "button" : "div", "row");
  if (node[2].length) {
    row.type = "button";
    row.setAttribute("aria-expanded", "false");
    row.append(el("span", "twist", "▸"));
  } else row.append(el("span", "twist leaf", ""));
  row.append(labelLine(set, id), usedSpan(set, node[3], ""));
  if (node[2].length) row.append(el("span", "kids", node[2].length + " under it"));
  li.append(row);
  if (node[2].length) row.addEventListener("click", () => {
    const open = row.getAttribute("aria-expanded") === "true";
    row.setAttribute("aria-expanded", String(!open));
    row.firstChild.textContent = open ? "▸" : "▾";
    if (open) { li.querySelector("ul")?.remove(); return; }
    const ul = el("ul");
    ul.setAttribute("role", "group");
    ul.append(...node[2].map((kid) => treeItem(set, kid)));
    li.append(ul);
  });
  return li;
}
function showLabels(set) {
  const data = BROWSER[set], about = DATASETS[set];
  $("labels-title").textContent = about.title;
  $("labels-about").textContent = about.about;
  $("labels-tiers").replaceChildren(...Object.entries(data.tree.tiers).map(([tier, n]) => {
    const box = el("div");
    box.append(el("dt", "", "Tier " + tier), el("dd", "", n.toLocaleString()));
    return box;
  }));
  showFrequency(set);
  $("labels-tree").replaceChildren(...data.tree.roots.map((id) => treeItem(set, id)));
  const find = $("labels-find");
  find.value = "";
  find.oninput = () => findLabels(set, find.value.trim().toLowerCase());
  findLabels(set, "");
  $("labels").showModal();
}
function findLabels(set, query) {
  const hits = $("labels-hits"), note = $("labels-hits-note"), tree = $("labels-tree");
  const searching = query.length >= 2;
  hits.hidden = note.hidden = !searching; tree.hidden = searching;
  if (!searching) return;
  const nodes = BROWSER[set].tree.nodes;
  const found = Object.keys(nodes).filter((id) => nodes[id][0].toLowerCase().includes(query));
  found.sort((a, b) => nodes[a][1] - nodes[b][1] || nodes[a][0].localeCompare(nodes[b][0]));
  note.textContent = found.length === 0 ? "No label matches." :
    found.length > 40 ? `${found.length.toLocaleString()} labels match; the first 40 are shown.` :
    `${found.length} label${found.length === 1 ? "" : "s"} match.`;
  hits.replaceChildren(...found.slice(0, 40).map((id) => {
    const li = el("li");
    const crumbs = el("span", "crumbs");
    pathTo(set, id).forEach((step, i, all) => {
      if (i) crumbs.append(el("span", "sep", "\u203a"));
      crumbs.append(el("span", i === all.length - 1 ? "here" : "", nodes[step][0]));
    });
    li.append(crumbs, usedSpan(set, nodes[id][3], "never in training"));
    return li;
  }));
}
function labelTree(set, labels) {
  const kids = {}, roots = [];
  for (const label of labels) {
    const path = pathTo(set, label);
    path.forEach((id, i) => {
      const above = i ? path[i - 1] : null;
      const list = above === null ? roots : (kids[above] ??= []);
      if (!list.includes(id)) list.push(id);
    });
  }
  const build = (ids) => {
    const ul = el("ul", "paths");
    for (const id of ids) {
      const li = el("li");
      li.append(labelLine(set, id, labels.includes(id) ? "given" : "above"));
      if (kids[id]) li.append(build(kids[id]));
      ul.append(li);
    }
    return ul;
  };
  return build(roots);
}
function showExamples(set) {
  const data = BROWSER[set], about = DATASETS[set];
  $("examples-title").textContent = `${data.examples.length} test ${about.units} and their labels`;
  $("examples-list").replaceChildren(...data.examples.map((record) => {
    const art = el("article", "record");
    const meta = el("p", "meta");
    const name = record.link ? el("a", "", record.id) : el("span", "", about.unit + " " + record.id);
    if (record.link) { name.href = record.link; name.target = "_blank"; name.rel = "noopener"; }
    meta.append(name, el("span", "", LANGUAGES[record.language] ?? record.language),
      el("span", "", record.words.toLocaleString() + " words"));
    const text = el("blockquote", "text", record.text + (record.cut ? " …" : ""));
    const head = el("h3", "", `${record.labels.length} labels given`);
    art.append(meta, text);
    const cutNote = `The first words only; the full ${about.unit} is linked above.`;
    if (record.cut) art.append(el("p", "note", cutNote));
    art.append(head, labelTree(set, record.labels));
    return art;
  }));
  $("examples").showModal();
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-browse]");
  if (b) (b.dataset.browse === "labels" ? showLabels : showExamples)(b.dataset.set);
});
for (const d of [$("labels"), $("examples")]) {
  d.querySelector("[data-close]").addEventListener("click", () => d.close());
  d.addEventListener("click", (e) => { if (e.target === d) d.close(); });
}
"""


def data_links(test_set: str, browser: Mapping[str, Any]) -> str:
    """Return the two buttons under a results table's heading that open its labels and its examples."""
    about = DATASETS[test_set]
    labels = len(browser[test_set]["tree"]["nodes"])
    examples = len(browser[test_set]["examples"])
    return (
        f'<div class=data-links><button type=button class=data data-browse=labels data-set="{test_set}">'
        f"Browse all {labels:,} labels</button>"
        f'<button type=button class=data data-browse=examples data-set="{test_set}">'
        f"See {examples} example {esc(about['units'])} and their labels</button></div>"
    )


def page_data(browser: Mapping[str, Any]) -> str:
    """Return the browser's data and the dataset descriptions as page script blocks, safe inside HTML."""
    data = json.dumps(browser, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    about = json.dumps(DATASETS, ensure_ascii=False).replace("</", "<\\/")
    return (
        f'<script type="application/json" id="labels-data">{data}</script>\n'
        f"<script>\nconst DATASETS = {about};\n</script>\n"
    )
