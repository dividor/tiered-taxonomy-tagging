"""The charts, in one style. Each method keeps its colour in every chart, so a reader only learns them
once."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"
NEUTRAL = "#9a9992"

# Fixed slots from a palette validated for colour-blind readers. A colour follows its method, never its rank.
METHOD_COLOURS = {
    "trained": "#2a78d6",
    "pipeline": "#eb6834",
    "funnel_cards": "#1baf7a",
    "funnel_hybrid": "#eda100",
    "walk_anchored": "#e87ba4",
    "select_all": "#008300",
    "walk_branching": "#6250d6",
    "walk_strict": "#e34948",
    "retrieval": NEUTRAL,
    "funnel_names": NEUTRAL,
    "funnel_cards_checked": "#1baf7a",
    "quote filter": "#eda100",
    "second reading": "#6250d6",
    # Trained models share blue, and the paper's LLM recipes share the pipeline's orange, as one family each.
    "omikuji": "#2a78d6",
    "xr_transformer": "#123f77",
    "embedding_classifier": "#6ba3e3",
    "knn": "#a7c8ee",
    "llm_ensemble": "#b8481f",
    "ki_fsprompt": "#f29b76",
    "combined": INK,
    "combined_llm": INK_SECONDARY,
    # Printed scores are grey because they are not ours. The new technique's parts reuse palette slots.
    "bert_printed": "#c9c8c1",
    "humbert_printed": NEUTRAL,
    "humset_paper_printed": "#c9c8c1",
    "classifier": "#6250d6",
    "classifier_train_only": "#b2a9ec",
    "reranker": "#e87ba4",
    "llm": "#eda100",
    "new_technique": "#e34948",
    "new_technique_train_only": "#f09a99",
    "gliner": "#1baa8b",
    "first_attempt": "#f09a99",
    "second_attempt": "#e34948",
    "first_attempt_train_only": "#a7c8ee",
    "second_attempt_train_only": "#2a78d6",
}
METHOD_NAMES = {
    "trained": "Trained classifier",
    "pipeline": "LLM pipeline",
    "funnel_cards": "Funnel with label cards",
    "funnel_names": "Funnel, names only",
    "funnel_hybrid": "Hybrid funnel",
    "walk_anchored": "Tree walk, anchored",
    "walk_branching": "Tree walk, branching",
    "walk_strict": "Tree walk, strict",
    "select_all": "LLM, all labels shown",
    "retrieval": "Retrieval only",
    "funnel_cards_checked": "Funnel, second reading",
    "quote filter": "Real-quote test",
    "second reading": "Second reading",
    "omikuji": "Omikuji (TF-IDF)",
    "xr_transformer": "XR-Transformer (paper)",
    "embedding_classifier": "Embedding classifier",
    "knn": "Similar documents",
    "llm_ensemble": "LLM-Ensemble (paper)",
    "ki_fsprompt": "KI-FSPrompt (paper)",
    "combined": "Combined, no LLM",
    "combined_llm": "Combined, with LLM",
    "label_search": "Label search",
    "bert_printed": "BERT (as printed)",
    "humbert_printed": "HumBERT (as printed)",
    "humset_paper_printed": "HumSet paper, XLM-R (as printed)",
    "classifier": "Fine-tuned encoder",
    "classifier_train_only": "Fine-tuned encoder, no newer laws",
    "reranker": "Fine-tuned reranker",
    "llm": "Fine-tuned small LLM",
    "new_technique": "New technique",
    "new_technique_train_only": "New technique, no newer laws",
    "gliner": "GLiNER2.5, fine-tuned",
    "first_attempt": "First attempt",
    "second_attempt": "Second attempt",
    "first_attempt_train_only": "First attempt, no newer laws",
    "second_attempt_train_only": "Second attempt, no newer laws",
}
HEADER_INCHES = 0.85
SEQUENCE = ("#a7c8ee", "#6ba3e3", "#2a78d6", "#1a5aa8", "#123f77")


def _style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=INK_SECONDARY, length=0, labelsize=10)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)


def _figure(columns: int, width: float = 5.2, height: float = 4.2) -> tuple[Figure, list[Axes]]:
    figure, axes = plt.subplots(
        1, columns, figsize=(width * columns, height), squeeze=False, facecolor=SURFACE
    )
    panels = list(axes[0])
    for panel in panels:
        _style(panel)
    return figure, panels


def _finish(figure: Figure, title: str, subtitle: str, path: Path) -> Path:
    figure.tight_layout()
    height = figure.get_figheight()
    figure.subplots_adjust(top=1 - HEADER_INCHES / height)
    figure.text(
        0.01, 1 - 0.2 / height, title, ha="left", va="center", fontsize=14, fontweight="bold", color=INK
    )
    figure.text(0.01, 1 - 0.5 / height, subtitle, ha="left", va="center", fontsize=10, color=INK_SECONDARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    return path


def method_colour(method: str) -> str:
    """Return a method's colour. A method with none gets the neutral grey rather than a generated hue."""
    return METHOD_COLOURS.get(method, NEUTRAL)


def method_name(method: str) -> str:
    """Return a method's readable name."""
    return METHOD_NAMES.get(method, method.replace("_", " "))


def _spread_labels(values: Sequence[float], gap: float) -> list[float]:
    """Move label heights apart just enough that they do not overlap, keeping their order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for previous, current in pairwise(order):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


def tier_profile(
    panels: Mapping[str, Mapping[str, Mapping[str, float]]], title: str, subtitle: str, path: Path
) -> Path:
    """Draw F1 by tier for each method, one panel per test set, with methods labelled at the end of each
    line."""
    figure, axes = _figure(len(panels))
    for panel, (name, methods) in zip(axes, panels.items(), strict=True):
        ends: list[float] = []
        labels: list[tuple[str, str]] = []
        for method, tiers in methods.items():
            xs = sorted(tiers, key=int)
            ys = [tiers[x] for x in xs]
            panel.plot(
                xs,
                ys,
                color=method_colour(method),
                linewidth=2,
                marker="o",
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
            )
            ends.append(ys[-1])
            labels.append((method, f"{method_name(method)} {ys[-1]:.2f}"))
        for height, (method, text) in zip(_spread_labels(ends, 0.045), labels, strict=True):
            panel.annotate(
                text,
                (len(xs) - 1, height),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                color=INK,
            )
            panel.plot([], [], color=method_colour(method))
        panel.set_title(name, loc="left", fontsize=11, color=INK)
        panel.set_ylim(0, 1)
        panel.set_ylabel("F1", color=INK_SECONDARY, fontsize=10)
        panel.set_xlabel("Tier (1 is broadest)", color=INK_SECONDARY, fontsize=10)
        panel.set_xlim(-0.2, len(xs) - 1 + 0.2)
    return _finish(figure, title, subtitle, path)


def method_bars(
    panels: Mapping[str, Mapping[str, float]], axis_label: str, title: str, subtitle: str, path: Path
) -> Path:
    """Draw one value per method as horizontal bars with direct labels, one panel per test set."""
    figure, axes = _figure(len(panels), height=max(2.6, 0.55 * max(len(v) for v in panels.values()) + 1.2))
    for panel, (name, values) in zip(axes, panels.items(), strict=True):
        ordered = sorted(values, key=lambda key: values[key])
        positions = range(len(ordered))
        panel.barh(
            list(positions),
            [values[m] for m in ordered],
            color=[method_colour(m) for m in ordered],
            height=0.62,
            edgecolor=SURFACE,
            linewidth=2,
        )
        top = max(values.values()) or 1.0
        for position, method in zip(positions, ordered, strict=True):
            panel.text(
                values[method] + top * 0.015,
                position,
                f"{values[method]:.2f}",
                va="center",
                fontsize=10,
                color=INK,
            )
        panel.set_yticks(list(positions), [method_name(m) for m in ordered])
        panel.set_title(name, loc="left", fontsize=11, color=INK)
        panel.set_xlabel(axis_label, color=INK_SECONDARY, fontsize=10)
        panel.set_xlim(0, top * 1.14)
        panel.grid(axis="y", visible=False)
        panel.grid(axis="x", color=GRID, linewidth=0.8)
    return _finish(figure, title, subtitle, path)


def recall_curves(
    series: Mapping[str, Mapping[str, float]], limit_note: str, title: str, subtitle: str, path: Path
) -> Path:
    """Draw how many gold labels survive retrieval as the number of candidates grows, labelled at line
    ends."""
    figure, axes = _figure(1, width=7.0, height=4.4)
    panel = axes[0]
    slots = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    for slot, (name, points) in zip(slots, series.items(), strict=False):
        xs = sorted(int(key) for key in points)
        ys = [points[str(x)] for x in xs]
        panel.plot(
            xs,
            ys,
            color=slot,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=f"{name} ({ys[-1]:.0%} at {xs[-1]})",
        )
    panel.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK)
    panel.set_xscale("log")
    ticks = sorted(int(key) for key in next(iter(series.values())))
    panel.set_xticks(ticks, [str(tick) for tick in ticks])
    panel.minorticks_off()
    panel.set_ylim(0, 1)
    panel.set_ylabel("Share of correct labels found", color=INK_SECONDARY, fontsize=10)
    panel.set_xlabel(limit_note, color=INK_SECONDARY, fontsize=10)
    return _finish(figure, title, subtitle, path)


def rarity_bars(panels: Mapping[str, Mapping[str, float]], title: str, subtitle: str, path: Path) -> Path:
    """Draw recall for each rarity band and method, as grouped bars with direct labels."""
    bands = ("common", "rare", "unseen")
    methods = list(panels)
    figure, axes = _figure(1, width=8.0, height=4.2)
    panel = axes[0]
    width = 0.8 / max(1, len(methods))
    for slot, method in enumerate(methods):
        xs = [position + slot * width for position in range(len(bands))]
        ys = [panels[method].get(band, 0.0) for band in bands]
        panel.bar(xs, ys, width=width * 0.92, color=method_colour(method), label=method_name(method))
        for x, y in zip(xs, ys, strict=True):
            panel.text(x, y + 0.012, f"{y:.0%}", ha="center", fontsize=9, color=INK)
    panel.set_xticks(
        [position + 0.4 - width / 2 for position in range(len(bands))],
        ["Common in training", "Rarest quarter", "Never seen"],
    )
    panel.set_ylim(0, 1)
    panel.set_ylabel("Recall", color=INK_SECONDARY, fontsize=10)
    panel.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK)
    return _finish(figure, title, subtitle, path)


def category_bars(
    series: Mapping[str, Mapping[str, float]], axis_label: str, title: str, subtitle: str, path: Path
) -> Path:
    """Draw one group of bars per category and one bar per method, with direct labels and a legend."""
    categories = list(next(iter(series.values())))
    methods = list(series)
    figure, axes = _figure(1, width=10.0, height=4.6)
    panel = axes[0]
    width = 0.84 / len(methods)
    for slot, method in enumerate(methods):
        xs = [position + slot * width for position in range(len(categories))]
        ys = [series[method][category] for category in categories]
        panel.bar(xs, ys, width=width * 0.92, color=method_colour(method), label=method_name(method))
        for x, y in zip(xs, ys, strict=True):
            panel.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=8.5, color=INK)
    panel.set_xticks([position + 0.42 - width / 2 for position in range(len(categories))], categories)
    panel.set_ylim(0, 1)
    panel.set_ylabel(axis_label, color=INK_SECONDARY, fontsize=10)
    panel.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK, ncols=2)
    return _finish(figure, title, subtitle, path)
