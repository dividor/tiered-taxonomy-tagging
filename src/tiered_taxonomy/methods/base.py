"""What every method returns, and the parsing they share."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tiered_taxonomy.llm import Reply

_SPACE = re.compile(r"\s+")


@dataclass
class Prediction:
    """A method's answer for one item. `labels` maps a label to its score, and `quotes` holds supporting
    text."""

    labels: dict[str, float] = field(default_factory=dict)
    quotes: dict[str, str] = field(default_factory=dict)
    calls: int = 0
    malformed: int = 0
    invalid: int = 0


def clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.5) -> float:
    """Return a number within limits, or the default if the value is not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, number))


def parse_selection(reply: Reply, allowed: Collection[str]) -> Prediction:
    """Read a `labels` reply, keeping only allowed ids and counting the rest as invalid."""
    prediction = Prediction(calls=1)
    if reply.parsed is None or not isinstance(reply.parsed.get("labels"), list):
        prediction.malformed = 1
        return prediction
    for entry in reply.parsed["labels"]:
        label = str(entry.get("id")) if isinstance(entry, Mapping) else ""
        if label not in allowed:
            prediction.invalid += 1
            continue
        prediction.labels[label] = clamp(entry.get("confidence"))
        prediction.quotes[label] = str(entry.get("quote") or "")
    return prediction


def normalise_text(text: str) -> str:
    """Lower-case a text and collapse its whitespace, for comparing a quote with its source."""
    return _SPACE.sub(" ", text).strip().lower()


def quote_found(quote: str, text: str) -> bool:
    """Return whether a non-empty quote really appears in the text."""
    wanted = normalise_text(quote)
    return bool(wanted) and wanted in normalise_text(text)


def combine(predictions: Iterable[Prediction]) -> tuple[int, int, int]:
    """Return the total calls, malformed replies and invalid labels across predictions."""
    calls = malformed = invalid = 0
    for prediction in predictions:
        calls += prediction.calls
        malformed += prediction.malformed
        invalid += prediction.invalid
    return calls, malformed, invalid
