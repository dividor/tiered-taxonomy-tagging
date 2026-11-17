"""Tests for the shared method helpers. Nothing is sent or paid for."""

from __future__ import annotations

from method_fakes import reply
from tiered_taxonomy.methods.base import Prediction, clamp, combine, parse_selection, quote_found


def test_clamp_and_combine() -> None:
    assert clamp(2) == 1.0
    assert clamp("x") == 0.5
    assert combine([Prediction(calls=2, malformed=1), Prediction(calls=1, invalid=3)]) == (3, 1, 3)


def test_parse_selection_keeps_allowed_ids_and_counts_the_rest() -> None:
    result = parse_selection(
        reply('{"labels": [{"id": "a", "confidence": 0.9, "quote": "q"}, {"id": "z"}]}'), {"a"}
    )
    assert result.labels == {"a": 0.9}
    assert result.quotes == {"a": "q"}
    assert result.invalid == 1


def test_parse_selection_marks_malformed_replies() -> None:
    assert parse_selection(reply("nonsense"), {"a"}).malformed == 1


def test_quote_found_ignores_case_and_spacing_but_not_absence() -> None:
    assert quote_found("Food  Security", "the food security\ncrisis")
    assert not quote_found("flood", "the food crisis")
    assert not quote_found("", "text")
