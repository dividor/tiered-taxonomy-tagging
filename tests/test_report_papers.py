"""Tests for the report's citations of papers."""

from __future__ import annotations

import pytest

from tiered_taxonomy.report_papers import author_year, cite, cited_methods, link_citations, with_citations

REFS = {
    "one": {"title": "Solo", "authors": "Ada Lovelace", "year": 2024},
    "two": {"title": "Pair", "authors": "Lisa Kluge and Maximilian Kähler", "year": 2025},
    "many": {"title": "Team", "authors": "Nicolò Tamagnone, Selim Fekih and Navid Rekabsaz", "year": 2023},
    "others": {"title": "Big", "authors": "Daoping Wu, Dong Wang and others", "year": 2026},
    "undated": {"title": "Software", "authors": "Amazon"},
}


def test_author_year_names_one_two_or_the_first_of_many_authors() -> None:
    assert author_year(REFS["one"]) == "Lovelace, 2024"
    assert author_year(REFS["two"]) == "Kluge and Kähler, 2025"
    assert author_year(REFS["many"]) == "Tamagnone et al., 2023"
    assert author_year(REFS["others"]) == "Wu et al., 2026"


def test_a_source_without_a_year_cannot_be_cited() -> None:
    with pytest.raises(ValueError, match="no year"):
        author_year(REFS["undated"])
    with pytest.raises(ValueError, match="at least one"):
        cite()


def test_with_citations_writes_out_one_or_several_sources_and_refuses_unknown_ones() -> None:
    assert (
        with_citations("as the paper {cite:many} says", REFS) == "as the paper (Tamagnone et al., 2023) says"
    )
    assert with_citations("{cite:two,others}", REFS) == "(Kluge and Kähler, 2025; Wu et al., 2026)"
    with pytest.raises(KeyError, match="nope"):
        with_citations("{cite:nope}", REFS)


def test_cited_methods_writes_out_citations_in_text_fields_only() -> None:
    methods = {
        "references": REFS,
        "methods": {"m": {"sub": "in {cite:one}", "differs": "the paper {cite:two}", "refs": ["one"]}},
    }
    made = cited_methods(methods)["methods"]["m"]
    assert made == {
        "sub": "in (Lovelace, 2024)",
        "differs": "the paper (Kluge and Kähler, 2025)",
        "refs": ["one"],
    }


LINKED = {
    "many": {**REFS["many"], "link": "https://example.org/humbert"},
    "two": {**REFS["two"], "link": "https://example.org/pair"},
    "undated": REFS["undated"],
}


def test_link_citations_links_text_but_not_tags_scripts_or_existing_links() -> None:
    page = (
        "<p>as shown (Tamagnone et al., 2023)</p>"
        '<button data-tip="see (Tamagnone et al., 2023)">i</button>'
        '<script>const s = "Tamagnone et al., 2023";</script>'
        '<a href="x">Tamagnone et al., 2023</a> and (Kluge and Kähler, 2025)'
    )
    linked = link_citations(page, LINKED)
    link = '<a href="https://example.org/humbert" target="_blank" rel="noopener">'
    assert f"({link}Tamagnone et al., 2023</a>)" in linked
    assert 'data-tip="see (Tamagnone et al., 2023)"' in linked
    assert 'const s = "Tamagnone et al., 2023";' in linked
    assert '<a href="x">Tamagnone et al., 2023</a>' in linked
    assert ">Kluge and Kähler, 2025</a>)" in linked


def test_link_citations_raises_when_two_sources_cite_the_same_way() -> None:
    clash = {**LINKED, "again": {**LINKED["many"], "title": "Other", "link": "https://example.org/other"}}
    with pytest.raises(ValueError, match="year_suffix"):
        link_citations("<p>x</p>", clash)
    suffixed = {**clash, "again": {**clash["again"], "year_suffix": "b"}}
    assert author_year(suffixed["again"]) == "Tamagnone et al., 2023b"
    assert "Tamagnone et al., 2023b</a>" in link_citations("<p>(Tamagnone et al., 2023b)</p>", suffixed)
