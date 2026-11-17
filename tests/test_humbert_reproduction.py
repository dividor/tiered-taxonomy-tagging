"""Tests for scoring the HumBERT reproduction per HumSet category."""

from __future__ import annotations

import pytest

from tiered_taxonomy import humbert_reproduction as hr

LABELS = [
    "first_level_tags->sectors->Health",
    "first_level_tags->sectors->Wash",
    "first_level_tags->pillars_1d->Context",
    "first_level_tags->pillars_2d->Impact",
    "subpillars_1d->Context->Environment",
    "subpillars_1d->Context->Economy",
    "subpillars_1d->Context->Politics",
    "subpillars_1d->Displacement->Type",
    "subpillars_2d->Impact->Drivers",
]


def test_category_names_each_label_and_raises_for_others() -> None:
    assert [hr.category(tag) for tag in LABELS[::4]] == ["Sectors", "Subpillars 1D", "Subpillars 2D"]
    with pytest.raises(ValueError, match="not a HumSet label"):
        hr.category("sectors->Health")


def test_label_f1_counts_hits_and_gives_zero_with_no_positives() -> None:
    records = [
        {"gold": ["first_level_tags->sectors->Health"], "predicted": ["first_level_tags->sectors->Health"]},
        {"gold": ["first_level_tags->sectors->Health"], "predicted": ["first_level_tags->sectors->Wash"]},
    ]
    f1 = hr.label_f1(records, LABELS[:3])
    assert f1["first_level_tags->sectors->Health"] == pytest.approx(2 / 3)
    assert f1["first_level_tags->sectors->Wash"] == 0.0
    assert f1["first_level_tags->pillars_1d->Context"] == 0.0


def test_nested_averaging_weights_each_pillar_equally() -> None:
    f1 = dict.fromkeys(LABELS, 1.0)
    f1["subpillars_1d->Displacement->Type"] = 0.0
    flat, nested = hr.category_scores(f1, nested=False), hr.category_scores(f1, nested=True)
    assert flat["Subpillars 1D"] == pytest.approx(0.75)
    assert nested["Subpillars 1D"] == pytest.approx(0.5)
    assert flat["Sectors"] == nested["Sectors"] == 1.0
    assert nested["Mean"] == pytest.approx((1 + 1 + 0.5 + 1 + 1) / 5)
    with pytest.raises(ValueError, match="no labels in"):
        hr.category_scores({LABELS[0]: 1.0}, nested=True)


def test_table_scores_read_the_trainers_rows_and_raise_when_one_is_missing() -> None:
    rows = [{"tag": row, "f_score": "0.5"} for row in hr.TABLE_ROWS.values()]
    assert hr.table_scores(rows) == dict.fromkeys(hr.TABLE_ROWS, 0.5)
    with pytest.raises(ValueError, match="lacks"):
        hr.table_scores(rows[1:])


def test_our_label_maps_each_category_to_this_repositorys_ids() -> None:
    assert hr.our_label("first_level_tags->sectors->Health") == "Health"
    assert hr.our_label("first_level_tags->pillars_2d->Impact") == "Impact"
    assert hr.our_label("subpillars_1d->Context->Environment") == "Context->Environment"
    with pytest.raises(ValueError, match="not a HumSet label"):
        hr.our_label("Health")


def test_to_rankings_renames_labels_keeps_an_identical_repeat_and_raises_on_a_different_one() -> None:
    records = [{"entry_id": 5, "ranking": {"subpillars_2d->Impact->Drivers": 0.7}}]
    assert hr.to_rankings(records) == {"5": {"Impact->Drivers": 0.7}}
    assert hr.to_rankings(records * 2) == {"5": {"Impact->Drivers": 0.7}}
    differing = [*records, {"entry_id": 5, "ranking": {"subpillars_2d->Impact->Drivers": 0.6}}]
    with pytest.raises(ValueError, match="appears twice with different rankings"):
        hr.to_rankings(differing)
