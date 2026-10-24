"""Tests for name normalization, alias tables, and conservative variant merging."""

from __future__ import annotations

import pytest

from character_network.names import (
    Merge,
    aliases_to_csv,
    aliases_to_json,
    apply_merges,
    name_tokens,
    normalize_aliases,
    normalize_surface,
    parse_alias_json,
    parse_alias_table,
    resolve_name,
    strip_titles,
    suggest_variant_merges,
    title_of,
)


def test_normalize_surface_handles_possessives_caps_and_unicode():
    assert normalize_surface("Mr. Darcy\u2019s") == "Mr. Darcy"
    assert normalize_surface("ELIZABETH") == "Elizabeth"
    assert normalize_surface("  Ad\u00e8le\u00a0Varens  ") == "Ad\u00e8le Varens"
    assert normalize_surface("O'BRIEN") == "O'Brien"


def test_strip_titles_removes_honorifics_but_keeps_st_john():
    assert strip_titles("Mrs. Bennet") == "Bennet"
    assert strip_titles("Miss Elizabeth Bennet") == "Elizabeth Bennet"
    assert strip_titles("Dr. Watson") == "Watson"
    assert strip_titles("St. John Rivers") == "St. John Rivers"


def test_title_of_returns_lower_case_honorific():
    assert title_of("Mr. Darcy") == "mr"
    assert title_of("Lady Catherine") == "lady"
    assert title_of("Elizabeth") == ""


def test_name_tokens_are_case_folded_and_titleless():
    assert name_tokens("Mr. Fitzwilliam Darcy") == ("fitzwilliam", "darcy")
    assert name_tokens("St John Rivers") == ("st", "john", "rivers")


def test_resolve_name_prefers_explicit_alias_then_strips_title():
    aliases = normalize_aliases({"Lizzy": "Elizabeth Bennet", "Mr. Darcy": "Fitzwilliam Darcy"})
    assert resolve_name("Lizzy", aliases) == "Elizabeth Bennet"
    assert resolve_name("Mr. Darcy", aliases) == "Fitzwilliam Darcy"
    assert resolve_name("Mrs. Bennet", aliases) == "Bennet"
    assert resolve_name("", aliases) == ""


def test_parse_alias_json_requires_object():
    assert parse_alias_json('{"Lizzy": "Elizabeth Bennet"}') == {"lizzy": "Elizabeth Bennet"}
    with pytest.raises(ValueError, match="JSON object"):
        parse_alias_json('["Lizzy"]')


def test_parse_alias_table_accepts_json_csv_and_lines():
    json_table = parse_alias_table('{"Lizzy": "Elizabeth Bennet"}')
    csv_table = parse_alias_table("variant,canonical\nLizzy,Elizabeth Bennet\n")
    line_table = parse_alias_table("Lizzy => Elizabeth Bennet")
    assert json_table == csv_table == line_table == {"lizzy": "Elizabeth Bennet"}


def test_parse_alias_table_rejects_unreadable_rows():
    with pytest.raises(ValueError, match="Could not read alias line"):
        parse_alias_table("Lizzy Elizabeth Bennet")


def test_alias_serialisation_round_trips():
    original = normalize_aliases({"Lizzy": "Elizabeth Bennet", "Mr. Darcy": "Fitzwilliam Darcy"})
    assert original == {"lizzy": "Elizabeth Bennet", "mr darcy": "Fitzwilliam Darcy"}
    assert parse_alias_table(aliases_to_json(original)) == original
    assert parse_alias_table(aliases_to_csv(original)) == original


def test_alias_keys_ignore_honorific_periods():
    aliases = normalize_aliases({"Mr. Bennet": "Mr Bennet"})
    assert resolve_name("Mr Bennet", aliases) == "Mr Bennet"
    assert resolve_name("MR. BENNET", aliases) == "Mr Bennet"


def test_suggest_variant_merges_only_for_unambiguous_partial_names():
    merges = suggest_variant_merges(
        {"Fitzwilliam Darcy": 300, "Darcy": 120, "Elizabeth Bennet": 400, "Elizabeth": 200}
    )
    assert merges == {"Darcy": "Fitzwilliam Darcy", "Elizabeth": "Elizabeth Bennet"}


def test_suggest_variant_merges_leaves_ambiguous_surnames_alone():
    counts = {
        "Elizabeth Bennet": 400,
        "Jane Bennet": 200,
        "Bennet": 150,
    }
    assert suggest_variant_merges(counts) == {}


def test_suggest_variant_merges_ignores_middle_fragments():
    counts = {"St John Rivers": 40, "John": 30}
    assert suggest_variant_merges(counts) == {}


def test_apply_merges_resolves_chains_and_reports_each_merge():
    counts = {"A": 1, "B A": 2, "C B A": 3}
    merges = {"A": "B A", "B A": "C B A"}
    final, applied = apply_merges(counts, merges)
    assert final["A"] == "C B A"
    assert final["B A"] == "C B A"
    assert {merge.variant for merge in applied} == {"A", "B A"}
    assert all(isinstance(merge, Merge) for merge in applied)


def test_apply_merges_drops_cycles_instead_of_guessing():
    final, applied = apply_merges({"A": 1, "B": 1}, {"A": "B", "B": "A"})
    assert final["A"] == "A"
    assert final["B"] == "B"
    assert applied == []
