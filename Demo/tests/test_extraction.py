"""Tests for spaCy-based mention extraction and the name filters."""

from __future__ import annotations

from types import SimpleNamespace

from character_network.extraction import (
    Mention,
    _deduplicate,
    _is_promoted_name,
    _preceding_title,
    _valid_character_name,
    extract_mentions,
)
from character_network.names import normalize_aliases

ALIASES = normalize_aliases(
    {
        "Mr. Bennet": "Mr Bennet",
        "Mrs. Bennet": "Mrs Bennet",
        "Lizzy": "Elizabeth Bennet",
        "Elizabeth": "Elizabeth Bennet",
        "Darcy": "Fitzwilliam Darcy",
        "St. John": "St John Rivers",
    }
)


def test_reattaches_honorific_split_off_by_the_model(nlp):
    result = extract_mentions("Mrs. Bennet spoke to Mr. Bennet about their daughters.", nlp, aliases=ALIASES)
    names = {mention.name for mention in result.mentions}
    assert "Mrs Bennet" in names
    assert "Mr Bennet" in names


def test_alias_table_unifies_variants(nlp):
    text = "Lizzy laughed. Elizabeth smiled. Miss Elizabeth left."
    result = extract_mentions(text, nlp, aliases=ALIASES)
    assert set(result.counts) == {"Elizabeth Bennet"}


def test_manual_names_add_missing_characters(nlp):
    text = "Charlotte Lucas asked Elizabeth about the evening at Netherfield."
    result = extract_mentions(text, nlp, aliases=ALIASES, manual_names=["Charlotte Lucas"])
    assert result.counts.get("Charlotte Lucas", 0) >= 1


def test_excluded_names_are_removed(nlp):
    text = "Elizabeth Bennet walked to Meryton with Jane Bennet."
    result = extract_mentions(text, nlp, aliases=ALIASES, excluded_names=["Meryton"])
    assert "Meryton" not in result.counts


def test_auto_merge_remaps_unambiguous_short_names(nlp):
    text = "Fitzwilliam Darcy bowed. Darcy said nothing. Fitzwilliam Darcy left."
    result = extract_mentions(text, nlp, aliases={}, auto_merge=True)
    assert "Darcy" not in result.counts
    assert result.counts.get("Fitzwilliam Darcy", 0) >= 3
    assert any(merge.variant == "Darcy" for merge in result.merges)


def test_auto_merge_can_be_disabled(nlp):
    text = "Fitzwilliam Darcy bowed. Darcy said nothing."
    result = extract_mentions(text, nlp, aliases={}, auto_merge=False)
    assert "Darcy" in result.counts


def test_extraction_reports_token_count(nlp):
    result = extract_mentions("Elizabeth met Darcy.", nlp, aliases=ALIASES)
    assert result.token_count == 4


def test_fragment_names_are_rejected():
    assert _valid_character_name("Elizabeth Bennet")
    assert _valid_character_name("O'Brien")
    assert _valid_character_name("van Helsing")
    assert not _valid_character_name("Chapter I")
    assert not _valid_character_name("chapter I")
    assert not _valid_character_name("satin sacques")
    assert not _valid_character_name("Charles I")
    assert not _valid_character_name("Miss")
    assert not _valid_character_name("St")
    assert not _valid_character_name("Bessie's Gytrash")
    assert not _valid_character_name("")


def test_location_label_is_promoted_only_for_known_given_names():
    entity = SimpleNamespace(label_="GPE", text="Charlotte")
    assert _is_promoted_name(entity, {"charlotte", "elizabeth"})
    assert not _is_promoted_name(entity, {"elizabeth"})
    assert not _is_promoted_name(SimpleNamespace(label_="PERSON", text="Charlotte"), {"charlotte"})
    assert not _is_promoted_name(SimpleNamespace(label_="GPE", text="New York"), {"new"})


def test_preceding_title_reads_the_token_before_a_name(nlp):
    doc = nlp("Mrs. Bennet and Dr. Watson met Colonel Fitzwilliam.")
    start = next(token.i for token in doc if token.text == "Bennet")
    assert _preceding_title(doc, start) == "Mrs"
    start = next(token.i for token in doc if token.text == "Watson")
    assert _preceding_title(doc, start) == "Dr"


def test_deduplicate_drops_shorter_overlapping_spans():
    mentions = [
        Mention("Mr Bennet", 0, 10, 0, 2),
        Mention("Bennet", 4, 10, 1, 2),
        Mention("Elizabeth", 20, 29, 5, 6),
    ]
    kept = _deduplicate(mentions)
    assert len(kept) == 2
    assert {mention.name for mention in kept} == {"Mr Bennet", "Elizabeth"}
