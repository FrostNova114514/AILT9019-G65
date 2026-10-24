"""End-to-end tests for the analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from character_network.analysis import (
    InputValidationError,
    analyze_text,
    detect_chapters,
    slice_to_chapters,
)
from character_network.presets import get_preset

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "sample_story.txt"

CHAPTERED_TEXT = (
    "CHAPTER I\n\n"
    + "Elizabeth Bennet met Mr. Darcy at the assembly. Elizabeth disliked him at once. "
    "Mr. Darcy said little to Elizabeth and left early. " * 3
    + "\n\nCHAPTER II\n\n"
    + "Charlotte Lucas spoke with Elizabeth Bennet about Mr. Collins. Elizabeth told "
    "Charlotte that Mr. Collins was tiresome. " * 3
    + "\n\nCHAPTER III\n\n"
    + "George Wickham arrived with Lydia Bennet. Lydia laughed with Wickham all evening, "
    "and Wickham told Lydia about the regiment. " * 3
)

THEORY_TEXT = (
    "Information theory studies the quantification, storage, and communication of "
    "digital information. The field was established by Claude Shannon in 1948. Coding "
    "theory asks how close a code can come to the Shannon limit while remaining "
    "decodable in linear time. Rate-distortion theory extends the framework to lossy "
    "compression, where the question becomes how much fidelity is exchanged for how "
    "many bits. Modern work on network information theory considers several senders "
    "and receivers at once, and the capacity region of such a network is unknown."
)


def test_sample_story_produces_expected_characters(nlp):
    text = SAMPLE_PATH.read_text(encoding="utf-8")
    result = analyze_text(text, nlp, aliases=get_preset("Pride and Prejudice"))
    assert result.graph.number_of_nodes() >= 5
    assert result.graph.number_of_edges() > 0
    assert result.top_characters(1) == ["Elizabeth Bennet"]
    assert ("Elizabeth Bennet", "Fitzwilliam Darcy") in [
        (source, target) if source < target else (target, source)
        for source, target, _ in result.strongest_pairs(10)
    ]
    assert result.metrics["nodes"] == result.graph.number_of_nodes()
    assert result.settings["mode"] == "paragraph"


def test_window_mode_produces_a_graph(nlp):
    text = SAMPLE_PATH.read_text(encoding="utf-8")
    result = analyze_text(text, nlp, aliases=get_preset("Pride and Prejudice"), mode="window", window_size=40)
    assert result.graph.number_of_edges() > 0
    assert result.mode == "window"


def test_theory_style_text_is_refused(nlp):
    with pytest.raises(InputValidationError, match="Not enough character relationships"):
        analyze_text(THEORY_TEXT, nlp)


def test_minimum_edge_weight_can_remove_every_relationship(nlp):
    text = SAMPLE_PATH.read_text(encoding="utf-8")
    with pytest.raises(InputValidationError, match="Not enough character relationships"):
        analyze_text(text, nlp, aliases=get_preset("Pride and Prejudice"), minimum_edge_weight=10)


def test_excluded_names_never_reach_the_graph(nlp):
    text = SAMPLE_PATH.read_text(encoding="utf-8")
    result = analyze_text(
        text,
        nlp,
        aliases=get_preset("Pride and Prejudice"),
        excluded_names=["Charlotte Lucas"],
    )
    assert "Charlotte Lucas" not in result.graph.nodes


def test_manual_names_are_added(nlp):
    text = SAMPLE_PATH.read_text(encoding="utf-8")
    result = analyze_text(
        text,
        nlp,
        aliases=get_preset("Pride and Prejudice"),
        manual_names=["Longbourn", "Netherfield"],
        require_english=False,
    )
    assert "Longbourn" in result.character_counts or "Longbourn" in result.graph.nodes


def test_chapters_are_detected_and_sliced(nlp):
    chapters = detect_chapters(CHAPTERED_TEXT)
    assert [chapter.title for chapter in chapters] == ["Chapter I", "Chapter II", "Chapter III"]
    sliced = slice_to_chapters(CHAPTERED_TEXT, chapters, 3, 3)
    assert "Wickham" in sliced
    assert "Charlotte" not in sliced

    whole = analyze_text(CHAPTERED_TEXT, nlp)
    assert len(whole.chapters) == 3

    last = analyze_text(sliced, nlp)
    assert "George Wickham" in last.character_counts
    assert "Mr. Collins" not in last.character_counts


def test_analysis_result_exposes_helpers(nlp):
    result = analyze_text(SAMPLE_PATH.read_text(encoding="utf-8"), nlp)
    assert result.characters == result.graph.number_of_nodes()
    assert result.relationships == result.graph.number_of_edges()
    assert len(result.top_characters(3)) <= 3
    strongest = result.strongest_pairs(3)
    assert strongest == sorted(strongest, key=lambda item: -item[2])
    assert result.elapsed_seconds >= 0


def test_auto_merge_can_be_turned_off(nlp):
    text = "Fitzwilliam Darcy bowed to Elizabeth Bennet.\n\nDarcy said nothing to Elizabeth.\n\n" + (
        "Elizabeth Bennet waited. "
    ) * 12
    merged = analyze_text(text, nlp, auto_merge=True)
    unmerged = analyze_text(text, nlp, auto_merge=False)
    assert "Darcy" not in merged.character_counts
    assert "Darcy" in unmerged.character_counts
