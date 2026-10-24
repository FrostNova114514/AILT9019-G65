"""Tests for the ground-truth metrics used by the evaluation report."""

from __future__ import annotations

import networkx as nx
import pytest

from character_network.evaluation import _recall_at_k, evaluate_book, match_names


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("Elizabeth Bennet", "Fitzwilliam Darcy", weight=5)
    graph.add_edge("Elizabeth Bennet", "Jane Bennet", weight=4)
    graph.add_edge("Elizabeth Bennet", "Bennet", weight=3)
    weights = {"Elizabeth Bennet": 12, "Fitzwilliam Darcy": 5, "Jane Bennet": 4, "Bennet": 3}
    for name, weight in weights.items():
        graph.nodes[name]["weighted_degree"] = weight
        graph.nodes[name]["frequency"] = weight
    return graph


def test_matching_ignores_honorifics_and_is_one_to_one():
    matches = match_names(
        ["Elizabeth Bennet", "Bennet", "Jane"],
        ["Elizabeth Bennet", "Mr Bennet", "Jane Bennet"],
    )
    references = [match.reference for match in matches]
    assert references.count("Mr Bennet") <= 1
    assert set(references) == {"Elizabeth Bennet", "Mr Bennet", "Jane Bennet"}


def test_recall_at_k_only_counts_matches_inside_the_top_k():
    detected = [f"Extra {index}" for index in range(10)] + ["Elizabeth Bennet"]
    reference_to_detected = {"Elizabeth Bennet": "Elizabeth Bennet"}
    assert _recall_at_k(10, detected, reference_to_detected, ["Elizabeth Bennet"]) == 0.0
    assert _recall_at_k(20, detected, reference_to_detected, ["Elizabeth Bennet"]) == 1.0


def test_evaluate_book_reports_recall_edges_and_centrality():
    ground_truth = {
        "title": "Test",
        "protagonist": "Elizabeth Bennet",
        "main_characters": ["Elizabeth Bennet", "Fitzwilliam Darcy", "Mr Bennet"],
        "relationships": [
            ["Elizabeth Bennet", "Fitzwilliam Darcy"],
            ["Elizabeth Bennet", "Mr Bennet"],
            ["Fitzwilliam Darcy", "Mr Bennet"],
        ],
    }
    detected = ["Elizabeth Bennet", "Fitzwilliam Darcy", "Bennet"]
    evaluation = evaluate_book("Test", _graph(), detected, ground_truth)

    assert evaluation.recall_at_10 == 1.0
    assert evaluation.edge_recall == pytest.approx(2 / 3, abs=1e-3)
    assert evaluation.missed_edges == [("Fitzwilliam Darcy", "Mr Bennet")]
    assert evaluation.protagonist_rank == 1

    payload = evaluation.as_dict()
    assert payload["title"] == "Test"
    assert payload["nodes"] == 4
    assert payload["reference_characters"] == ground_truth["main_characters"]


def test_evaluate_book_flags_unexpected_top_names():
    ground_truth = {
        "protagonist": "Elizabeth Bennet",
        "main_characters": ["Elizabeth Bennet"],
        "relationships": [],
    }
    detected = ["Elizabeth Bennet", "Meryton"]
    evaluation = evaluate_book("Test", _graph(), detected, ground_truth)
    assert "Meryton" in evaluation.unexpected_top_10
    assert evaluation.precision_at_10 == 0.5
