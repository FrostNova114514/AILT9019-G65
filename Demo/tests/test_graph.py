"""Tests for co-occurrence counting, graph construction, metrics, and layout."""

from __future__ import annotations

from collections import Counter

import networkx as nx

from character_network.graph import (
    annotate_graph,
    build_graph,
    chapter_presence,
    count_cooccurrences,
    graph_metrics,
    layout_graph,
)
from character_network.textproc import Chapter


def test_count_cooccurrences_counts_each_pair_once_per_unit():
    text = "Elizabeth and Darcy talked. Jane and Bingley danced."
    second_start = text.index("Jane")
    segments = [
        (frozenset({"Darcy", "Elizabeth"}), 0, second_start),
        (frozenset({"Bingley", "Jane"}), second_start, len(text)),
        (frozenset({"Darcy", "Elizabeth"}), 0, second_start),
    ]
    characters, edges, count, examples = count_cooccurrences(segments, text)
    assert count == 3
    assert characters == Counter({"Darcy": 2, "Elizabeth": 2, "Jane": 1, "Bingley": 1})
    assert edges[("Darcy", "Elizabeth")] == 2
    assert edges[("Bingley", "Jane")] == 1
    assert examples[("Darcy", "Elizabeth")].startswith("Elizabeth and Darcy")
    assert "Jane" in examples[("Bingley", "Jane")]


def test_count_cooccurrences_truncates_long_examples():
    text = "word " * 200
    _, _, _, examples = count_cooccurrences(
        [(frozenset({"A", "B"}), 0, len(text))], text, example_characters=50
    )
    assert len(examples[("A", "B")]) <= 50


def test_build_graph_applies_thresholds_and_keeps_attributes():
    mentions = {"Elizabeth": 8, "Darcy": 7, "Jane": 5, "Bingley": 4, "Once": 1}
    edges = Counter(
        {
            ("Darcy", "Elizabeth"): 5,
            ("Elizabeth", "Jane"): 2,
            ("Bingley", "Jane"): 1,
        }
    )
    graph = build_graph(
        mentions,
        edges,
        minimum_edge_weight=2,
        minimum_mentions=2,
        maximum_characters=3,
        examples={("Darcy", "Elizabeth"): "They spoke at the assembly."},
    )
    assert set(graph.nodes) == {"Elizabeth", "Darcy", "Jane"}
    assert graph["Darcy"]["Elizabeth"]["weight"] == 5
    assert graph["Darcy"]["Elizabeth"]["example"].startswith("They spoke")
    assert graph.nodes["Elizabeth"]["frequency"] == 8


def test_build_graph_zero_maximum_keeps_every_character():
    mentions = {f"Person {index}": 2 for index in range(30)}
    edges = {("Person 0", f"Person {index}"): 2 for index in range(1, 30)}
    graph = build_graph(mentions, edges, maximum_characters=0)
    assert graph.number_of_nodes() == 30


def test_build_graph_removes_isolates():
    graph = build_graph({"A": 3, "B": 2, "Lonely": 4}, {("A", "B"): 2})
    assert "Lonely" not in graph.nodes


def test_build_graph_rejects_invalid_thresholds():
    for kwargs in ({"minimum_edge_weight": 0}, {"minimum_mentions": 0}):
        try:
            build_graph({"A": 1}, {}, **kwargs)
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")


def test_annotate_graph_adds_metrics_and_communities():
    graph = nx.Graph()
    graph.add_edge("Elizabeth", "Darcy", weight=5)
    graph.add_edge("Darcy", "Bingley", weight=3)
    graph.add_edge("Jane", "Bingley", weight=4)
    annotate_graph(graph)

    for node in graph.nodes:
        assert "weighted_degree" in graph.nodes[node]
        assert "community" in graph.nodes[node]
        assert "betweenness" in graph.nodes[node]
    assert graph.nodes["Darcy"]["weighted_degree"] == 8
    assert graph.nodes["Darcy"]["degree"] == 2
    assert graph.number_of_nodes() == 4


def test_graph_metrics_summarises_the_network():
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=2)
    graph.add_edge("B", "C", weight=4)
    metrics = graph_metrics(graph)
    assert metrics["nodes"] == 3
    assert metrics["edges"] == 2
    assert metrics["maximum_weight"] == 4
    assert metrics["average_weight"] == 3.0
    assert metrics["components"] == 1


def test_layout_is_deterministic_and_normalised():
    graph = nx.Graph()
    for index in range(6):
        graph.add_edge(f"N{index}", f"N{(index + 1) % 6}", weight=index + 1)
    first = layout_graph(graph, seed=42)
    second = layout_graph(graph, seed=42)
    assert first == second
    assert all(-1.01 <= x <= 1.01 and -1.01 <= y <= 1.01 for x, y in first.values())
    assert layout_graph(nx.Graph(), seed=42) == {}


def test_chapter_presence_builds_a_character_by_chapter_matrix():
    chapters = [Chapter(1, "Chapter I", 0, 100), Chapter(2, "Chapter II", 100, 200)]
    mentions = [
        type("M", (), {"name": "Elizabeth", "start_char": 10}),
        type("M", (), {"name": "Elizabeth", "start_char": 120}),
        type("M", (), {"name": "Darcy", "start_char": 130}),
    ]
    names, labels, matrix = chapter_presence(chapters, mentions, character_limit=5)
    assert labels == ["Chapter I", "Chapter II"]
    assert names == ["Elizabeth", "Darcy"]
    assert matrix[0] == [1, 1]
    assert matrix[1] == [0, 1]
