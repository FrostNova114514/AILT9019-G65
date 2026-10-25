"""Tests for the non-overlapping canvas layout and label placement."""

from __future__ import annotations

import math

import networkx as nx
import plotly.graph_objects as go

from character_network.graph import annotate_graph
from character_network.visualization import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    create_network_figure,
    display_layout,
    node_sizes,
)


def _graph(size: int = 8) -> nx.Graph:
    graph = nx.Graph()
    for index in range(size):
        graph.add_edge(f"Person {index}", f"Person {(index + 1) % size}", weight=index + 1)
    for index, node in enumerate(graph.nodes):
        graph.nodes[node]["frequency"] = 20 - index
    return annotate_graph(graph)


def _minimum_gap(graph: nx.Graph, positions: dict[str, tuple[float, float]]) -> float:
    sizes = node_sizes(graph)
    names = list(positions)
    gap = math.inf
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            distance = math.dist(positions[first], positions[second])
            gap = min(gap, distance - (sizes[first] + sizes[second]) / 2)
    return gap


def test_layout_keeps_circles_apart():
    graph = _graph(9)
    positions = display_layout(graph)
    assert _minimum_gap(graph, positions) >= 11.0


def test_layout_keeps_every_circle_inside_the_canvas():
    graph = _graph(12)
    positions = display_layout(graph)
    sizes = node_sizes(graph)
    for name, (x, y) in positions.items():
        radius = sizes[name] / 2
        assert -1 <= x - radius and x + radius <= CANVAS_WIDTH + 1
        assert -1 <= y - radius and y + radius <= CANVAS_HEIGHT + 1


def test_layout_is_deterministic_and_handles_tiny_graphs():
    graph = _graph(6)
    assert display_layout(graph) == display_layout(graph)
    assert display_layout(nx.Graph()) == {}

    single = nx.Graph()
    single.add_node("Only", frequency=3)
    positions = display_layout(single)
    assert positions["Only"] == (CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)


def test_figure_uses_canvas_coordinates_and_hidden_axes():
    graph = _graph(8)
    figure = create_network_figure(graph)
    assert figure.layout.xaxis.range == (0, figure.layout.xaxis.range[1])
    assert figure.layout.xaxis.visible is False
    assert figure.layout.yaxis.scaleanchor == "x"


def test_figure_labels_are_split_by_placement():
    graph = _graph(10)
    figure = create_network_figure(graph)
    label_traces = [trace for trace in figure.data if trace.mode == "text"]
    assert label_traces
    assert all(str(trace.name).startswith("labels-") for trace in label_traces)
    positions = {trace.textposition for trace in label_traces}
    assert positions
    assert all(isinstance(trace.text, tuple) for trace in label_traces)


def test_figure_keeps_one_edge_trace():
    graph = _graph(10)
    figure = create_network_figure(graph)
    assert len([trace for trace in figure.data if trace.mode == "lines"]) == 1
    assert isinstance(figure, go.Figure)


def test_figure_can_hide_labels():
    figure = create_network_figure(_graph(5), show_labels=False)
    assert not [trace for trace in figure.data if trace.mode == "text"]
