"""Tests for the Plotly figure, PNG export, and download tables."""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go

from character_network.graph import annotate_graph
from character_network.names import Merge
from character_network.visualization import (
    chapter_heatmap,
    create_network_figure,
    edge_table,
    figure_to_html,
    merge_table,
    node_table,
    render_png,
)


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("Elizabeth Bennet", "Fitzwilliam Darcy", weight=6, example="They danced.")
    graph.add_edge("Elizabeth Bennet", "Jane Bennet", weight=4, example="Sisters talked.")
    graph.add_edge("Jane Bennet", "Charles Bingley", weight=3, example="They met.")
    for name in ("Elizabeth Bennet", "Fitzwilliam Darcy", "Jane Bennet", "Charles Bingley"):
        graph.nodes[name]["frequency"] = 10
    return annotate_graph(graph)


def test_network_figure_batches_edges_and_draws_nodes():
    figure = create_network_figure(_graph())
    assert isinstance(figure, go.Figure)
    modes = [trace.mode for trace in figure.data]
    assert modes.count("lines") == 1  # all edges in a single trace
    assert any("markers" in mode for mode in modes)
    assert figure.layout.dragmode == "pan"
    assert figure.layout.xaxis.visible is False


def test_network_figure_can_colour_by_community():
    figure = create_network_figure(_graph(), color_by="community")
    assert any(trace.name and str(trace.name).startswith("Group") for trace in figure.data)


def test_network_figure_requires_characters():
    try:
        create_network_figure(nx.Graph())
    except ValueError as exc:
        assert "no characters" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_png_export_returns_a_real_image():
    payload = render_png(_graph())
    assert payload.startswith(b"\x89PNG")
    assert len(payload) > 5000


def test_html_export_is_standalone_and_labelled():
    html = figure_to_html(create_network_figure(_graph()), title="My network")
    assert "<html" in html.lower()
    assert "plotly" in html.lower()
    assert "<title>My network</title>" in html


def test_node_table_has_metrics_and_is_sorted_by_mentions():
    frame = node_table(_graph())
    assert list(frame.columns) == [
        "character",
        "mentions",
        "connections",
        "weighted_degree",
        "betweenness",
        "community",
    ]
    assert frame["mentions"].is_monotonic_decreasing


def test_edge_table_is_sorted_by_weight():
    frame = edge_table(_graph())
    assert list(frame.columns) == ["source", "target", "cooccurrence_count", "example"]
    assert frame["cooccurrence_count"].is_monotonic_decreasing
    assert frame.iloc[0]["cooccurrence_count"] == 6


def test_merge_table_lists_every_decision():
    frame = merge_table([Merge("Darcy", "Fitzwilliam Darcy", 120, "unique partial-name match")])
    assert frame.iloc[0]["merged variant"] == "Darcy"
    assert frame.iloc[0]["merged into"] == "Fitzwilliam Darcy"


def test_chapter_heatmap_builds_a_heatmap_trace():
    figure = chapter_heatmap(
        ["Elizabeth", "Darcy"],
        ["Chapter I", "Chapter II"],
        [[3, 5], [1, 2]],
    )
    assert isinstance(figure.data[0], go.Heatmap)
    assert figure.data[0].z[1][1] == 2
