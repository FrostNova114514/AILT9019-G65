"""Plotly rendering and export-table helpers."""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def create_network_figure(graph: nx.Graph) -> go.Figure:
    """Create an interactive, deterministic character-network visualization."""

    positions = nx.spring_layout(graph, seed=42, weight="weight", k=None)
    maximum_weight = max(
        (data["weight"] for _, _, data in graph.edges(data=True)),
        default=1,
    )

    figure = go.Figure()
    for source, target, data in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        weight = data["weight"]
        width = 1.0 + 6.0 * math.sqrt(weight / maximum_weight)
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": width, "color": "rgba(91, 107, 135, 0.38)"},
                hoverinfo="text",
                text=[
                    f"{source} ↔ {target}<br>Co-occurrence count: {weight}",
                    f"{source} ↔ {target}<br>Co-occurrence count: {weight}",
                ],
                showlegend=False,
            )
        )

    names = list(graph.nodes())
    degrees = dict(graph.degree(weight="weight"))
    frequencies = [graph.nodes[name].get("frequency", 1) for name in names]
    maximum_frequency = max(frequencies, default=1)
    sizes = [18 + 34 * math.sqrt(value / maximum_frequency) for value in frequencies]
    hover_text = [
        (
            f"<b>{name}</b><br>Mentions: {graph.nodes[name].get('frequency', 0)}"
            f"<br>Weighted degree: {degrees.get(name, 0):.0f}"
        )
        for name in names
    ]

    figure.add_trace(
        go.Scatter(
            x=[positions[name][0] for name in names],
            y=[positions[name][1] for name in names],
            mode="markers+text",
            text=names,
            textposition="top center",
            textfont={"size": 12, "color": "#17223b"},
            hovertext=hover_text,
            hoverinfo="text",
            marker={
                "size": sizes,
                "color": [degrees.get(name, 0) for name in names],
                "colorscale": [[0, "#8bd3dd"], [0.5, "#5b8def"], [1, "#6941c6"]],
                "showscale": True,
                "colorbar": {"title": {"text": "Weighted<br>degree"}, "thickness": 14},
                "line": {"width": 2, "color": "#ffffff"},
            },
            showlegend=False,
        )
    )

    figure.update_layout(
        height=680,
        margin={"l": 15, "r": 15, "t": 30, "b": 15},
        paper_bgcolor="#f7f8fc",
        plot_bgcolor="#f7f8fc",
        hovermode="closest",
        xaxis={"visible": False},
        yaxis={"visible": False},
        dragmode="pan",
    )
    return figure


def node_table(graph: nx.Graph) -> pd.DataFrame:
    """Return character metrics in a stable, downloadable table."""

    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    rows = [
        {
            "character": name,
            "mentions": graph.nodes[name].get("frequency", 0),
            "connections": degree[name],
            "weighted_degree": weighted_degree[name],
        }
        for name in graph.nodes()
    ]
    return pd.DataFrame(rows).sort_values(
        ["mentions", "weighted_degree", "character"],
        ascending=[False, False, True],
    )


def edge_table(graph: nx.Graph) -> pd.DataFrame:
    """Return network edges in descending weight order."""

    rows = [
        {"source": source, "target": target, "cooccurrence_count": data["weight"]}
        for source, target, data in graph.edges(data=True)
    ]
    return pd.DataFrame(rows).sort_values(
        ["cooccurrence_count", "source", "target"],
        ascending=[False, True, True],
    )
