"""Interactive Plotly rendering, static PNG export, and download tables."""

from __future__ import annotations

import io
import math
import re
from typing import Any, Mapping, Sequence

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from .graph import layout_graph

#: Colour-blind-safe qualitative palette (Okabe–Ito).
COMMUNITY_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#8C6D31",
    "#666666",
    "#4B0082",
    "#2E8B57",
    "#B22222",
    "#4682B4",
]

#: Sequential scale for the degree-based colouring.
DEGREE_COLORS = [[0.0, "#8BD3DD"], [0.5, "#5B8DEF"], [1.0, "#6941C6"]]

BACKGROUND = "rgba(0,0,0,0)"


def node_size(frequency: int, maximum: int) -> float:
    """Map a mention count to a readable marker size."""

    ratio = frequency / max(maximum, 1)
    return 16.0 + 46.0 * math.sqrt(ratio)


def edge_width(weight: int, maximum: int) -> float:
    """Map a co-occurrence count to a line width."""

    ratio = weight / max(maximum, 1)
    return 0.6 + 6.4 * math.sqrt(ratio)


def _truncate(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def create_network_figure(
    graph: nx.Graph,
    *,
    positions: Mapping[str, tuple[float, float]] | None = None,
    color_by: str = "degree",
    show_labels: bool = True,
    height: int = 680,
    seed: int = 42,
) -> go.Figure:
    """Create an interactive character network.

    All edges are drawn in a single trace with ``None`` separators, so a full novel
    with hundreds of relationships still renders quickly. Edge weights are carried by
    line width and by an invisible midpoint marker that shows the count and a short
    quotation from the text.
    """

    if graph.number_of_nodes() == 0:
        raise ValueError("The graph has no characters to draw.")

    layout = dict(positions) if positions else layout_graph(graph, seed=seed)
    figure = go.Figure()

    maximum_weight = max(
        (data.get("weight", 1) for _, _, data in graph.edges(data=True)), default=1
    )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target, data in graph.edges(data=True):
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    if edge_x:
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 1, "color": "rgba(91,107,135,0.30)"},
                hoverinfo="skip",
                showlegend=False,
                name="relationships",
            )
        )

    # One extra point per edge gives a reliable hover target even on thin lines.
    mid_x, mid_y, mid_text, mid_size = [], [], [], []
    for source, target, data in graph.edges(data=True):
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        weight = int(data.get("weight", 1))
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        example = _truncate(data.get("example", ""))
        text = f"<b>{source}</b> ↔ <b>{target}</b><br>Co-occurrence count: {weight}"
        if example:
            text += f"<br><i>“{example}”</i>"
        mid_text.append(text)
        mid_size.append(8 + 6 * math.sqrt(weight / max(maximum_weight, 1)))

    if mid_x:
        figure.add_trace(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="markers",
                marker={
                    "size": mid_size,
                    "color": "rgba(91,107,135,0.01)",
                    "line": {"width": 0},
                },
                hovertext=mid_text,
                hoverinfo="text",
                showlegend=False,
                name="co-occurrence",
            )
        )

    names = list(graph.nodes())
    frequencies = [graph.nodes[name].get("frequency", 1) for name in names]
    maximum_frequency = max(frequencies, default=1)
    sizes = [node_size(value, maximum_frequency) for value in frequencies]
    hover = [
        (
            f"<b>{name}</b><br>Mentions: {graph.nodes[name].get('frequency', 0)}"
            f"<br>Connections: {graph.nodes[name].get('degree', 0)}"
            f"<br>Weighted degree: {graph.nodes[name].get('weighted_degree', 0):.0f}"
        )
        for name in names
    ]

    if color_by == "community":
        communities: dict[int, list[str]] = {}
        for name in names:
            communities.setdefault(int(graph.nodes[name].get("community", 0)), []).append(name)
        ordered = sorted(communities.items(), key=lambda item: len(item[1]), reverse=True)
        for index, (community, members) in enumerate(ordered):
            color = COMMUNITY_COLORS[index % len(COMMUNITY_COLORS)]
            figure.add_trace(
                go.Scatter(
                    x=[layout[name][0] for name in members],
                    y=[layout[name][1] for name in members],
                    mode="markers+text" if show_labels else "markers",
                    text=members,
                    textposition="top center",
                    textfont={"size": 11, "color": "#17223B"},
                    hovertext=[hover[names.index(name)] for name in members],
                    hoverinfo="text",
                    marker={
                        "size": [sizes[names.index(name)] for name in members],
                        "color": color,
                        "line": {"width": 2, "color": "#FFFFFF"},
                    },
                    name=f"Group {index + 1} ({len(members)})",
                    legendgroup=f"group-{index}",
                )
            )
    else:
        degrees = [graph.nodes[name].get("weighted_degree", 0) for name in names]
        figure.add_trace(
            go.Scatter(
                x=[layout[name][0] for name in names],
                y=[layout[name][1] for name in names],
                mode="markers+text" if show_labels else "markers",
                text=names,
                textposition="top center",
                textfont={"size": 11, "color": "#17223B"},
                hovertext=hover,
                hoverinfo="text",
                marker={
                    "size": sizes,
                    "color": degrees,
                    "colorscale": DEGREE_COLORS,
                    "showscale": True,
                    "colorbar": {"title": {"text": "Weighted<br>degree"}, "thickness": 14},
                    "line": {"width": 2, "color": "#FFFFFF"},
                },
                showlegend=False,
                name="characters",
            )
        )

    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        hovermode="closest",
        hoverlabel={"bgcolor": "#FFFFFF", "font": {"color": "#17223B"}},
        xaxis={"visible": False, "range": [-1.35, 1.35], "fixedrange": False},
        yaxis={"visible": False, "range": [-1.30, 1.30], "fixedrange": False},
        dragmode="pan",
        legend={"bgcolor": "rgba(255,255,255,0.75)", "borderwidth": 0},
        showlegend=color_by == "community" and graph.number_of_nodes() > 3,
    )
    return figure


def figure_to_html(figure: go.Figure, *, title: str = "Character relationship network") -> str:
    """Return a standalone, offline HTML page for the figure."""

    html = figure.to_html(
        full_html=True,
        include_plotlyjs="inline",
        config={"scrollZoom": True, "displaylogo": False},
        div_id="character-network",
        default_width="100%",
        default_height="720px",
    )
    # Plotly omits a document title; add one so the saved file is labelled in a tab.
    html = re.sub(r"<title>.*?</title>", "", html, flags=re.DOTALL)
    return html.replace("<head>", f"<head><title>{title}</title>", 1)


def render_png(
    graph: nx.Graph,
    *,
    positions: Mapping[str, tuple[float, float]] | None = None,
    color_by: str = "community",
    title: str | None = None,
    dpi: int = 200,
    seed: int = 42,
) -> bytes:
    """Render a static PNG of the network with matplotlib.

    A separate renderer is used on purpose: it works fully offline, needs no browser
    engine, and produces a print-quality image for slides and reports.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError as exc:  # pragma: no cover - matplotlib is a declared dependency
        raise RuntimeError(
            "PNG export needs matplotlib. Run: python -m pip install -r requirements.txt"
        ) from exc

    if graph.number_of_nodes() == 0:
        raise ValueError("The graph has no characters to draw.")

    layout = dict(positions) if positions else layout_graph(graph, seed=seed)
    maximum_weight = max(
        (data.get("weight", 1) for _, _, data in graph.edges(data=True)), default=1
    )
    frequencies = [graph.nodes[name].get("frequency", 1) for name in graph.nodes()]
    maximum_frequency = max(frequencies, default=1)

    figure = plt.figure(figsize=(12, 8.5), dpi=dpi)
    axis = figure.add_subplot(111)
    axis.set_axis_off()

    segments, widths = [], []
    for source, target, data in graph.edges(data=True):
        segments.append([layout[source], layout[target]])
        widths.append(edge_width(int(data.get("weight", 1)), maximum_weight) * 1.1)
    if segments:
        axis.add_collection(
            LineCollection(segments, colors="#5B6B8759", linewidths=widths, zorder=1)
        )

    names = list(graph.nodes())
    if color_by == "community":
        colors = [
            COMMUNITY_COLORS[int(graph.nodes[name].get("community", 0)) % len(COMMUNITY_COLORS)]
            for name in names
        ]
    else:
        weights = [graph.nodes[name].get("weighted_degree", 0) for name in names]
        top = max(weights) or 1
        colors = [
            (0.16 + 0.62 * (weight / top), 0.28 + 0.35 * (weight / top), 0.82 - 0.35 * (weight / top))
            for weight in weights
        ]

    axis.scatter(
        [layout[name][0] for name in names],
        [layout[name][1] for name in names],
        s=[node_size(value, maximum_frequency) * 2.6 for value in frequencies],
        c=colors,
        edgecolors="#FFFFFF",
        linewidths=1.6,
        zorder=2,
    )
    # Push labels away from the centre so that hubs in a dense graph do not stack
    # their names on top of each other.
    label_size = 8.5 if len(names) <= 20 else (7.2 if len(names) <= 35 else 6.2)
    for name in names:
        x, y = layout[name]
        length = math.hypot(x, y) or 1.0
        offset_x, offset_y = x / length, y / length
        axis.annotate(
            name,
            xy=(x, y),
            xytext=(10 * offset_x, 8 * offset_y),
            textcoords="offset points",
            ha="left" if offset_x > 0.25 else ("right" if offset_x < -0.25 else "center"),
            va="center",
            fontsize=label_size,
            color="#17223B",
            zorder=3,
        )

    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.20, 1.20)
    axis.set_aspect("equal")
    axis.set_title(
        title
        or (
            f"Character co-occurrence network · {graph.number_of_nodes()} characters · "
            f"{graph.number_of_edges()} relationships"
        ),
        fontsize=12,
        color="#17223B",
    )
    figure.text(
        0.5,
        0.015,
        "Edge width = co-occurrence count. An edge means two names appear near each other; "
        "it does not identify the nature of the relationship.",
        ha="center",
        fontsize=7.5,
        color="#5B6B87",
    )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def chapter_heatmap(
    names: Sequence[str],
    labels: Sequence[str],
    matrix: Sequence[Sequence[int]],
) -> go.Figure:
    """Return a character-by-chapter mention heatmap."""

    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=list(labels),
            y=list(names),
            colorscale="Blues",
            hoverongaps=False,
            hovertemplate="%{y}<br>%{x}: %{z} mentions<extra></extra>",
            colorbar={"title": {"text": "Mentions"}, "thickness": 12},
        )
    )
    figure.update_layout(
        height=max(320, 22 * len(names) + 140),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        xaxis={"title": None, "tickangle": -45, "tickfont": {"size": 10}},
        yaxis={"autorange": "reversed", "tickfont": {"size": 10}},
    )
    return figure


def node_table(graph: nx.Graph) -> pd.DataFrame:
    """Return character metrics in a stable, downloadable table."""

    rows = [
        {
            "character": name,
            "mentions": int(data.get("frequency", 0)),
            "connections": int(data.get("degree", 0)),
            "weighted_degree": int(data.get("weighted_degree", 0)),
            "betweenness": data.get("betweenness", 0.0),
            "community": int(data.get("community", 0)) + 1,
        }
        for name, data in graph.nodes(data=True)
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["mentions", "weighted_degree", "character"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def edge_table(graph: nx.Graph) -> pd.DataFrame:
    """Return network edges in descending weight order."""

    rows = [
        {
            "source": source,
            "target": target,
            "cooccurrence_count": int(data.get("weight", 0)),
            "example": _truncate(data.get("example", ""), 200),
        }
        for source, target, data in graph.edges(data=True)
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["cooccurrence_count", "source", "target"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def merge_table(merges: Sequence[Any]) -> pd.DataFrame:
    """Return the audit trail of automatic name merges."""

    rows = [
        {
            "merged variant": merge.variant,
            "merged into": merge.canonical,
            "variant mentions": merge.mentions,
            "rule": merge.reason,
        }
        for merge in merges
    ]
    frame = pd.DataFrame(rows)
    return frame
