"""Interactive Plotly rendering, static PNG export, and download tables.

Two coordinate systems are used deliberately. :func:`character_network.graph.layout_graph`
returns a normalised spring layout. :func:`display_layout` then places that layout on a
fixed pixel canvas and pushes circles apart until they no longer intersect, so node size
(marker pixels) and node distance (canvas pixels) share one unit. The same canvas
positions feed both the Plotly figure and the matplotlib PNG, which keeps the two
exports identical.
"""

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

#: Nominal drawing area in pixels. Marker sizes are pixels and node positions are
#: canvas pixels, so a separation of 80 canvas units is 80 marker-pixels.
CANVAS_WIDTH = 980.0
CANVAS_HEIGHT = 700.0

#: Candidate label positions in preference order, with a unit direction vector.
LABEL_SLOTS = (
    ("top center", 0.0, 1.0),
    ("bottom center", 0.0, -1.0),
    ("middle right", 1.0, 0.0),
    ("middle left", -1.0, 0.0),
    ("top right", 0.8, 0.7),
    ("top left", -0.8, 0.7),
    ("bottom right", 0.8, -0.7),
    ("bottom left", -0.8, -0.7),
)


def node_size(frequency: int, maximum: int) -> float:
    """Map a mention count to a marker diameter in pixels."""

    ratio = frequency / max(maximum, 1)
    return 15.0 + 34.0 * math.sqrt(ratio)


def edge_width(weight: int, maximum: int) -> float:
    """Map a co-occurrence count to a line width."""

    ratio = weight / max(maximum, 1)
    return 0.6 + 6.4 * math.sqrt(ratio)


def node_sizes(graph: nx.Graph) -> dict[str, float]:
    """Return a marker diameter for every node."""

    maximum = max((data.get("frequency", 1) for _, data in graph.nodes(data=True)), default=1)
    return {
        name: node_size(int(data.get("frequency", 1)), maximum)
        for name, data in graph.nodes(data=True)
    }


def _map_to_canvas(
    positions: Mapping[str, tuple[float, float]],
    width: float,
    height: float,
    margin: float,
) -> dict[str, tuple[float, float]]:
    """Stretch a normalised layout to fill the canvas (each axis independently)."""

    if not positions:
        return {}
    xs = [value[0] for value in positions.values()]
    ys = [value[1] for value in positions.values()]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)
    usable_w = max(width - 2 * margin, 1.0)
    usable_h = max(height - 2 * margin, 1.0)
    return {
        name: (
            margin + (x - min(xs)) / span_x * usable_w,
            margin + (y - min(ys)) / span_y * usable_h,
        )
        for name, (x, y) in positions.items()
    }


def _separate(
    positions: Mapping[str, tuple[float, float]],
    radii: Mapping[str, float],
    *,
    width: float,
    height: float,
    padding: float,
    iterations: int,
) -> dict[str, tuple[float, float]]:
    """Push overlapping circles apart until their edges keep ``padding`` pixels apart."""

    names = list(positions)
    px = {name: [positions[name][0], positions[name][1]] for name in names}

    for _ in range(iterations):
        moved = 0
        for first_index in range(len(names)):
            first = names[first_index]
            for second_index in range(first_index + 1, len(names)):
                second = names[second_index]
                delta_x = px[second][0] - px[first][0]
                delta_y = px[second][1] - px[first][1]
                required = radii[first] + radii[second] + padding
                squared = delta_x * delta_x + delta_y * delta_y
                if squared >= required * required:
                    continue
                distance = math.sqrt(squared)
                if distance < 1e-6:
                    delta_x, delta_y, distance = 1.0, 0.0, 1.0
                push = (required - distance) / 2
                unit_x = delta_x / distance
                unit_y = delta_y / distance
                px[first][0] -= unit_x * push
                px[first][1] -= unit_y * push
                px[second][0] += unit_x * push
                px[second][1] += unit_y * push
                moved += 1
        # Keep every circle fully inside the canvas.
        for name in names:
            radius = radii[name] + 4.0
            px[name][0] = min(max(px[name][0], radius), width - radius)
            px[name][1] = min(max(px[name][1], radius), height - radius)
        if not moved:
            break

    return {name: (px[name][0], px[name][1]) for name in names}


def display_layout(
    graph: nx.Graph,
    *,
    seed: int = 42,
    width: float = CANVAS_WIDTH,
    height: float = CANVAS_HEIGHT,
    padding: float = 12.0,
) -> dict[str, tuple[float, float]]:
    """Return canvas positions whose circles do not overlap.

    The spring layout decides the overall shape; this function only guarantees that
    every node keeps a readable distance from its neighbours.
    """

    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes)): (width / 2, height / 2)}

    base = layout_graph(graph, seed=seed)
    canvas = _map_to_canvas(base, width, height, margin=80.0)
    radii = {name: size / 2 for name, size in node_sizes(graph).items()}
    iterations = 300 if len(radii) <= 30 else 140
    return _separate(
        canvas,
        radii,
        width=width,
        height=height,
        padding=padding,
        iterations=iterations,
    )


def _label_box(
    name: str,
    position: tuple[float, float],
    radius: float,
    direction: tuple[float, float],
    *,
    font_size: float,
    gap: float,
) -> tuple[float, float, float, float]:
    """Return the (x0, y0, x1, y1) box a label would occupy."""

    length = math.hypot(*direction) or 1.0
    unit_x, unit_y = direction[0] / length, direction[1] / length
    centre_x = position[0] + unit_x * (radius + gap)
    centre_y = position[1] + unit_y * (radius + gap)
    text_width = font_size * 0.62 * len(name) + 8
    text_height = font_size * 1.5
    return (
        centre_x - text_width / 2,
        centre_y - text_height / 2,
        centre_x + text_width / 2,
        centre_y + text_height / 2,
    )


def _box_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _place_labels(
    names: Sequence[str],
    positions: Mapping[str, tuple[float, float]],
    sizes: Mapping[str, float],
    *,
    font_size: float,
    gap: float,
    width: float,
    height: float,
) -> dict[str, str]:
    """Choose a non-overlapping label slot for each node, largest node first."""

    circles = [(positions[name][0], positions[name][1], sizes[name] / 2) for name in names]
    placed: list[tuple[float, float, float, float]] = []
    slots: dict[str, str] = {}

    for name in sorted(names, key=lambda item: (-sizes[item], item)):
        position = positions[name]
        radius = sizes[name] / 2
        best_slot = "top center"
        best_score = math.inf
        for slot, direction_x, direction_y in LABEL_SLOTS:
            box = _label_box(
                name,
                position,
                radius,
                (direction_x, direction_y),
                font_size=font_size,
                gap=gap,
            )
            score = 0.0
            for other in placed:
                score += _box_overlap(box, other)
            for circle_x, circle_y, circle_radius in circles:
                if (circle_x, circle_y, circle_radius) == (position[0], position[1], radius):
                    continue
                nearest_x = min(max(circle_x, box[0]), box[2])
                nearest_y = min(max(circle_y, box[1]), box[3])
                if math.hypot(circle_x - nearest_x, circle_y - nearest_y) < circle_radius + 2:
                    score += 400.0
            if box[0] < 2 or box[1] < 2 or box[2] > width - 2 or box[3] > height - 2:
                score += 300.0
            if score == 0:
                best_slot = slot
                break
            if score < best_score:
                best_score, best_slot = score, slot
        slots[name] = best_slot
        direction_x, direction_y = next(
            (dx, dy) for label, dx, dy in LABEL_SLOTS if label == best_slot
        )
        placed.append(
            _label_box(
                name, position, radius, (direction_x, direction_y), font_size=font_size, gap=gap
            )
        )
    return slots


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
    height: int = 760,
    seed: int = 42,
) -> go.Figure:
    """Create an interactive character network with non-overlapping nodes.

    All edges are drawn in one trace with ``None`` separators, so a full novel with
    hundreds of relationships still renders quickly. Every edge also gets an invisible
    midpoint marker that shows the count and a short quotation on hover.
    """

    if graph.number_of_nodes() == 0:
        raise ValueError("The graph has no characters to draw.")

    layout = dict(positions) if positions else display_layout(graph, seed=seed)
    sizes = node_sizes(graph)
    font_size = 11.0 if len(sizes) <= 30 else 10.0

    width = max(max(x for x, _ in layout.values()), CANVAS_WIDTH) + 20
    height_units = max(max(y for _, y in layout.values()), CANVAS_HEIGHT) + 20

    figure = go.Figure()
    maximum_weight = max(
        (data.get("weight", 1) for _, _, data in graph.edges(data=True)), default=1
    )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
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
                line={"width": 1, "color": "rgba(91,107,135,0.32)"},
                hoverinfo="skip",
                showlegend=False,
                name="relationships",
            )
        )

    midpoint_x, midpoint_y, midpoint_text, midpoint_size = [], [], [], []
    for source, target, data in graph.edges(data=True):
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        weight = int(data.get("weight", 1))
        midpoint_x.append((x0 + x1) / 2)
        midpoint_y.append((y0 + y1) / 2)
        example = _truncate(data.get("example", ""))
        text = f"<b>{source}</b> ↔ <b>{target}</b><br>Co-occurrence count: {weight}"
        if example:
            text += f"<br><i>“{example}”</i>"
        midpoint_text.append(text)
        midpoint_size.append(14 + 10 * math.sqrt(weight / max(maximum_weight, 1)))

    if midpoint_x:
        figure.add_trace(
            go.Scatter(
                x=midpoint_x,
                y=midpoint_y,
                mode="markers",
                marker={
                    "size": midpoint_size,
                    "color": "rgba(91,107,135,0.01)",
                    "line": {"width": 0},
                },
                hovertext=midpoint_text,
                hoverinfo="text",
                showlegend=False,
                name="co-occurrence",
            )
        )

    names = list(graph.nodes())
    hover = [
        (
            f"<b>{name}</b><br>Mentions: {graph.nodes[name].get('frequency', 0)}"
            f"<br>Connections: {graph.nodes[name].get('degree', 0)}"
            f"<br>Weighted degree: {graph.nodes[name].get('weighted_degree', 0):.0f}"
        )
        for name in names
    ]
    marker_sizes = [sizes[name] for name in names]

    if color_by == "community":
        communities: dict[int, list[str]] = {}
        for name in names:
            communities.setdefault(int(graph.nodes[name].get("community", 0)), []).append(name)
        ordered = sorted(communities.items(), key=lambda item: len(item[1]), reverse=True)
        for index, (community, members) in enumerate(ordered):
            colour = COMMUNITY_COLORS[index % len(COMMUNITY_COLORS)]
            figure.add_trace(
                go.Scatter(
                    x=[layout[name][0] for name in members],
                    y=[layout[name][1] for name in members],
                    mode="markers",
                    hovertext=[hover[names.index(name)] for name in members],
                    hoverinfo="text",
                    marker={
                        "size": [sizes[name] for name in members],
                        "color": colour,
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
                mode="markers",
                hovertext=hover,
                hoverinfo="text",
                marker={
                    "size": marker_sizes,
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

    if show_labels:
        slots = _place_labels(
            names,
            layout,
            sizes,
            font_size=font_size,
            gap=10.0,
            width=width,
            height=height_units,
        )
        grouped: dict[str, list[str]] = {}
        for name, slot in slots.items():
            grouped.setdefault(slot, []).append(name)
        for slot, members in grouped.items():
            figure.add_trace(
                go.Scatter(
                    x=[layout[name][0] for name in members],
                    y=[layout[name][1] for name in members],
                    mode="text",
                    text=members,
                    textposition=slot,
                    textfont={"size": font_size, "color": "#17223B"},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"labels-{slot}",
                )
            )

    figure.update_layout(
        height=height,
        margin={"l": 6, "r": 6, "t": 16, "b": 6},
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        hovermode="closest",
        hoverlabel={"bgcolor": "#FFFFFF", "font": {"color": "#17223B"}},
        xaxis={"visible": False, "range": [0, width], "fixedrange": False},
        yaxis={
            "visible": False,
            "range": [0, height_units],
            "fixedrange": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        dragmode="pan",
        legend={"bgcolor": "rgba(255,255,255,0.75)", "borderwidth": 0},
        showlegend=color_by == "community" and graph.number_of_nodes() > 3,
        uirevision="keep",
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
        default_height="780px",
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
    """Render a static PNG with the same canvas layout as the interactive figure.

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

    layout = dict(positions) if positions else display_layout(graph, seed=seed)
    sizes = node_sizes(graph)
    maximum_weight = max(
        (data.get("weight", 1) for _, _, data in graph.edges(data=True)), default=1
    )

    figure_width, figure_height = 13.0, 9.3
    canvas_width = max(max(x for x, _ in layout.values()) + 20, CANVAS_WIDTH)
    canvas_height = max(max(y for _, y in layout.values()) + 20, CANVAS_HEIGHT)

    figure = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)
    axis = figure.add_axes((0.02, 0.06, 0.96, 0.86))
    axis.set_axis_off()

    # One canvas unit becomes this many points, so PNG node sizes match the layout.
    points_per_unit = figure_width * 0.96 * 72 / canvas_width

    segments, widths = [], []
    for source, target, data in graph.edges(data=True):
        segments.append([layout[source], layout[target]])
        widths.append(edge_width(int(data.get("weight", 1)), maximum_weight) * 0.9)
    if segments:
        axis.add_collection(
            LineCollection(segments, colors="#5B6B8759", linewidths=widths, zorder=1)
        )

    names = list(graph.nodes())
    if color_by == "community":
        colours = [
            COMMUNITY_COLORS[int(graph.nodes[name].get("community", 0)) % len(COMMUNITY_COLORS)]
            for name in names
        ]
    else:
        weights = [graph.nodes[name].get("weighted_degree", 0) for name in names]
        top = max(weights) or 1
        colours = [
            (0.16 + 0.62 * (weight / top), 0.28 + 0.35 * (weight / top), 0.82 - 0.35 * (weight / top))
            for weight in weights
        ]

    axis.scatter(
        [layout[name][0] for name in names],
        [layout[name][1] for name in names],
        s=[(sizes[name] * points_per_unit) ** 2 for name in names],
        c=colours,
        edgecolors="#FFFFFF",
        linewidths=1.6,
        zorder=2,
    )

    label_size = 9.0 if len(names) <= 20 else (7.6 if len(names) <= 35 else 6.6)
    placements = _place_labels(
        names,
        layout,
        sizes,
        font_size=label_size,
        gap=8.0,
        width=canvas_width,
        height=canvas_height,
    )
    for name in names:
        slot = placements[name]
        direction_x, direction_y = next(
            (dx, dy) for label, dx, dy in LABEL_SLOTS if label == slot
        )
        length = math.hypot(direction_x, direction_y) or 1.0
        gap_units = (sizes[name] / 2 + 12) * 1.0
        offset_units_x = direction_x / length * gap_units
        offset_units_y = direction_y / length * gap_units
        axis.annotate(
            name,
            xy=(layout[name][0] + offset_units_x, layout[name][1] + offset_units_y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=label_size,
            color="#17223B",
            zorder=3,
        )

    axis.set_xlim(0, canvas_width)
    axis.set_ylim(0, canvas_height)
    axis.set_aspect("equal")
    figure.suptitle(
        title
        or (
            f"Character co-occurrence network · {graph.number_of_nodes()} characters · "
            f"{graph.number_of_edges()} relationships"
        ),
        fontsize=14,
        color="#17223B",
        y=0.965,
    )
    figure.text(
        0.5,
        0.022,
        "Edge width = co-occurrence count. An edge means two names appear near each other; "
        "it does not identify the nature of the relationship.",
        ha="center",
        fontsize=7.5,
        color="#5B6B87",
    )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
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
