"""Graph construction, network metrics, and deterministic layouts."""

from __future__ import annotations

import bisect
import math
from collections import Counter
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import networkx as nx


def count_cooccurrences(
    segments: Iterable[tuple[frozenset[str], int, int]],
    text: str,
    *,
    example_characters: int = 240,
) -> tuple[Counter[str], Counter[tuple[str, str]], int, dict[tuple[str, str], str]]:
    """Count character presence and unordered pairs per unit, keeping one example each.

    ``segments`` yields ``(names, start_char, end_char)``. The first unit in which a
    pair co-occurs is stored as a short quotation so that every edge can be checked
    against the source text.
    """

    characters: Counter[str] = Counter()
    edges: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}
    segment_count = 0

    for names, start, end in segments:
        segment_count += 1
        ordered = sorted(names, key=str.casefold)
        characters.update(ordered)
        for pair in combinations(ordered, 2):
            edges[pair] += 1
            if pair not in examples:
                snippet = " ".join(text[start:end].split())
                if len(snippet) > example_characters:
                    snippet = snippet[: example_characters - 1].rstrip() + "…"
                examples[pair] = snippet
    return characters, edges, segment_count, examples


def build_graph(
    mention_counts: Mapping[str, int],
    edge_counts: Mapping[tuple[str, str], int],
    *,
    minimum_edge_weight: int = 1,
    minimum_mentions: int = 1,
    maximum_characters: int = 0,
    examples: Mapping[tuple[str, str], str] | None = None,
    keep_isolates: bool = False,
) -> nx.Graph:
    """Build a readable weighted graph using the most frequent characters.

    ``maximum_characters=0`` keeps every detected character, which is the mode the
    proposal calls "one graph for all characters". ``minimum_mentions`` drops names
    that occur only once or twice, which are usually NER noise.
    """

    if minimum_edge_weight < 1:
        raise ValueError("The minimum co-occurrence count must be at least 1.")
    if minimum_mentions < 1:
        raise ValueError("The minimum mention count must be at least 1.")

    ranked = [
        (name, count)
        for name, count in sorted(mention_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_mentions
    ]
    if maximum_characters and maximum_characters > 0:
        ranked = ranked[:maximum_characters]
    allowed = {name for name, _ in ranked}

    graph = nx.Graph()
    for name, count in ranked:
        graph.add_node(name, frequency=int(count), mentions=int(count))

    example_map = examples or {}
    for (source, target), weight in edge_counts.items():
        if source in allowed and target in allowed and weight >= minimum_edge_weight:
            graph.add_edge(
                source,
                target,
                weight=int(weight),
                cooccurrence=int(weight),
                example=example_map.get((source, target), ""),
            )

    if not keep_isolates:
        graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def annotate_graph(graph: nx.Graph, *, seed: int = 42) -> nx.Graph:
    """Add degree, weighted degree, betweenness, and community attributes to nodes."""

    if graph.number_of_nodes() == 0:
        return graph

    degree = dict(graph.degree())
    weighted = {
        node: sum(data.get("weight", 1) for _, _, data in graph.edges(node, data=True))
        for node in graph.nodes
    }

    for source, target, data in graph.edges(data=True):
        weight = max(int(data.get("weight", 1)), 1)
        data["distance"] = 1.0 / weight

    betweenness = (
        nx.betweenness_centrality(graph, weight="distance", seed=seed)
        if graph.number_of_nodes() <= 400
        else {}
    )

    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    except Exception:  # pragma: no cover - defensive for unusual graphs
        communities = [set(graph.nodes)]

    community_index: dict[str, int] = {}
    ordered = sorted(communities, key=len, reverse=True)
    for index, group in enumerate(ordered):
        for node in group:
            community_index[node] = index

    maximum_weighted = max(weighted.values(), default=1) or 1
    for node in graph.nodes:
        graph.nodes[node]["degree"] = degree.get(node, 0)
        graph.nodes[node]["weighted_degree"] = weighted.get(node, 0)
        graph.nodes[node]["betweenness"] = round(betweenness.get(node, 0.0), 4)
        graph.nodes[node]["community"] = community_index.get(node, 0)
        graph.nodes[node]["strength"] = round(
            weighted.get(node, 0) / maximum_weighted, 4
        )
    return graph


def graph_metrics(graph: nx.Graph) -> dict[str, float]:
    """Return summary statistics used by the interface and the evaluation report."""

    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    weights = [data.get("weight", 1) for _, _, data in graph.edges(data=True)]
    return {
        "nodes": nodes,
        "edges": edges,
        "density": round(nx.density(graph), 4) if nodes > 1 else 0.0,
        "average_degree": round(2 * edges / nodes, 2) if nodes else 0.0,
        "components": nx.number_connected_components(graph) if nodes else 0,
        "average_weight": round(sum(weights) / len(weights), 2) if weights else 0.0,
        "maximum_weight": max(weights, default=0),
    }


def layout_graph(
    graph: nx.Graph,
    *,
    seed: int = 42,
    scale: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """Return a deterministic spring layout normalised to ``[-1, 1]``."""

    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes)): (0.0, 0.0)}

    node_count = graph.number_of_nodes()
    spacing = 1.6 if node_count <= 25 else 2.3
    positions = nx.spring_layout(
        graph,
        seed=seed,
        weight="weight",
        iterations=120 if node_count <= 25 else 220,
        k=spacing / math.sqrt(node_count),
    )
    xs = [value[0] for value in positions.values()]
    ys = [value[1] for value in positions.values()]
    x_span = max(max(xs) - min(xs), 1e-6)
    y_span = max(max(ys) - min(ys), 1e-6)
    centre_x = (max(xs) + min(xs)) / 2
    centre_y = (max(ys) + min(ys)) / 2
    return {
        node: (
            (value[0] - centre_x) / x_span * 2 * scale,
            (value[1] - centre_y) / y_span * 2 * scale,
        )
        for node, value in positions.items()
    }


def chapter_presence(
    chapters: Sequence[object],
    mentions: Iterable[object],
    *,
    character_limit: int = 25,
) -> tuple[list[str], list[str], list[list[int]]]:
    """Build a character × chapter mention matrix for the heatmap view.

    Returns ``(character_names, chapter_labels, matrix)``.
    """

    labels = [getattr(chapter, "title", str(chapter)) for chapter in chapters]
    totals: Counter[str] = Counter()
    matrix: dict[str, list[int]] = {}
    starts = [getattr(chapter, "start", 0) for chapter in chapters]

    for mention in mentions:
        name = getattr(mention, "name", str(mention))
        offset = getattr(mention, "start_char", 0)
        position = bisect.bisect_right(starts, offset) - 1
        if position < 0 or position >= len(chapters):
            continue
        if offset >= getattr(chapters[position], "end", 0):
            continue
        row = matrix.setdefault(name, [0] * len(labels))
        row[position] += 1
        totals[name] += 1

    if not matrix:
        return [], labels, []

    names = [name for name, _ in totals.most_common(character_limit)]
    return names, labels, [matrix[name] for name in names]
