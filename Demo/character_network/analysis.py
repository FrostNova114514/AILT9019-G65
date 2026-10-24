"""End-to-end analysis pipeline: text in, weighted character network out."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import networkx as nx

from .extraction import Mention, extract_mentions
from .graph import (
    annotate_graph,
    build_graph,
    count_cooccurrences,
    graph_metrics,
)
from .names import Merge, parse_alias_json
from .textproc import (
    Chapter,
    InputValidationError,
    find_chapters,
    paragraph_ranges,
    slice_text,
    validate_text,
    window_ranges,
)

__all__ = [
    "AnalysisResult",
    "InputValidationError",
    "analyze_text",
    "detect_chapters",
    "parse_alias_json",
    "slice_to_chapters",
    "validate_text",
]


@dataclass
class AnalysisResult:
    """Structured outputs used by the web interface, the CLI, and the evaluation."""

    graph: nx.Graph
    mentions: list[Mention]
    character_counts: Counter[str]
    edge_counts: Counter[tuple[str, str]]
    detected_names: list[str]
    mode: str
    segment_count: int
    text: str = ""
    chapters: list[Chapter] = field(default_factory=list)
    merges: list[Merge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    alias_map: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def characters(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def relationships(self) -> int:
        return self.graph.number_of_edges()

    def top_characters(self, limit: int = 10) -> list[str]:
        """Return detected characters ordered by mention count."""

        ranked = sorted(
            self.character_counts.items(), key=lambda item: (-item[1], item[0])
        )
        return [name for name, _ in ranked[:limit]]

    def strongest_pairs(self, limit: int = 5) -> list[tuple[str, str, int]]:
        """Return the heaviest co-occurrence pairs."""

        pairs = sorted(
            self.edge_counts.items(), key=lambda item: (-item[1], item[0])
        )
        return [(source, target, weight) for (source, target), weight in pairs[:limit]]


def detect_chapters(text: str) -> list[Chapter]:
    """Return the chapters detected in ``text`` (empty when there are none)."""

    return find_chapters(text)


def slice_to_chapters(
    text: str, chapters: Sequence[Chapter], first: int, last: int
) -> str:
    """Return the text of chapters ``first``..``last`` (1-based, inclusive)."""

    return slice_text(text, chapters, first, last)


def _sweep_segments(
    ranges: Sequence[tuple[int, int]],
    mentions: Sequence[Mention],
    *,
    start_attr: str,
    end_attr: str,
    text: str,
) -> list[tuple[frozenset[str], int, int]]:
    """Collect the characters active in each range with a single left-to-right sweep.

    Ranges must be sorted and non-decreasing, which paragraph and sliding-window
    ranges both are. This keeps a full novel at linear cost instead of comparing
    every mention against every unit.
    """

    ordered = sorted(mentions, key=lambda item: getattr(item, start_attr))
    active: list[Mention] = []
    pointer = 0
    segments: list[tuple[frozenset[str], int, int]] = []

    for start, end in ranges:
        while pointer < len(ordered) and getattr(ordered[pointer], start_attr) < end:
            mention = ordered[pointer]
            if getattr(mention, end_attr) > start:
                active.append(mention)
            pointer += 1
        active = [mention for mention in active if getattr(mention, end_attr) > start]

        names = frozenset(mention.name for mention in active)
        if active:
            char_start = min(mention.start_char for mention in active)
            char_end = max(mention.end_char for mention in active)
        else:
            char_start, char_end = start, min(end, start + 1)
        segments.append((names, max(char_start, 0), min(char_end, len(text))))
    return segments


def _segments(
    text: str,
    mentions: Sequence[Mention],
    mode: str,
    window_size: int,
    token_count: int,
) -> list[tuple[frozenset[str], int, int]]:
    if mode == "paragraph":
        return _sweep_segments(
            paragraph_ranges(text),
            mentions,
            start_attr="start_char",
            end_attr="end_char",
            text=text,
        )
    if mode == "window":
        windows = window_ranges(token_count, window_size)
        return _sweep_segments(
            windows,
            mentions,
            start_attr="start_token",
            end_attr="end_token",
            text=text,
        )
    raise InputValidationError(
        "Unknown co-occurrence mode. Choose paragraphs or a sliding token window."
    )


def analyze_text(
    text: str,
    nlp: Any,
    *,
    aliases: Mapping[str, str] | None = None,
    manual_names: Sequence[str] | None = None,
    excluded_names: Sequence[str] | None = None,
    mode: str = "paragraph",
    window_size: int = 120,
    minimum_edge_weight: int = 1,
    minimum_mentions: int = 1,
    maximum_characters: int = 0,
    auto_merge: bool = True,
    require_english: bool = True,
) -> AnalysisResult:
    """Run the complete pipeline and return a structured, reusable result.

    Pipeline: validate text → detect PERSON entities → apply aliases and safe variant
    merges → group mentions into paragraphs or windows → count pairs → build and
    annotate the NetworkX graph.
    """

    started = time.perf_counter()
    cleaned = validate_text(text, require_english=require_english)
    chapters = find_chapters(cleaned)

    extraction = extract_mentions(
        cleaned,
        nlp,
        aliases=aliases,
        manual_names=manual_names,
        excluded_names=excluded_names,
        auto_merge=auto_merge,
    )
    mentions = extraction.mentions

    distinct_names = {mention.name for mention in mentions}
    if len(distinct_names) < 2:
        raise InputValidationError(
            "Not enough character relationships were detected: fewer than two person names "
            "were found. Try a narrative English text, add character names manually, or load "
            "an alias preset for the book."
        )

    segments = _segments(cleaned, mentions, mode, window_size, extraction.token_count)
    _, edge_counts, segment_count, examples = count_cooccurrences(segments, cleaned)

    # Node size and the ``minimum_mentions`` filter use real mention counts, not the
    # number of paragraphs a character appears in.
    mention_counts = Counter(mention.name for mention in mentions)

    graph = build_graph(
        mention_counts,
        edge_counts,
        minimum_edge_weight=minimum_edge_weight,
        minimum_mentions=minimum_mentions,
        maximum_characters=maximum_characters,
        examples=examples,
    )
    if graph.number_of_nodes() < 2:
        raise InputValidationError(
            "Not enough character relationships were detected. The detected names do not share "
            "any paragraph or window. Lower the minimum co-occurrence count or use a more "
            "character-focused English text."
        )
    if graph.number_of_edges() == 0:
        raise InputValidationError(
            "Not enough character relationships were detected at the current threshold. "
            "Lower the minimum co-occurrence count, widen the window, or try another text."
        )

    annotate_graph(graph)
    elapsed = time.perf_counter() - started

    settings = {
        "mode": mode,
        "window_size": window_size,
        "minimum_edge_weight": minimum_edge_weight,
        "minimum_mentions": minimum_mentions,
        "maximum_characters": maximum_characters,
        "auto_merge": auto_merge,
        "alias_count": len(aliases or {}),
        "manual_names": list(manual_names or []),
        "excluded_names": list(excluded_names or []),
        "require_english": require_english,
    }

    return AnalysisResult(
        graph=graph,
        mentions=mentions,
        character_counts=mention_counts,
        edge_counts=Counter(edge_counts),
        detected_names=[name for name, _ in mention_counts.most_common()],
        mode=mode,
        segment_count=segment_count,
        text=cleaned,
        chapters=chapters,
        merges=extraction.merges,
        warnings=extraction.warnings,
        alias_map=dict(aliases or {}),
        settings=settings,
        elapsed_seconds=round(elapsed, 2),
        metrics=graph_metrics(graph),
    )
