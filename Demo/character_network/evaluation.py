"""Ground-truth metrics used to check the network against a manual reference list.

The criteria follow the revised proposal: how many of the top ten main characters are
recovered, whether the expected major relationships appear as edges, and whether the
protagonist is a central node. Matching is one-to-one and tolerant of partial name
forms, so ``"Bennet"`` can still be credited for ``"Mr Bennet"`` exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import networkx as nx

from .names import name_tokens, strip_titles


@dataclass(frozen=True)
class NameMatch:
    """A matched pair of a detected name and a reference name."""

    detected: str
    reference: str
    score: int


def _score(detected: str, reference: str) -> int:
    detected_clean = strip_titles(detected).casefold()
    reference_clean = strip_titles(reference).casefold()
    if detected_clean == reference_clean:
        return 100

    detected_tokens = set(name_tokens(detected))
    reference_tokens = set(name_tokens(reference))
    if not detected_tokens or not reference_tokens:
        return 0
    if detected_tokens == reference_tokens:
        return 90
    if detected_tokens <= reference_tokens or reference_tokens <= detected_tokens:
        if abs(len(detected_tokens) - len(reference_tokens)) <= 1:
            return 70
        return 50
    if detected_tokens & reference_tokens:
        overlap = len(detected_tokens & reference_tokens)
        if overlap >= min(len(detected_tokens), len(reference_tokens)):
            return 40
    return 0


def match_names(detected: Sequence[str], references: Sequence[str]) -> list[NameMatch]:
    """Return a one-to-one greedy matching between detected and reference names."""

    candidates: list[tuple[int, int, int, str, str]] = []
    for detected_index, detected_name in enumerate(detected):
        for reference_index, reference_name in enumerate(references):
            score = _score(detected_name, reference_name)
            if score:
                candidates.append((score, -detected_index, -reference_index, detected_name, reference_name))

    candidates.sort(reverse=True)
    used_detected: set[str] = set()
    used_reference: set[str] = set()
    matches: list[NameMatch] = []
    for score, _, _, detected_name, reference_name in candidates:
        if detected_name in used_detected or reference_name in used_reference:
            continue
        used_detected.add(detected_name)
        used_reference.add(reference_name)
        matches.append(NameMatch(detected=detected_name, reference=reference_name, score=score))
    return matches


def _recall(found: Iterable[str], references: Sequence[str]) -> float:
    references = list(references)
    if not references:
        return 0.0
    found_set = set(found)
    return round(sum(1 for reference in references if reference in found_set) / len(references), 3)


def _recall_at_k(
    k: int,
    detected_ranked: Sequence[str],
    reference_to_detected: Mapping[str, str],
    references: Sequence[str],
) -> float:
    """Fraction of the reference top-k that also appears in the detected top-k."""

    targets = list(references[:k])
    if not targets:
        return 0.0
    detected_top = set(detected_ranked[:k])
    hits = sum(1 for reference in targets if reference_to_detected.get(reference) in detected_top)
    return round(hits / len(targets), 3)


@dataclass
class BookEvaluation:
    """Metrics for one book evaluated against its reference list."""

    title: str
    detected_ranked: list[str]
    reference_characters: list[str]
    matches: list[NameMatch]
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    character_recall: float
    edge_recall: float
    matched_edges: list[tuple[str, str, str, str]]
    missed_edges: list[tuple[str, str]]
    protagonist: str
    protagonist_detected: str | None
    protagonist_rank: int | None
    unexpected_top_10: list[str]
    node_count: int
    edge_count: int
    elapsed_seconds: float

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "detected_top_10": self.detected_ranked[:10],
            "reference_characters": self.reference_characters,
            "precision_at_10": self.precision_at_10,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "recall_at_20": self.recall_at_20,
            "character_recall": self.character_recall,
            "edge_recall": self.edge_recall,
            "protagonist": self.protagonist,
            "protagonist_detected": self.protagonist_detected,
            "protagonist_rank_by_weighted_degree": self.protagonist_rank,
            "unexpected_top_10": self.unexpected_top_10,
            "matched_edges": self.matched_edges,
            "missed_edges": self.missed_edges,
            "nodes": self.node_count,
            "edges": self.edge_count,
            "elapsed_seconds": self.elapsed_seconds,
        }


def evaluate_book(
    title: str,
    graph: nx.Graph,
    detected_ranked: Sequence[str],
    ground_truth: Mapping,
    *,
    elapsed_seconds: float = 0.0,
) -> BookEvaluation:
    """Compare a generated graph with a manual reference list."""

    reference_characters = list(ground_truth.get("main_characters", []))
    reference_pairs = [tuple(pair) for pair in ground_truth.get("relationships", [])]
    protagonist = str(ground_truth.get("protagonist", reference_characters[0] if reference_characters else ""))

    matches = match_names(detected_ranked, reference_characters)
    detected_to_reference = {match.detected: match.reference for match in matches}
    reference_to_detected = {match.reference: match.detected for match in matches}

    top10_detected = list(detected_ranked[:10])
    matched_top10 = [
        detected for detected in top10_detected if detected_to_reference.get(detected) in set(reference_characters)
    ]
    precision_at_10 = round(len(matched_top10) / max(len(top10_detected), 1), 3)

    recall_at_5 = _recall_at_k(5, detected_ranked, reference_to_detected, reference_characters)
    recall_at_10 = _recall_at_k(10, detected_ranked, reference_to_detected, reference_characters)
    recall_at_20 = _recall_at_k(20, detected_ranked, reference_to_detected, reference_characters)
    character_recall = _recall(detected_to_reference.keys(), reference_characters)

    matched_edges: list[tuple[str, str, str, str]] = []
    missed_edges: list[tuple[str, str]] = []
    for pair in reference_pairs:
        source, target = pair[0], pair[1]
        detected_source = reference_to_detected.get(source)
        detected_target = reference_to_detected.get(target)
        if (
            detected_source
            and detected_target
            and graph.has_edge(detected_source, detected_target)
        ):
            matched_edges.append((source, target, detected_source, detected_target))
        else:
            missed_edges.append((source, target))
    edge_recall = round(len(matched_edges) / max(len(reference_pairs), 1), 3)

    ranked_by_weight = sorted(
        graph.nodes,
        key=lambda name: (-graph.nodes[name].get("weighted_degree", 0), name),
    )
    protagonist_detected = reference_to_detected.get(protagonist)
    protagonist_rank = (
        ranked_by_weight.index(protagonist_detected) + 1
        if protagonist_detected in ranked_by_weight
        else None
    )

    reference_set = set(reference_characters)
    unexpected = [
        detected
        for detected in top10_detected
        if detected_to_reference.get(detected) not in reference_set
    ]

    return BookEvaluation(
        title=title,
        detected_ranked=list(detected_ranked),
        reference_characters=reference_characters,
        matches=matches,
        precision_at_10=precision_at_10,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        recall_at_20=recall_at_20,
        character_recall=character_recall,
        edge_recall=edge_recall,
        matched_edges=matched_edges,
        missed_edges=missed_edges,
        protagonist=protagonist,
        protagonist_detected=protagonist_detected,
        protagonist_rank=protagonist_rank,
        unexpected_top_10=unexpected,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        elapsed_seconds=elapsed_seconds,
    )
