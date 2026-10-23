"""Text validation, character extraction, normalization, and co-occurrence analysis."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
from langdetect import DetectorFactory, LangDetectException, detect


DetectorFactory.seed = 0

TITLE_PATTERN = re.compile(
    r"^(?:(?:mr|mrs|miss|ms|dr|sir|lady|lord|prof|professor)\.?\s+)+",
    flags=re.IGNORECASE,
)
SPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


class InputValidationError(ValueError):
    """Raised when input cannot be processed safely or meaningfully."""


@dataclass(frozen=True)
class Mention:
    """A normalized character mention and its location in the input text."""

    name: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int


@dataclass
class AnalysisResult:
    """Structured outputs used by the web interface and downloads."""

    graph: nx.Graph
    mentions: list[Mention]
    character_counts: Counter[str]
    edge_counts: Counter[tuple[str, str]]
    detected_names: list[str]
    mode: str
    segment_count: int


def validate_text(text: str, *, require_english: bool = True) -> str:
    """Validate and normalize text before analysis."""

    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        raise InputValidationError("Please upload or paste a non-empty English text.")
    if len(cleaned) < 40:
        raise InputValidationError(
            "The text is too short to build a meaningful relationship network."
        )
    if len(cleaned) > 2_000_000:
        raise InputValidationError(
            "The demo accepts at most 2,000,000 characters per analysis."
        )

    words = WORD_PATTERN.findall(cleaned)
    if len(words) < 8:
        raise InputValidationError(
            "The text does not contain enough words for character analysis."
        )

    if require_english:
        sample = cleaned[:10_000]
        try:
            language = detect(sample)
        except LangDetectException as exc:
            raise InputValidationError(
                "The language could not be identified. Please provide English prose."
            ) from exc
        if language != "en":
            raise InputValidationError(
                f"The text appears to be '{language}', but this version supports English only."
            )
    return cleaned


def parse_alias_json(raw: str) -> dict[str, str]:
    """Parse a JSON object whose keys are variants and values are canonical names."""

    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"Alias JSON is invalid: {exc.msg}.") from exc
    if not isinstance(parsed, dict):
        raise InputValidationError("Alias data must be a JSON object.")

    aliases: dict[str, str] = {}
    for variant, canonical in parsed.items():
        if not isinstance(variant, str) or not isinstance(canonical, str):
            raise InputValidationError("Every alias and canonical name must be text.")
        variant_clean = normalize_name_surface(variant)
        canonical_clean = normalize_name_surface(canonical)
        if variant_clean and canonical_clean:
            aliases[variant_clean.casefold()] = canonical_clean
    return aliases


def normalize_name_surface(name: str) -> str:
    """Normalize a name while preserving meaningful title distinctions."""

    cleaned = name.replace("’", "'").strip(" \t\r\n,;:!?()[]{}\"“”")
    cleaned = re.sub(r"(?:'s|')$", "", cleaned, flags=re.IGNORECASE)
    cleaned = SPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def clean_person_name(name: str) -> str:
    """Apply conservative whitespace, possessive, punctuation, and title cleanup."""

    return TITLE_PATTERN.sub("", normalize_name_surface(name)).strip()


def canonicalize_name(name: str, aliases: Mapping[str, str]) -> str:
    """Normalize a detected name and apply an explicit alias mapping."""

    surface = normalize_name_surface(name)
    if not surface:
        return ""
    direct_match = aliases.get(surface.casefold())
    if direct_match:
        return direct_match
    cleaned = TITLE_PATTERN.sub("", surface).strip()
    if not cleaned:
        return ""
    return aliases.get(cleaned.casefold(), cleaned)


def _entity_mentions(
    doc: Any,
    aliases: Mapping[str, str],
    excluded: set[str],
) -> list[Mention]:
    mentions: list[Mention] = []
    for entity in doc.ents:
        if entity.label_ != "PERSON":
            continue
        name = canonicalize_name(entity.text, aliases)
        if not _valid_character_name(name) or name.casefold() in excluded:
            continue
        mentions.append(
            Mention(
                name=name,
                start_char=entity.start_char,
                end_char=entity.end_char,
                start_token=entity.start,
                end_token=entity.end,
            )
        )
    return mentions


def _manual_mentions(
    text: str,
    doc: Any,
    names: Sequence[str],
    aliases: Mapping[str, str],
    excluded: set[str],
) -> list[Mention]:
    mentions: list[Mention] = []
    for raw_name in names:
        canonical = canonicalize_name(raw_name, aliases)
        if not _valid_character_name(canonical) or canonical.casefold() in excluded:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(raw_name.strip())}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
            if span is None:
                continue
            mentions.append(
                Mention(
                    name=canonical,
                    start_char=match.start(),
                    end_char=match.end(),
                    start_token=span.start,
                    end_token=span.end,
                )
            )
    return mentions


def _valid_character_name(name: str) -> bool:
    if not name or len(name) > 80:
        return False
    words = name.split()
    if len(words) > 6:
        return False
    return any(character.isalpha() for character in name)


def extract_mentions(
    text: str,
    nlp: Any,
    aliases: Mapping[str, str] | None = None,
    manual_names: Sequence[str] | None = None,
    excluded_names: Sequence[str] | None = None,
) -> tuple[Any, list[Mention]]:
    """Run NER and merge exact matches for user-supplied character names."""

    alias_map = aliases or {}
    excluded = {
        canonicalize_name(name, alias_map).casefold()
        for name in (excluded_names or [])
        if canonicalize_name(name, alias_map)
    }
    doc = nlp(text)
    mentions = _entity_mentions(doc, alias_map, excluded)
    mentions.extend(
        _manual_mentions(text, doc, manual_names or [], alias_map, excluded)
    )

    unique: dict[tuple[int, int, str], Mention] = {}
    for mention in mentions:
        unique[(mention.start_char, mention.end_char, mention.name.casefold())] = mention
    ordered = sorted(unique.values(), key=lambda item: (item.start_char, item.end_char))
    return doc, ordered


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        ranges.append((match.start(), match.end()))
    return ranges or [(0, len(text))]


def _paragraph_segments(text: str, mentions: Sequence[Mention]) -> list[set[str]]:
    return [
        {
            mention.name
            for mention in mentions
            if mention.start_char >= start and mention.start_char < end
        }
        for start, end in _paragraph_ranges(text)
    ]


def _window_segments(
    token_count: int,
    mentions: Sequence[Mention],
    window_size: int,
) -> list[set[str]]:
    if window_size < 10:
        raise InputValidationError("The sliding-window size must be at least 10 tokens.")
    stride = max(1, window_size // 2)
    starts = list(range(0, max(token_count, 1), stride))
    segments: list[set[str]] = []
    for start in starts:
        end = min(token_count, start + window_size)
        segments.append(
            {
                mention.name
                for mention in mentions
                if mention.start_token < end and mention.end_token > start
            }
        )
        if end >= token_count:
            break
    return segments


def count_cooccurrences(
    segments: Iterable[set[str]],
) -> tuple[Counter[str], Counter[tuple[str, str]], int]:
    """Count segment presence and unordered character pairs."""

    characters: Counter[str] = Counter()
    edges: Counter[tuple[str, str]] = Counter()
    segment_count = 0
    for names in segments:
        segment_count += 1
        sorted_names = sorted(names, key=str.casefold)
        characters.update(sorted_names)
        edges.update(combinations(sorted_names, 2))
    return characters, edges, segment_count


def build_graph(
    mention_counts: Counter[str],
    edge_counts: Counter[tuple[str, str]],
    *,
    minimum_edge_weight: int = 1,
    maximum_characters: int = 40,
) -> nx.Graph:
    """Build a readable weighted graph using the most frequent characters."""

    if minimum_edge_weight < 1:
        raise InputValidationError("Minimum edge weight must be at least 1.")
    allowed = {
        name for name, _ in mention_counts.most_common(maximum_characters)
    }
    graph = nx.Graph()
    for name in allowed:
        graph.add_node(name, frequency=mention_counts[name])
    for (source, target), weight in edge_counts.items():
        if source in allowed and target in allowed and weight >= minimum_edge_weight:
            graph.add_edge(source, target, weight=weight)

    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    return graph


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
    maximum_characters: int = 40,
    require_english: bool = True,
) -> AnalysisResult:
    """Run the complete analysis pipeline and return structured results."""

    cleaned = validate_text(text, require_english=require_english)
    doc, mentions = extract_mentions(
        cleaned,
        nlp,
        aliases=aliases,
        manual_names=manual_names,
        excluded_names=excluded_names,
    )
    if len({mention.name for mention in mentions}) < 2:
        raise InputValidationError(
            "Fewer than two character names were detected. Try adding names manually or using a more character-focused English text."
        )

    if mode == "paragraph":
        segments = _paragraph_segments(cleaned, mentions)
    elif mode == "window":
        segments = _window_segments(len(doc), mentions, window_size)
    else:
        raise InputValidationError("Unknown co-occurrence mode.")

    segment_counts, edges, segment_count = count_cooccurrences(segments)
    mention_counts = Counter(mention.name for mention in mentions)
    graph = build_graph(
        mention_counts,
        edges,
        minimum_edge_weight=minimum_edge_weight,
        maximum_characters=maximum_characters,
    )
    if graph.number_of_edges() == 0:
        raise InputValidationError(
            "No character relationships remain at the current threshold. Lower the minimum edge weight or try another text."
        )

    return AnalysisResult(
        graph=graph,
        mentions=mentions,
        character_counts=mention_counts,
        edge_counts=edges,
        detected_names=[name for name, _ in mention_counts.most_common()],
        mode=mode,
        segment_count=segment_count,
    )
