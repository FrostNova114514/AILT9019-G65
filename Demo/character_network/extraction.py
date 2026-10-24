"""Person-name recognition with spaCy and assembly of character mentions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .names import (
    Merge,
    apply_merges,
    normalize_surface,
    resolve_name,
    suggest_variant_merges,
    title_of,
)

DEFAULT_MODEL = "en_core_web_sm"

#: Words that never belong to a character name. They appear when NER swallows part of
#: a sentence ("Charlotte said") or a chapter heading ("Chapter I").
_STOPWORD_TOKENS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by", "for", "with",
    "from", "as", "but", "not", "no", "yes", "is", "was", "were", "are", "be", "been",
    "he", "she", "it", "they", "we", "you", "i", "his", "her", "their", "our", "my",
    "your", "its", "this", "that", "these", "those", "said", "says", "say", "replied",
    "cried", "asked", "answered", "added", "observed", "thought", "chapter", "part",
    "book", "volume", "act", "scene", "illustration", "contents", "preface",
}

#: Honorifics that are not names on their own ("Miss" tagged as a person).
_HONORIFIC_TOKENS = {
    "mr", "mrs", "miss", "ms", "dr", "sir", "lady", "lord", "madam", "madame",
    "colonel", "captain", "major", "general", "lieutenant", "sergeant", "reverend",
    "rev", "professor", "prof", "judge", "mister", "mistress",
}

#: Lower-case particles that are legitimate inside a name ("van Helsing").
_NAME_PARTICLES = {
    "de", "da", "di", "du", "del", "della", "van", "von", "der", "den", "la", "le",
    "ten", "ter", "bin", "ibn", "al", "ap", "mac", "mc",
}

#: Prefixes re-attached to a following name. ``St`` is only treated as a prefix here
#: (not as a strippable title) so that "St John" survives as a name.
_ATTACHABLE_PREFIXES = _HONORIFIC_TOKENS | {"st"}

#: Entity labels that indicate a place or an organisation rather than a person.
_NON_PERSON_LABELS = {"GPE", "LOC", "FAC", "ORG", "NORP"}


@dataclass(frozen=True)
class Mention:
    """A normalized character mention and its location in the input text."""

    name: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    surface: str = ""


@dataclass
class ExtractionResult:
    """Everything the analysis needs to know about detected names."""

    mentions: list[Mention]
    counts: dict[str, int]
    token_count: int = 0
    merges: list[Merge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.counts)


class ModelLoadError(RuntimeError):
    """Raised when the spaCy English pipeline cannot be loaded."""


def load_pipeline(model: str = DEFAULT_MODEL) -> Any:
    """Load the local English spaCy pipeline with only the NER components enabled."""

    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - spaCy is a declared dependency
        raise ModelLoadError(
            "spaCy is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    try:
        nlp = spacy.load(model)
    except OSError as exc:
        raise ModelLoadError(
            f"The spaCy English model '{model}' is missing. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc

    # NER needs the tokenizer and the shared tok2vec weights only. Dropping the
    # tagger, parser, and lemmatizer roughly halves the run time on full novels.
    try:
        nlp.select_pipes(enable=["tok2vec", "ner"])
    except ValueError:  # pragma: no cover - very old or unusual models
        pass
    return nlp


def _prepare_doc(nlp: Any, text: str) -> Any:
    """Run the pipeline, raising the model token limit when a long book requires it."""

    if getattr(nlp, "max_length", 0) and nlp.max_length < len(text):
        nlp.max_length = len(text) + 1000
    return nlp(text)


def _valid_character_name(name: str) -> bool:
    if not name or len(name) > 80:
        return False
    words = name.split()
    if len(words) > 6:
        return False
    if not any(character.isalpha() for character in name):
        return False
    if len(words) == 1 and len(name.strip(".,;:")) < 3:
        return False  # "St", "Ed", and other fragments

    lowered = name.casefold().strip(".,;:!?")
    if lowered in _HONORIFIC_TOKENS or lowered in _STOPWORD_TOKENS:
        return False

    for index, word in enumerate(words):
        token = word.casefold().strip(".,;:!?")
        if not token:
            continue
        if token in _STOPWORD_TOKENS and word[:1].islower():
            return False
        if token in _HONORIFIC_TOKENS and index == 0 and len(words) == 1:
            return False
        # NER fragments such as "satin sacques" or "varry good-natured".
        if word[:1].islower() and token not in _NAME_PARTICLES:
            return False
        # A trailing single letter or Roman numeral ("Charles I").
        if len(token) == 1 and index > 0:
            return False
        # Possessive fragments such as "Bessie's Gytrash".
        if re.search(r"['\u2019]s$", word, flags=re.IGNORECASE):
            return False

    first = words[0].casefold().strip(".,;:!?")
    return first not in {"chapter", "part", "book", "volume", "act", "scene", "illustration"}


def _given_names(doc: Any) -> set[str]:
    """Return the first tokens of names that spaCy is confident are people.

    The small English model sometimes labels a bare given name as a place (it tags
    ``"Charlotte"`` as GPE while tagging ``"Charlotte Lucas"`` as PERSON). A location
    label is therefore promoted to a person only when the same word is used as the
    first name of a detected person somewhere else in the same text.
    """

    names: set[str] = set()
    for entity in doc.ents:
        if entity.label_ != "PERSON":
            continue
        tokens = normalize_surface(entity.text).split()
        if len(tokens) >= 2:
            names.add(tokens[0].casefold())
    return names


def _is_promoted_name(entity: Any, given_names: set[str]) -> bool:
    """Return True when a location-labelled single token is really a given name."""

    if entity.label_ not in {"GPE", "LOC", "FAC"}:
        return False
    surface = normalize_surface(entity.text)
    if len(surface.split()) != 1:
        return False
    return surface.casefold() in given_names


def _preceding_title(doc: Any, start_token: int) -> str:
    """Return an honorific written immediately before a name, or ``""``.

    spaCy often tags ``"Mrs. Bennet"`` as the single name ``"Bennet"`` because the
    honorific is its own token. Re-attaching it keeps ``Mr Bennet`` and ``Mrs Bennet``
    as separate people instead of collapsing both parents into one node.
    """

    index = start_token - 1
    if index < 0:
        return ""
    token = doc[index]
    if token.text == "." and index - 1 >= 0:
        index -= 1
        token = doc[index]
    surface = token.text.rstrip(".").casefold()
    if surface not in _ATTACHABLE_PREFIXES:
        return ""
    if not token.text[:1].isupper():
        return ""
    if token.i + 1 < len(doc) and doc[token.i + 1].is_punct and doc[token.i + 1].text != ".":
        return ""
    return token.text.rstrip(".")


def _entity_mentions(
    doc: Any,
    aliases: Mapping[str, str],
    given_names: set[str],
) -> list[Mention]:
    mentions: list[Mention] = []
    for entity in doc.ents:
        if entity.label_ != "PERSON" and not _is_promoted_name(entity, given_names):
            continue
        surface = normalize_surface(entity.text)
        title = _preceding_title(doc, entity.start)
        if title and not surface.casefold().startswith(title.casefold()):
            surface = f"{title} {surface}"
        name = resolve_name(surface, aliases)
        if not _valid_character_name(name):
            continue
        mentions.append(
            Mention(
                name=name,
                start_char=entity.start_char,
                end_char=entity.end_char,
                start_token=entity.start,
                end_token=entity.end,
                surface=surface,
            )
        )
    return mentions


def _manual_mentions(
    text: str,
    doc: Any,
    names: Sequence[str],
    aliases: Mapping[str, str],
) -> list[Mention]:
    mentions: list[Mention] = []
    for raw_name in names:
        raw_name = normalize_surface(raw_name)
        if not raw_name:
            continue
        canonical = resolve_name(raw_name, aliases)
        if not _valid_character_name(canonical):
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(raw_name)}(?!\w)", re.IGNORECASE)
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
                    surface=raw_name,
                )
            )
    return mentions


def _deduplicate(mentions: Iterable[Mention]) -> list[Mention]:
    """Drop repeated spans and shorter spans that overlap a longer character mention."""

    ordered = sorted(mentions, key=lambda item: (item.start_char, -item.end_char))
    kept: list[Mention] = []
    for mention in ordered:
        if any(
            mention.start_char < existing.end_char and mention.end_char > existing.start_char
            for existing in kept
        ):
            continue
        kept.append(mention)
    return kept


def _title_warnings(mentions: Sequence[Mention]) -> list[str]:
    """Flag one node that is used with several honorifics (likely two people).

    The check runs on the *resolved* name, so a preset that keeps ``Mr Bennet`` and
    ``Mrs Bennet`` apart produces no warning. It only fires when honorifics were
    stripped and two different people ended up in the same node.
    """

    titles_by_name: dict[str, set[str]] = {}
    for mention in mentions:
        title = title_of(mention.surface or mention.name)
        if not title:
            continue
        if title_of(mention.name):  # the canonical name kept its honorific
            continue
        titles_by_name.setdefault(mention.name, set()).add(title)

    warnings: list[str] = []
    for name, titles in sorted(titles_by_name.items()):
        family = {title for title in titles if title in {"mr", "mrs", "miss", "ms", "lady", "sir"}}
        if len(family) > 1:
            formatted = ", ".join(sorted(title.title() for title in family))
            warnings.append(
                f"'{name}' appears with different honorifics ({formatted}). "
                "They are probably different people but share one node; add aliases to separate them."
            )
    return warnings


def extract_mentions(
    text: str,
    nlp: Any,
    *,
    aliases: Mapping[str, str] | None = None,
    manual_names: Sequence[str] | None = None,
    excluded_names: Sequence[str] | None = None,
    auto_merge: bool = True,
) -> ExtractionResult:
    """Detect PERSON entities, apply aliases, and optionally merge safe variants."""

    alias_map = dict(aliases or {})
    doc = _prepare_doc(nlp, text)
    given_names = _given_names(doc)

    mentions = _entity_mentions(doc, alias_map, given_names)
    mentions.extend(_manual_mentions(text, doc, manual_names or [], alias_map))
    mentions = _deduplicate(mentions)

    excluded = {
        resolve_name(name, alias_map).casefold()
        for name in (excluded_names or [])
        if resolve_name(name, alias_map)
    }
    if excluded:
        mentions = [mention for mention in mentions if mention.name.casefold() not in excluded]

    counts: dict[str, int] = {}
    for mention in mentions:
        counts[mention.name] = counts.get(mention.name, 0) + 1

    merges: list[Merge] = []
    if auto_merge and counts:
        suggested = suggest_variant_merges(counts)
        final_map, merges = apply_merges(counts, suggested)
        if merges:
            counts = {}
            remapped: list[Mention] = []
            for mention in mentions:
                canonical = final_map.get(mention.name, mention.name)
                remapped.append(
                    Mention(
                        name=canonical,
                        start_char=mention.start_char,
                        end_char=mention.end_char,
                        start_token=mention.start_token,
                        end_token=mention.end_token,
                        surface=mention.surface,
                    )
                )
                counts[canonical] = counts.get(canonical, 0) + 1
            mentions = remapped
            mentions = _deduplicate(mentions)

    return ExtractionResult(
        mentions=mentions,
        counts=dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        token_count=len(doc),
        merges=merges,
        warnings=_title_warnings(mentions),
    )
