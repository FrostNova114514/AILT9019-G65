"""Character-name normalization, alias resolution, and conservative merging.

The normalization pipeline follows the revised proposal: a small, explicit alias
table plus rule-based title handling. In addition, :func:`suggest_variant_merges`
offers a *conservative* rule for variants such as ``"Darcy"`` and ``"Fitzwilliam
Darcy"``. It only merges when exactly one longer name contains the shorter one, so
ambiguous surnames (``"Bennet"`` in *Pride and Prejudice* belongs to six characters)
are deliberately left alone and reported instead.
"""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

#: Honorifics and ranks stripped before names are compared. ``St`` is excluded on
#: purpose so that ``St John Rivers`` is not reduced to ``John``.
TITLE_WORDS = (
    "mr",
    "mrs",
    "miss",
    "ms",
    "dr",
    "sir",
    "lady",
    "lord",
    "madam",
    "madame",
    "colonel",
    "captain",
    "major",
    "general",
    "lieutenant",
    "sergeant",
    "reverend",
    "rev",
    "professor",
    "prof",
    "judge",
    "mister",
    "mistress",
    "father",
    "mother",
    "aunt",
    "uncle",
)

TITLE_PATTERN = re.compile(
    r"^(?:(?:" + "|".join(TITLE_WORDS) + r")\.?\s+)+",
    flags=re.IGNORECASE,
)

_SPACE_PATTERN = re.compile(r"\s+")
_TRIM_PATTERN = re.compile(r"^[\s\W_]+|[\s\W_]+$", flags=re.UNICODE)
_POSSESSIVE_PATTERN = re.compile(r"(?:['\u2019]s|['\u2019])$", flags=re.IGNORECASE)
_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
)


@dataclass(frozen=True)
class Merge:
    """A recorded name-variant merge so that every decision stays auditable."""

    variant: str
    canonical: str
    mentions: int
    reason: str


def normalize_surface(name: str) -> str:
    """Normalize a name surface form without deciding what it refers to."""

    if not name:
        return ""
    cleaned = unicodedata.normalize("NFC", str(name)).translate(_QUOTE_MAP)
    cleaned = cleaned.strip(" \t\r\n,;:!?()[]{}\"'")
    cleaned = _POSSESSIVE_PATTERN.sub("", cleaned)
    cleaned = _TRIM_PATTERN.sub("", cleaned)
    cleaned = _SPACE_PATTERN.sub(" ", cleaned).strip()

    tokens = []
    for token in cleaned.split():
        if len(token) > 1 and token.isupper():
            # Gutenberg texts often print names in full capitals.
            token = token.capitalize()
            token = re.sub(r"(?<=['-])[a-z]", lambda match: match.group(0).upper(), token)
        tokens.append(token)
    return " ".join(tokens)


def strip_titles(name: str) -> str:
    """Remove leading honorifics such as ``Mr.``, ``Mrs.``, ``Miss``, and ``Dr.``."""

    surface = normalize_surface(name)
    stripped = TITLE_PATTERN.sub("", surface).strip()
    return stripped or surface


def title_of(name: str) -> str:
    """Return the leading honorific of ``name`` in lower case, or ``""``."""

    surface = normalize_surface(name)
    match = TITLE_PATTERN.match(surface)
    if not match:
        return ""
    return match.group(0).strip().rstrip(".").strip().casefold()


def name_tokens(name: str, *, strip: bool = True) -> tuple[str, ...]:
    """Return the comparison tokens of a name (case-folded, titles removed)."""

    surface = strip_titles(name) if strip else normalize_surface(name)
    return tuple(token.casefold().strip(".,;:!?") for token in surface.split() if token)


def is_name_subsequence(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    """Return True when ``shorter`` is a contiguous sub-sequence of ``longer``."""

    if not shorter or len(shorter) >= len(longer):
        return False
    width = len(shorter)
    return any(
        tuple(longer[start : start + width]) == tuple(shorter)
        for start in range(len(longer) - width + 1)
    )


def normalize_aliases(aliases: Mapping[str, str]) -> dict[str, str]:
    """Return an alias mapping with case-folded keys and normalized display names."""

    return _normalize_alias_mapping(aliases.items())


def parse_alias_json(raw: str) -> dict[str, str]:
    """Parse a JSON object whose keys are variants and values are canonical names."""

    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Alias JSON is invalid: {exc.msg}.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Alias data must be a JSON object mapping variants to canonical names.")
    return _normalize_alias_mapping(parsed.items())


def canonical_key(name: str) -> str:
    """Return the comparison key of a name with honorific periods removed.

    ``"Mr. Darcy"`` and ``"Mr Darcy"`` (and the re-attached ``"Mr Darcy"`` that the
    extractor builds) must share one key, otherwise an alias written with a period
    silently stops matching.
    """

    surface = normalize_surface(name)
    surface = re.sub(
        r"\b(" + "|".join(TITLE_WORDS) + r"|st|mr|mrs|dr)\.(?=\s|$)",
        r"\1",
        surface,
        flags=re.IGNORECASE,
    )
    return surface.casefold()


def _normalize_alias_mapping(items: Iterable[tuple[object, object]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for variant, canonical in items:
        if not isinstance(variant, str) or not isinstance(canonical, str):
            raise ValueError("Every alias and canonical name must be text.")
        variant_clean = normalize_surface(variant)
        canonical_clean = normalize_surface(canonical)
        if variant_clean and canonical_clean:
            aliases[canonical_key(variant_clean)] = canonical_clean
    return aliases


def parse_alias_table(raw: str) -> dict[str, str]:
    """Parse an alias table pasted or uploaded by the user.

    JSON objects, CSV rows (``variant,canonical``) and simple ``variant => canonical``
    lines are all accepted so that non-programmers can edit a plain table.
    """

    text = (raw or "").strip()
    if not text:
        return {}
    if text[0] in "{[":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Alias JSON is invalid: {exc.msg}.") from exc
        if isinstance(parsed, dict):
            return _normalize_alias_mapping(parsed.items())
        if isinstance(parsed, list):
            items = []
            for entry in parsed:
                if isinstance(entry, dict) and {"variant", "canonical"} <= set(entry):
                    items.append((entry["variant"], entry["canonical"]))
                else:
                    raise ValueError(
                        "Each alias entry must be an object with 'variant' and 'canonical'."
                    )
            return _normalize_alias_mapping(items)
        raise ValueError("Alias data must be a JSON object or a list of alias entries.")

    items: list[tuple[object, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        separator = "=>" if "=>" in line else ("," if "," in line else (":" if ":" in line else None))
        if separator is None:
            raise ValueError(
                f"Could not read alias line: {line!r}. Use 'variant,canonical' or 'variant => canonical'."
            )
        variant, canonical = line.split(separator, 1)
        variant, canonical = variant.strip().strip('"'), canonical.strip().strip('"')
        if variant.casefold() in {"variant", "alias", "name", "from"} and canonical.casefold() in {
            "canonical",
            "to",
            "target",
        }:
            continue  # CSV header
        items.append((variant, canonical))
    if not items:
        raise ValueError("No alias rows were found. Use one 'variant,canonical' pair per line.")
    return _normalize_alias_mapping(items)


def aliases_to_json(aliases: Mapping[str, str]) -> str:
    """Render an alias mapping as pretty JSON for the editor."""

    return json.dumps(dict(sorted(aliases.items())), indent=2, ensure_ascii=False)


def aliases_to_csv(aliases: Mapping[str, str]) -> str:
    """Render an alias mapping as CSV with a header row."""

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["variant", "canonical"])
    for variant, canonical in sorted(aliases.items()):
        writer.writerow([variant, canonical])
    return buffer.getvalue()


def resolve_name(raw: str, aliases: Mapping[str, str] | None = None) -> str:
    """Normalize a detected name and apply an explicit alias mapping.

    The lookup order is: the honorific-normalized surface, the exact surface, then the
    title-stripped form. Titles are removed from the display name when no alias claims
    them, which keeps ``Mr. Darcy`` and ``Darcy`` in the same node.
    """

    alias_map = aliases or {}
    surface = normalize_surface(raw)
    if not surface:
        return ""
    direct = alias_map.get(canonical_key(surface))
    if direct:
        return direct
    stripped = strip_titles(surface)
    if not stripped:
        return ""
    direct = alias_map.get(canonical_key(stripped))
    if direct:
        return direct
    return stripped


def _is_edge_fragment(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    """True when ``shorter`` is the leading or trailing part of ``longer``.

    Only the edges of a name are merged: ``"Darcy"`` into ``"Fitzwilliam Darcy"`` and
    ``"Elizabeth"`` into ``"Elizabeth Bennet"``. A word from the middle of a longer
    span is kept separate because it is usually a NER fragment rather than a name.
    """

    width = len(shorter)
    if not width or width >= len(longer):
        return False
    return tuple(longer[:width]) == tuple(shorter) or tuple(longer[-width:]) == tuple(shorter)


def suggest_variant_merges(counts: Mapping[str, int]) -> dict[str, str]:
    """Return safe ``variant -> canonical`` merges for partial name forms.

    A merge is proposed only when the shorter name is the leading or trailing part of
    exactly one longer name in the same text. Two-character families such as ``Bennet``
    / ``Elizabeth Bennet`` / ``Jane Bennet`` therefore produce no merge.
    """

    names = [name for name in counts if normalize_surface(name)]
    tokens = {name: name_tokens(name) for name in names}
    merges: dict[str, str] = {}

    for variant in names:
        variant_tokens = tokens[variant]
        if not variant_tokens or len(variant_tokens) > 3:
            continue
        candidates = [
            other
            for other in names
            if other != variant
            and len(tokens[other]) <= 5
            and _is_edge_fragment(variant_tokens, tokens[other])
        ]
        if len(candidates) != 1:
            continue  # ambiguous short form: leave it alone
        merges[variant] = candidates[0]
    return merges


def apply_merges(
    counts: Mapping[str, int],
    merges: Mapping[str, str],
) -> tuple[dict[str, str], list[Merge]]:
    """Attach every merged variant to its final canonical name.

    Chains (``A -> B`` and ``B -> C``) are resolved transitively and ambiguous or
    cyclic mappings are dropped rather than guessed.
    """

    final: dict[str, str] = {}

    def resolve(name: str, trail: tuple[str, ...] = ()) -> str | None:
        if name in final:
            return final[name]
        if name in trail:
            # A cycle: keep every member as its own name rather than merging blindly.
            for member in trail:
                final[member] = member
            return None
        target = merges.get(name)
        if not target:
            final[name] = name
            return name
        resolved = resolve(target, trail + (name,))
        if resolved is None:
            final[name] = name
            return None
        final[name] = resolved
        return resolved

    applied: list[Merge] = []
    for variant in merges:
        canonical = resolve(variant)
        if canonical is None or canonical == variant:
            final[variant] = variant
            continue
        applied.append(
            Merge(
                variant=variant,
                canonical=canonical,
                mentions=counts.get(variant, 0),
                reason="unique partial-name match",
            )
        )
    return final, applied


def merge_alias_mapping(
    aliases: Mapping[str, str],
    auto_merges: Mapping[str, str],
) -> dict[str, str]:
    """Combine explicit aliases with automatic merges, giving aliases priority."""

    combined = {variant: canonical for variant, canonical in aliases.items()}
    for variant, canonical in auto_merges.items():
        key = normalize_surface(variant).casefold()
        combined.setdefault(key, canonical)
    return combined
