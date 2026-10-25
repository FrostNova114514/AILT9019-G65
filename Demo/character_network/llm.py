"""Optional DeepSeek-assisted cast extraction.

The local spaCy model misses minor characters and occasionally mislabels a person as a
place. This module asks a language model for a *cast list* — canonical names, aliases,
and place names to ignore — and then hands that list to the deterministic pipeline. The
model never sees the whole book: it reads evenly spaced excerpts capped at a few
thousand tokens. Counting, aliasing, thresholds, and drawing all stay local and
reproducible.

The feature is off by default. When it is enabled, the excerpts below are sent to the
configured endpoint (DeepSeek by default); nothing else leaves the machine.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .names import canonical_key, name_tokens, normalize_surface
from .textproc import Chapter

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_EXCERPT = 24_000
DEFAULT_MAX_CHARACTERS = 60

#: Files checked when ``DEEPSEEK_API_KEY`` is not in the environment. They live beside
#: ``app.py`` and are gitignored, so a key can be configured once without retyping it in
#: every command.
KEY_FILES = (
    Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",
    Path(__file__).resolve().parents[1] / ".env",
)

SYSTEM_PROMPT = (
    "You build character lists for a reading aid that maps which people appear near "
    "each other in an English novel. You are precise: you only list people that are "
    "named in the excerpts, and you do not invent facts."
)

USER_PROMPT = """From the excerpts below, list the named characters of this book.

Rules:
1. Use the full name as printed, e.g. "Elizabeth Bennet", "Fitzwilliam Darcy".
2. Put short forms and title variants in "aliases" for that character: "Lizzy",
   "Miss Bennet", "Mr. Darcy", "Darcy".
3. Be thorough: include every named person you can see, including minor relatives,
   servants, and neighbours who have a personal name. Do not include places,
   buildings, organisations, ships, regiments, or unnamed roles such as "the housekeeper".
4. If one surname belongs to several people (a family), list each person separately
   and do not put the bare surname in any alias list.
5. Prefer names you can see in the excerpts, but you may also list major characters of
   this novel that these excerpts happen not to show. Every name you return is checked
   against the full text and dropped if it never appears there, so a wrong name costs
   nothing; never invent a person who is not in the book.
6. List places and buildings in "places" so they can be ignored.

Return one JSON object and nothing else:
{"characters": [{"name": "Elizabeth Bennet", "aliases": ["Lizzy", "Miss Elizabeth"]}],
 "places": ["Meryton", "Netherfield"]}

Excerpts:
{excerpt}
"""


class LLMError(RuntimeError):
    """Raised when the cast list cannot be obtained."""


@dataclass
class CastResult:
    """A cast list proposed by the language model."""

    characters: list[str]
    aliases: dict[str, str]
    places: list[str]
    alias_surfaces: list[str] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    usage: dict = field(default_factory=dict)
    excerpt_characters: int = 0
    raw_response: str = ""

    @property
    def summary(self) -> str:
        return (
            f"{len(self.characters)} characters, {len(self.aliases)} alias links, "
            f"{len(self.places)} places ignored"
        )


def _read_key_file(path: Path, name: str = "DEEPSEEK_API_KEY") -> str:
    """Return a ``KEY = value`` entry from a TOML or dotenv style file."""

    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().strip("'\"") != name:
            continue
        return value.strip().strip("'\"").strip()
    return ""


def api_key_from_env(paths: Sequence[Path] | None = None) -> str:
    """Return the configured key: environment first, then ``secrets.toml``, then ``.env``."""

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    for path in paths if paths is not None else KEY_FILES:
        key = _read_key_file(path)
        if key:
            return key
    return ""


def key_source() -> str:
    """Describe where the key came from, for the interface caption."""

    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return "DEEPSEEK_API_KEY environment variable"
    for path in KEY_FILES:
        if _read_key_file(path):
            try:
                return str(path.relative_to(path.parents[1]))
            except (ValueError, IndexError):  # pragma: no cover
                return str(path)
    return ""


def build_excerpt(
    text: str,
    chapters: Sequence[Chapter] | None = None,
    *,
    limit: int = DEFAULT_MAX_EXCERPT,
) -> str:
    """Return a character-rich sample of ``text`` bounded by ``limit`` characters.

    Evenly spaced windows are sampled across the whole book rather than only its
    opening, because minor characters first appear late. Each window is labelled with
    its chapter when headings were detected, which helps the model name people
    consistently.
    """

    text = text or ""
    if len(text) <= limit:
        return text

    window = max(500, limit // 30)
    count = max(6, min(40, limit // window))
    step = max(1.0, (len(text) - window) / max(count - 1, 1))

    pieces: list[str] = []
    for index in range(count):
        start = int(index * step)
        # Start at a word boundary so the first name is not cut in half.
        if start and text[start - 1].isalnum():
            space = text.find(" ", start, min(len(text), start + 80))
            if space != -1:
                start = space + 1
        piece = text[start : start + window].strip()
        if not piece:
            continue
        label = next(
            (
                chapter.title
                for chapter in (chapters or ())
                if chapter.start <= start < chapter.end
            ),
            "",
        )
        pieces.append(f"[{label}]\n{piece}" if label else piece)

    return "\n\n---\n\n".join(pieces)


def _post_json(url: str, payload: Mapping, api_key: str, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = json.loads(exc.read().decode("utf-8"))
            detail = str(body.get("error", {}).get("message", ""))[:200]
        except Exception:  # pragma: no cover - best effort only
            detail = ""
        if exc.code in (401, 403):
            raise LLMError(
                "The API key was rejected. Check DEEPSEEK_API_KEY and try again."
            ) from exc
        if exc.code == 429:
            raise LLMError("The API rate limit was reached. Wait a moment and retry.") from exc
        raise LLMError(f"The API returned HTTP {exc.code}. {detail}".strip()) from exc
    except urllib.error.URLError as exc:
        raise LLMError(
            "Could not reach the API. Check the network connection, or turn the AI cast "
            "list off to stay fully local."
        ) from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise LLMError("The API returned a response that is not JSON.") from exc


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        text = text[start + 3 : end] if end > start else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("The model did not return the expected JSON object.")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model returned invalid JSON: {exc.msg}.") from exc
    if not isinstance(data, dict):
        raise LLMError("The model returned JSON that is not an object.")
    return data


def _normalise_entries(
    data: dict,
) -> tuple[list[str], dict[str, str], list[str], list[str]]:
    raw_characters = data.get("characters") or []
    characters: list[str] = []
    aliases: dict[str, str] = {}
    alias_surfaces: list[str] = []
    seen: set[str] = set()

    if isinstance(raw_characters, str):
        raw_characters = [raw_characters]
    if isinstance(raw_characters, dict):
        raw_characters = [
            {"name": name, "aliases": variants} for name, variants in raw_characters.items()
        ]

    for entry in raw_characters:
        if isinstance(entry, str):
            name, variants = entry, []
        elif isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("character") or ""
            variants = entry.get("aliases") or entry.get("variants") or []
            if isinstance(variants, str):
                variants = [variants]
        else:
            continue

        canonical = normalize_surface(str(name))
        if not canonical or canonical.casefold() in seen:
            continue
        seen.add(canonical.casefold())
        characters.append(canonical)
        for variant in variants:
            alias = normalize_surface(str(variant))
            if not alias or alias.casefold() == canonical.casefold():
                continue
            key = canonical_key(alias)
            if key in aliases:
                continue
            aliases[key] = canonical
            alias_surfaces.append(alias)

    # Drop aliases that are a bare surname shared by several characters: merging those
    # would silently join a family into one node.
    surname_counts: dict[str, int] = {}
    for character in characters:
        tokens = name_tokens(character)
        if len(tokens) >= 2:
            surname_counts[tokens[-1]] = surname_counts.get(tokens[-1], 0) + 1
    for key in list(aliases):
        tokens = name_tokens(key)
        if len(tokens) == 1 and surname_counts.get(tokens[0], 0) > 1:
            aliases.pop(key, None)
    alias_surfaces = [surface for surface in alias_surfaces if canonical_key(surface) in aliases]

    places: list[str] = []
    raw_places = data.get("places") or []
    if isinstance(raw_places, (str, Mapping)):
        raw_places = [raw_places]
    for place in raw_places:
        cleaned = normalize_surface(str(place))
        if cleaned:
            places.append(cleaned)

    return characters, aliases, alias_surfaces, places


def extract_cast(
    text: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_excerpt: int = DEFAULT_MAX_EXCERPT,
    chapters: Sequence[Chapter] | None = None,
) -> CastResult:
    """Ask the model for a cast list and return it in pipeline-ready form."""

    key = (api_key or "").strip()
    if not key:
        raise LLMError(
            "No API key was provided. Set DEEPSEEK_API_KEY or paste a key in the app."
        )
    excerpt = build_excerpt(text, chapters, limit=max_excerpt)
    if not excerpt.strip():
        raise LLMError("There is no text to send. Add a book first.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.replace("{excerpt}", excerpt)},
        ],
        "temperature": 0,
        "max_tokens": 4000,
        "stream": False,
    }
    response = _post_json(
        base_url.rstrip("/") + "/chat/completions", payload, key, timeout
    )
    try:
        raw = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("The API response did not contain a message.") from exc

    characters, aliases, alias_surfaces, places = _normalise_entries(_extract_json(raw))
    if not characters:
        raise LLMError(
            "The model did not find any character names in the excerpt. Try a more "
            "narrative text, or leave the AI cast list off."
        )
    return CastResult(
        characters=characters[:DEFAULT_MAX_CHARACTERS],
        aliases=aliases,
        places=places,
        alias_surfaces=alias_surfaces,
        model=str(response.get("model", model)),
        usage=dict(response.get("usage") or {}),
        excerpt_characters=len(excerpt),
        raw_response=raw,
    )


def apply_cast(
    cast: CastResult,
    aliases: Mapping[str, str],
    manual_names: Iterable[str],
    excluded_names: Iterable[str],
) -> tuple[dict[str, str], list[str], list[str]]:
    """Merge a cast list into the analysis inputs.

    Explicit entries written by the user win; the model only fills the gaps. Every
    canonical name becomes a literal search term, which is what recovers characters
    that spaCy missed, and every alias becomes an alias-table entry.
    """

    merged_aliases = {normalize_surface(key).casefold(): value for key, value in aliases.items()}
    for alias, canonical in cast.aliases.items():
        merged_aliases.setdefault(alias, canonical)

    manual = list(manual_names)
    seen = {item.casefold() for item in manual}
    for name in cast.characters:
        if name.casefold() not in seen:
            manual.append(name)
            seen.add(name.casefold())
    for surface in cast.alias_surfaces:
        if surface.casefold() not in seen:
            manual.append(surface)
            seen.add(surface.casefold())

    excluded = list(excluded_names)
    excluded_seen = {item.casefold() for item in excluded}
    for place in cast.places:
        if place.casefold() not in excluded_seen:
            excluded.append(place)
            excluded_seen.add(place.casefold())

    return merged_aliases, manual, excluded
