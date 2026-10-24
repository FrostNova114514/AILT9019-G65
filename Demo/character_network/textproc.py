"""Decoding, validation, and segmentation of English long-form text.

The functions here are deliberately conservative: anything that cannot be read as
plain English prose is rejected with a message that tells the user what to do next,
rather than producing a meaningless graph.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

MAX_CHARACTERS = 2_000_000
MIN_CHARACTERS = 40
MIN_WORDS = 8

#: Characters of readable prose are required before NER is attempted.
_SUPPORTED_EXTENSIONS = (".txt", ".text", ".md")

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

#: Scripts that cannot contain an English novel.
_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "Chinese": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)),
    "Japanese": ((0x3040, 0x30FF),),
    "Korean": ((0xAC00, 0xD7AF), (0x1100, 0x11FF)),
    "Cyrillic": ((0x0400, 0x04FF),),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "Hebrew": ((0x0590, 0x05FF),),
    "Devanagari": ((0x0900, 0x097F),),
    "Thai": ((0x0E00, 0x0E7F),),
    "Greek": ((0x0370, 0x03FF),),
}

#: Human-readable labels for the codes ``langdetect`` returns.
_LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
}

#: A blank line separates paragraphs. Gutenberg novels are hard-wrapped inside a
#: paragraph, so a single newline must *not* split a unit.
_PARAGRAPH_SEPARATOR = re.compile(r"\n[ \t]*\n+")


class InputValidationError(ValueError):
    """Raised when input cannot be processed safely or meaningfully."""


@dataclass(frozen=True)
class Chapter:
    """A detected chapter heading and the character range it covers."""

    number: int
    title: str
    start: int
    end: int


def _looks_binary(data: bytes) -> bool:
    """Return True when the payload contains NUL bytes or many control characters."""

    if b"\x00" in data[:4096]:
        return True
    sample = data[:8192]
    if not sample:
        return False
    control = sum(
        1
        for byte in sample
        if byte < 32 and byte not in (9, 10, 13, 12)
    )
    return control / len(sample) > 0.02


def _decode_with_fallback(data: bytes) -> str:
    """Decode bytes as UTF-8, UTF-16, or a Windows single-byte encoding."""

    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        candidates = ("utf-16", "utf-8-sig", "cp1252", "latin-1")
    else:
        candidates = ("utf-8-sig", "utf-16", "cp1252", "latin-1")

    for encoding in candidates:
        try:
            decoded = data.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        # Reject mojibake produced by a single-byte codec on binary payloads.
        printable = sum(1 for char in decoded[:8192] if char.isprintable() or char in "\n\r\t")
        if decoded and printable / max(len(decoded[:8192]), 1) < 0.85:
            continue
        # Normalise line endings so chapter headings match on Windows-authored files.
        return decoded.replace("\r\n", "\n").replace("\r", "\n")
    raise InputValidationError(
        "The uploaded file could not be decoded as text. Save it as UTF-8 plain text and try again."
    )


def decode_upload(data: bytes, filename: str = "") -> str:
    """Decode an uploaded file, rejecting binary and non-plain-text payloads."""

    if not data:
        raise InputValidationError("The uploaded file is empty. Please choose a non-empty .txt file.")
    name = (filename or "").strip().lower()
    suffix = ""
    if "." in name:
        suffix = name[name.rfind(".") :]
    if suffix and suffix not in _SUPPORTED_EXTENSIONS:
        raise InputValidationError(
            f"'{suffix}' files are not supported. Please upload an English plain-text .txt file "
            "(copy the text out of PDF or Word first)."
        )
    # A UTF-16 file legitimately contains NUL bytes, so its BOM is checked first.
    utf16_bom = data[:2] in (b"\xff\xfe", b"\xfe\xff")
    if not utf16_bom and _looks_binary(data):
        raise InputValidationError(
            "This looks like a binary file rather than plain text. "
            "Please upload an English plain-text .txt file."
        )
    return _decode_with_fallback(data)


def detect_language(text: str) -> str:
    """Return an ISO-639-1 code, or ``"unknown"`` when detection is not possible."""

    sample = text[:20_000]
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
    except ImportError:  # pragma: no cover - langdetect is a declared dependency
        return "unknown"

    DetectorFactory.seed = 0
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def _dominant_non_latin_script(text: str) -> tuple[str, float]:
    """Return the dominant non-Latin script and the share of letters it covers."""

    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "", 0.0
    counts: dict[str, int] = {}
    for char in letters:
        code = ord(char)
        if code < 0x0250:  # Basic Latin + Latin-1 Supplement + Latin Extended-A
            continue
        for script, ranges in _SCRIPT_RANGES.items():
            if any(low <= code <= high for low, high in ranges):
                counts[script] = counts.get(script, 0) + 1
                break
    if not counts:
        return "", 0.0
    script, count = max(counts.items(), key=lambda item: item[1])
    return script, count / len(letters)


def validate_text(text: str, *, require_english: bool = True) -> str:
    """Validate and normalize text before analysis.

    Raises :class:`InputValidationError` with an actionable message when the input
    is empty, binary, too short, too long, or not English.
    """

    cleaned = unicodedata.normalize("NFC", text or "").replace("\x00", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        raise InputValidationError(
            "Please upload or paste a non-empty English text before generating the network."
        )

    letters = sum(1 for char in cleaned if char.isalpha())
    if letters / max(len(cleaned), 1) < 0.35:
        raise InputValidationError(
            "The input does not look like prose: fewer than 35% of its characters are letters. "
            "Please provide an English plain-text book."
        )

    if len(cleaned) < MIN_CHARACTERS:
        raise InputValidationError(
            "The text is too short to build a meaningful relationship network. "
            "Paste at least a few paragraphs of English prose."
        )
    if len(cleaned) > MAX_CHARACTERS:
        raise InputValidationError(
            f"The text is longer than the {MAX_CHARACTERS:,}-character limit. "
            "Split the book into parts and analyse them one at a time."
        )

    # Check the language before counting words: a Chinese paragraph has few
    # whitespace-separated "words", and "English only" is the useful message.
    if require_english:
        script, share = _dominant_non_latin_script(cleaned[:40_000])
        if script and share > 0.05:
            raise InputValidationError(
                f"The text is mostly {script} script, but this version supports English only. "
                "Please provide an English text."
            )
        language = detect_language(cleaned)
        if language not in ("en", "unknown"):
            raise InputValidationError(
                f"The text appears to be {_LANGUAGE_NAMES.get(language, language)}, "
                "but this version supports English only."
            )

    words = WORD_PATTERN.findall(cleaned)
    if len(words) < MIN_WORDS:
        raise InputValidationError(
            "The text does not contain enough words for character analysis."
        )
    return cleaned


def paragraph_ranges(text: str, *, split_long_blocks: bool = True) -> list[tuple[int, int]]:
    """Return the character ranges of the paragraphs in ``text``.

    Paragraphs are separated by blank lines. When a document uses no blank lines at
    all (common in pasted text), single newlines and hard character limits are used as
    a fallback so that co-occurrence is still measured over local context.
    """

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_SEPARATOR.finditer(text):
        ranges.append((cursor, match.start()))
        cursor = match.end()
    ranges.append((cursor, len(text)))

    blocks = [(start, end) for start, end in ranges if text[start:end].strip()]
    if not blocks:
        return [(0, len(text))]

    average = sum(end - start for start, end in blocks) / len(blocks)
    if not split_long_blocks or average <= 3000:
        return blocks

    # No paragraph breaks in use: rebuild units from blank lines *or* single newlines.
    lines: list[tuple[int, int]] = []
    for line_match in re.finditer(r"[^\n]+", text):
        if line_match.group().strip():
            lines.append((line_match.start(), line_match.end()))
    if len(lines) < 2:
        return blocks

    merged: list[tuple[int, int]] = []
    start, end = lines[0]
    for line_start, line_end in lines[1:]:
        if end - start > 1200:
            merged.append((start, end))
            start, end = line_start, line_end
        else:
            end = line_end
    merged.append((start, end))
    return merged


_CHAPTER_HEADING = re.compile(
    r"^[ \t]{0,8}"
    r"(?:(?P<kind>chapter|part|book|volume|act|scene)\s+"
    r"(?P<label>[IVXLCDM]{1,8}|\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty))"
    r"[ \t]*(?:[.:]|\]|$)[^\n]{0,60}$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def find_chapters(text: str) -> list[Chapter]:
    """Detect chapter headings and return contiguous chapter ranges.

    Returns an empty list when the text has no recognisable chapter structure.
    """

    matches = [
        match
        for match in _CHAPTER_HEADING.finditer(text)
        if len(match.group(0).strip()) <= 70
    ]
    if len(matches) < 2:
        return []

    chapters: list[Chapter] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if end - start < 200:
            continue
        kind = match.group("kind").strip().capitalize()
        label = match.group("label").strip().upper()
        chapters.append(
            Chapter(number=len(chapters) + 1, title=f"{kind} {label}", start=start, end=end)
        )
    if len(chapters) < 2:
        return []
    return chapters


def chapter_titles(chapters: Sequence[Chapter]) -> list[str]:
    """Return short labels such as ``"Chapter I"`` for display."""

    return [chapter.title for chapter in chapters]


def slice_text(text: str, chapters: Sequence[Chapter], first: int, last: int) -> str:
    """Return the text covered by chapters ``first`` through ``last`` (1-based, inclusive)."""

    if not chapters:
        return text
    first = max(1, min(first, len(chapters)))
    last = max(first, min(last, len(chapters)))
    start = chapters[first - 1].start
    end = chapters[last - 1].end
    return text[start:end]


def window_ranges(token_count: int, window_size: int) -> list[tuple[int, int]]:
    """Return overlapping token windows covering ``token_count`` tokens."""

    if window_size < 10:
        raise InputValidationError("The sliding-window size must be at least 10 tokens.")
    stride = max(1, window_size // 2)
    total = max(token_count, 1)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + window_size)
        windows.append((start, end))
        if end >= total:
            break
        start += stride
    return windows
