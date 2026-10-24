"""Remove Project Gutenberg boilerplate so the evaluation uses the novel text only.

Usage::

    python evaluation/prepare_data.py

Reads the raw downloads in ``evaluation/data/*_raw.txt`` and writes
``evaluation/data/<book>.txt`` with the licence header and footer stripped.
Both novels were published before 1900 and are in the public domain; the Gutenberg
licence applies to the header and footer, which this script removes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
DATA_DIR = EVALUATION_DIR / "data"

BOOKS = {
    "pride_and_prejudice_raw.txt": "pride_and_prejudice.txt",
    "jane_eyre_raw.txt": "jane_eyre.txt",
}

START_PATTERN = re.compile(r"\*\*\*\s*START OF TH[EI]S? PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE)
END_PATTERN = re.compile(r"\*\*\*\s*END OF TH[EI]S? PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE)
FIRST_CHAPTER = re.compile(
    r"^[ \t]*(?:chapter|part|book)\s+(?:[IVXLCDM]+|\d{1,3})[ \t]*[.:\]]*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: Plate captions such as "[Illustration: ... [_Copyright 1894 by George Allen._]]"
#: interrupt sentences and inject a publisher name into the text. The close bracket is
#: searched within a bounded window so that one unterminated caption cannot swallow
#: the rest of the book.
ILLUSTRATION_START = "[Illustration"


def strip_illustrations(text: str) -> tuple[str, int]:
    """Remove plate captions and their copyright lines; return the text and count."""

    pieces: list[str] = []
    cursor = 0
    removed = 0
    while True:
        start = text.find(ILLUSTRATION_START, cursor)
        if start == -1:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor:start])
        close = text.find("]]", start, min(len(text), start + 900))
        if close != -1:
            cursor = close + 2
        else:
            close = text.find("]", start, min(len(text), start + 300))
            cursor = close + 1 if close != -1 else start + len(ILLUSTRATION_START)
        removed += 1
    return "".join(pieces), removed


def strip_gutenberg(text: str) -> tuple[str, bool]:
    """Return the novel body and whether a Gutenberg marker was found."""

    start = START_PATTERN.search(text)
    end = END_PATTERN.search(text)
    changed = bool(start or end)
    body = text[start.end() if start else 0 : end.start() if end else len(text)]
    return body.strip("\n ") + "\n", changed


def strip_front_matter(text: str) -> tuple[str, bool]:
    """Cut an editor's introduction before the first chapter heading.

    The Gutenberg edition of *Pride and Prejudice* starts with a critical essay that
    names characters from other novels. Cutting at the first chapter heading keeps the
    evaluation focused on the novel itself.
    """

    match = FIRST_CHAPTER.search(text)
    if not match or match.start() > len(text) * 0.25:
        return text, False
    return text[match.start() :].lstrip("\n"), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    arguments = parser.parse_args()

    for raw_name, clean_name in BOOKS.items():
        raw_path = arguments.data_dir / raw_name
        if not raw_path.exists():
            print(f"skip {raw_name}: not found (download it from Project Gutenberg first)")
            continue
        text = raw_path.read_text(encoding="utf-8-sig")
        body, changed = strip_gutenberg(text)
        body, trimmed = strip_front_matter(body)
        body, illustrations = strip_illustrations(body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip("\n") + "\n"
        out_path = arguments.data_dir / clean_name
        out_path.write_text(body, encoding="utf-8", newline="\n")
        print(
            f"{clean_name}: {len(body):,} characters, "
            f"{illustrations} illustration blocks removed"
            f"{', introduction trimmed' if trimmed else ''}"
            f"{'' if changed else ' (no Gutenberg marker)'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
