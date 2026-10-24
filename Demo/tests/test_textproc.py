"""Tests for input decoding, validation, and segmentation."""

from __future__ import annotations

import pytest

from character_network.textproc import (
    Chapter,
    InputValidationError,
    decode_upload,
    detect_language,
    find_chapters,
    paragraph_ranges,
    slice_text,
    validate_text,
    window_ranges,
)

ENGLISH = (
    "Elizabeth Bennet walked to Netherfield to see her sister Jane. Mr. Darcy said little, "
    "but he watched her carefully. Later, Charlotte Lucas asked Elizabeth what she thought "
    "of the evening, and Elizabeth admitted that she had been surprised by his manners."
)


def test_decodes_utf8_and_strips_bom():
    assert decode_upload("\ufeffHello world".encode("utf-8"), "book.txt") == "Hello world"


def test_decodes_utf16_with_bom():
    payload = "Hello world".encode("utf-16")
    assert "Hello world" in decode_upload(payload, "book.txt")


def test_decodes_windows_encoding():
    payload = "café résumé".encode("cp1252")
    assert "café" in decode_upload(payload, "book.txt")


def test_rejects_empty_binary_and_unsupported_files():
    with pytest.raises(InputValidationError, match="empty"):
        decode_upload(b"", "book.txt")
    with pytest.raises(InputValidationError, match="binary"):
        decode_upload(b"PK\x03\x04\x00\x00binary", "book.txt")
    with pytest.raises(InputValidationError, match="not supported"):
        decode_upload(b"%PDF-1.7 text", "book.pdf")


def test_validate_text_accepts_english_prose():
    assert validate_text(ENGLISH).startswith("Elizabeth Bennet")


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "non-empty"),
        ("   \n\t  ", "non-empty"),
        ("Elizabeth met Darcy.", "too short"),
        ("1234 5678 90!@# $%^ &*() " * 4, "does not look like prose"),
    ],
)
def test_validate_text_rejects_unusable_input(text, message):
    with pytest.raises(InputValidationError, match=message):
        validate_text(text, require_english=False)


def test_validate_text_rejects_non_english_script():
    chinese = (
        "伊丽莎白·班纳特是一位年轻女子，她住在赫特福德郡的浪搏恩庄园，"
        "与她的四个姐妹和父母一起生活。达西先生是一位富有的绅士，"
        "他在舞会上显得骄傲而冷淡，伊丽莎白因此对他产生了偏见。"
    )
    with pytest.raises(InputValidationError, match="English only"):
        validate_text(chinese)


def test_validate_text_rejects_text_over_the_limit():
    oversized = (ENGLISH + " ") * 40_000
    with pytest.raises(InputValidationError, match="longer than"):
        validate_text(oversized, require_english=False)


def test_detect_language_returns_en_for_english():
    assert detect_language(ENGLISH) == "en"


def test_paragraph_ranges_split_on_blank_lines():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
    ranges = paragraph_ranges(text)
    assert len(ranges) == 3
    assert text[ranges[1][0] : ranges[1][1]].startswith("Second")


def test_paragraph_ranges_split_wrapped_text_without_blank_lines():
    line = "This is one line of a hard-wrapped book chapter with enough words.\n"
    text = line * 60
    ranges = paragraph_ranges(text)
    assert len(ranges) > 1


def test_find_chapters_detects_headings_and_ranges():
    text = (
        "CHAPTER I\n\n" + "First chapter text. " * 40 + "\n\n"
        "CHAPTER II\n\n" + "Second chapter text. " * 40 + "\n\n"
        "CHAPTER III\n\n" + "Third chapter text. " * 40
    )
    chapters = find_chapters(text)
    assert [chapter.title for chapter in chapters] == ["Chapter I", "Chapter II", "Chapter III"]
    assert chapters[0].start < chapters[1].start < chapters[2].start
    assert chapters[2].end == len(text)


def test_find_chapters_returns_empty_without_headings():
    assert find_chapters("Just a paragraph. " * 30) == []


def test_slice_text_uses_one_based_inclusive_range():
    text = (
        "CHAPTER I\n\n" + "First. " * 40 + "\nCHAPTER II\n\n" + "Second. " * 40 + "\nCHAPTER III\n\nThird. " * 40
    )
    chapters = find_chapters(text)
    sliced = slice_text(text, chapters, 2, 2)
    assert sliced.startswith("CHAPTER II")
    assert "First." not in sliced
    assert "Third." not in sliced


def test_slice_text_clamps_out_of_range_values():
    chapters = [Chapter(1, "Chapter I", 0, 100)]
    text = "x" * 100
    assert slice_text(text, chapters, 5, 9) == text


def test_window_ranges_overlap_by_half():
    windows = window_ranges(token_count=100, window_size=20)
    assert windows[0] == (0, 20)
    assert windows[1] == (10, 30)
    assert windows[-1][1] == 100


def test_window_size_must_be_reasonable():
    with pytest.raises(InputValidationError, match="at least 10"):
        window_ranges(token_count=50, window_size=5)
