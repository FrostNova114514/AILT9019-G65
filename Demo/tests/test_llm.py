"""Tests for the optional DeepSeek cast-list helper.

The network is never touched: ``_post_json`` is monkeypatched, so the tests run offline
and deterministic.
"""

from __future__ import annotations

import json

import pytest

from character_network import llm
from character_network.llm import (
    LLMError,
    apply_cast,
    build_excerpt,
    extract_cast,
)
from character_network.textproc import Chapter, find_chapters


def _response(content: str) -> dict:
    return {
        "model": "deepseek-chat",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "choices": [{"message": {"content": content}}],
    }


def test_extract_cast_parses_json_and_alias_links(monkeypatch):
    payload = json.dumps(
        {
            "characters": [
                {"name": "Elizabeth Bennet", "aliases": ["Lizzy", "Miss Elizabeth"]},
                {"name": "Fitzwilliam Darcy", "aliases": ["Mr. Darcy", "Darcy"]},
            ],
            "places": ["Meryton"],
        }
    )
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: _response(payload))
    cast = extract_cast("Elizabeth met Darcy.", api_key="test-key")

    assert cast.characters == ["Elizabeth Bennet", "Fitzwilliam Darcy"]
    assert cast.aliases["lizzy"] == "Elizabeth Bennet"
    assert cast.aliases["mr darcy"] == "Fitzwilliam Darcy"
    assert cast.places == ["Meryton"]
    assert cast.usage["total_tokens"] == 150


def test_extract_cast_accepts_fenced_and_plain_string_lists(monkeypatch):
    fenced = (
        "```json\n"
        '{"characters": ["Elizabeth Bennet", "Darcy"], "places": "Meryton"}\n'
        "```"
    )
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: _response(fenced))
    cast = extract_cast("text", api_key="k")
    assert cast.characters == ["Elizabeth Bennet", "Darcy"]
    assert cast.places == ["Meryton"]


def test_extract_cast_drops_ambiguous_bare_surnames(monkeypatch):
    payload = json.dumps(
        {
            "characters": [
                {"name": "Elizabeth Bennet", "aliases": ["Bennet", "Lizzy"]},
                {"name": "Jane Bennet", "aliases": ["Bennet"]},
            ]
        }
    )
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: _response(payload))
    cast = extract_cast("text", api_key="k")
    assert "bennet" not in cast.aliases
    assert cast.aliases["lizzy"] == "Elizabeth Bennet"


def test_extract_cast_requires_a_key():
    with pytest.raises(LLMError, match="No API key"):
        extract_cast("text", api_key="")


def test_api_key_falls_back_to_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    secrets = tmp_path / "secrets.toml"
    secrets.write_text(
        '# local key\nDEEPSEEK_API_KEY = "file-key"\n', encoding="utf-8"
    )
    assert llm.api_key_from_env([secrets]) == "file-key"

    secrets.write_text("DEEPSEEK_API_KEY=plain-key\n", encoding="utf-8")
    assert llm.api_key_from_env([secrets]) == "plain-key"

    assert llm.api_key_from_env([tmp_path / "missing.toml"]) == ""


def test_environment_variable_wins_over_the_file(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('DEEPSEEK_API_KEY = "file-key"\n', encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    assert llm.api_key_from_env([secrets]) == "env-key"


def test_extract_cast_rejects_unparseable_answer(monkeypatch):
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: _response("I cannot help."))
    with pytest.raises(LLMError, match="JSON object"):
        extract_cast("text", api_key="k")


def test_extract_cast_rejects_empty_cast(monkeypatch):
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: _response('{"characters": []}'))
    with pytest.raises(LLMError, match="did not find any character"):
        extract_cast("text", api_key="k")


def test_build_excerpt_is_bounded_and_samples_the_whole_book():
    text = "".join(f"Chapter {index}. Elizabeth Bennet met Darcy number {index}. " for index in range(2000))
    excerpt = build_excerpt(text, limit=4000)
    assert len(excerpt) <= 4600
    assert "Darcy number 0" in excerpt
    assert "Darcy number 1999" in excerpt


def test_build_excerpt_labels_chapters_when_available():
    text = (
        "CHAPTER I\n\n" + "Elizabeth met Darcy. " * 60 + "\n\n"
        "CHAPTER II\n\n" + "Jane met Bingley. " * 60
    )
    chapters = find_chapters(text)
    excerpt = build_excerpt(text, chapters, limit=400)
    assert "[Chapter" in excerpt


def test_build_excerpt_returns_short_text_unchanged():
    assert build_excerpt("Short text.", limit=4000) == "Short text."


def test_apply_cast_prefers_user_aliases_and_adds_names():
    cast = llm.CastResult(
        characters=["Elizabeth Bennet"],
        aliases={"lizzy": "Elizabeth Bennet", "eliza": "Elizabeth Bennet"},
        places=["Meryton"],
        alias_surfaces=["Lizzy", "Eliza"],
    )
    aliases, manual, excluded = apply_cast(
        cast,
        {"lizzy": "Beth"},
        ["Charlotte Lucas"],
        ["London"],
    )
    assert aliases["lizzy"] == "Beth"  # the user's mapping wins
    assert aliases["eliza"] == "Elizabeth Bennet"
    assert "Elizabeth Bennet" in manual
    assert "Charlotte Lucas" in manual
    assert "Lizzy" in manual
    assert "Meryton" in excluded and "London" in excluded


def test_apply_cast_does_not_duplicate_manual_names():
    cast = llm.CastResult(
        characters=["Jane Eyre"],
        aliases={},
        places=[],
        alias_surfaces=[],
    )
    _, manual, _ = apply_cast(cast, {}, ["jane eyre"], [])
    assert manual.count("jane eyre") + manual.count("Jane Eyre") == 1


def test_chapter_dataclass_is_used_for_labels():
    chapter = Chapter(1, "Chapter I", 0, 100)
    assert chapter.title == "Chapter I"
