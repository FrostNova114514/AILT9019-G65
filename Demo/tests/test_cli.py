"""Tests for the headless command-line runner."""

from __future__ import annotations

from pathlib import Path

from cli import main

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "sample_story.txt"


def test_cli_writes_every_artifact(tmp_path, capsys):
    exit_code = main(
        [
            "--input",
            str(SAMPLE_PATH),
            "--outdir",
            str(tmp_path),
            "--preset",
            "Pride and Prejudice",
            "--quiet",
        ]
    )
    assert exit_code == 0
    for name in (
        "network.html",
        "network.png",
        "characters.csv",
        "relationships.csv",
        "summary.json",
        "aliases.json",
    ):
        path = tmp_path / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name


def test_cli_prints_a_summary(tmp_path, capsys):
    assert main(["--input", str(SAMPLE_PATH), "--outdir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Characters:" in output
    assert "Elizabeth Bennet" in output


def test_cli_rejects_unusable_input(tmp_path, capsys):
    bad = tmp_path / "chinese.txt"
    bad.write_text(
        "伊丽莎白·班纳特是一位年轻女子，她住在赫特福德郡的浪搏恩庄园，与她的四个姐妹生活。",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert main(["--input", str(bad), "--outdir", str(out)]) == 2
    assert not out.exists()


def test_cli_chapter_range(tmp_path):
    chaptered = tmp_path / "book.txt"
    chaptered.write_text(
        "CHAPTER I\n\n"
        + "Charlotte Lucas spoke with William Collins about the evening. "
        "William Collins bowed to Charlotte Lucas. " * 4
        + "\n\nCHAPTER II\n\n"
        + "George Wickham arrived with Fitzwilliam Darcy. "
        "Fitzwilliam Darcy greeted George Wickham. " * 4,
        encoding="utf-8",
    )
    assert main(
        ["--input", str(chaptered), "--outdir", str(tmp_path / "out"), "--chapters", "2", "--quiet"]
    ) == 0
    summary = (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    assert "Wickham" in summary
    assert "Collins" not in summary
