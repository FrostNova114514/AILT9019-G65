"""Command-line runner for the Character Relationship Network.

The web application is the primary interface; this CLI runs the same pipeline without
a browser so that results can be reproduced in a script or in a test.

Examples::

    python cli.py --input ../evaluation/data/pride_and_prejudice.txt \
        --preset "Pride and Prejudice" --chapters 1-6 --outdir output

    python cli.py --input my_novel.txt --mode window --window 200 --min-weight 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from character_network.analysis import (
    AnalysisResult,
    InputValidationError,
    analyze_text,
    detect_chapters,
    slice_to_chapters,
)
from character_network.extraction import ModelLoadError, load_pipeline
from character_network.graph import layout_graph
from character_network.names import aliases_to_json, parse_alias_table
from character_network.presets import get_preset_raw
from character_network.textproc import decode_upload
from character_network.visualization import (
    create_network_figure,
    edge_table,
    figure_to_html,
    node_table,
    render_png,
)


def _parse_chapter_range(raw: str | None, chapter_count: int) -> tuple[int, int]:
    if not raw:
        return 1, max(chapter_count, 1)
    try:
        if "-" in raw:
            first, last = raw.split("-", 1)
            return int(first), int(last)
        value = int(raw)
        return value, value
    except ValueError as exc:
        raise SystemExit(f"--chapters must look like '3' or '3-8', not {raw!r}") from exc


def _split(raw: str | None) -> list[str]:
    return [item.strip() for item in (raw or "").replace(";", ",").split(",") if item.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(result: AnalysisResult) -> dict:
    return {
        "settings": result.settings,
        "metrics": result.metrics,
        "chapters": [chapter.title for chapter in result.chapters],
        "characters": node_table(result.graph).to_dict(orient="records"),
        "relationships": edge_table(result.graph).to_dict(orient="records"),
        "auto_merges": [
            {"variant": merge.variant, "canonical": merge.canonical, "mentions": merge.mentions}
            for merge in result.merges
        ],
        "warnings": result.warnings,
        "elapsed_seconds": result.elapsed_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 plain-text file to analyse")
    parser.add_argument("--outdir", type=Path, default=Path("output"), help="Directory for the results")
    parser.add_argument("--mode", choices=["paragraph", "window"], default="paragraph")
    parser.add_argument("--window", type=int, default=120, help="Window size in tokens (window mode)")
    parser.add_argument("--min-weight", type=int, default=1, help="Minimum co-occurrence count")
    parser.add_argument("--min-mentions", type=int, default=2, help="Minimum mentions per character")
    parser.add_argument("--max-characters", default="40", help="'all' or a number of characters to draw")
    parser.add_argument("--preset", default=None, help="Alias preset name, e.g. 'Pride and Prejudice'")
    parser.add_argument("--alias", type=Path, default=None, help="Alias table (.json or .csv)")
    parser.add_argument("--manual-names", default="", help="Comma-separated extra names to search for")
    parser.add_argument("--exclude", default="", help="Comma-separated names to ignore")
    parser.add_argument("--chapters", default=None, help="Chapter range such as '1-6'")
    parser.add_argument("--no-auto-merge", action="store_true", help="Disable automatic partial-name merges")
    parser.add_argument("--no-png", action="store_true", help="Skip the static PNG export")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    raw = arguments.input.read_bytes()
    text = decode_upload(raw, arguments.input.name)

    chapters = detect_chapters(text)
    if chapters:
        first, last = _parse_chapter_range(arguments.chapters, len(chapters))
        text = slice_to_chapters(text, chapters, first, last)
        if not arguments.quiet and arguments.chapters:
            print(f"Using chapters {first}-{last} of {len(chapters)} detected chapters.")

    aliases: dict[str, str] = {}
    if arguments.preset:
        aliases.update(parse_alias_table(json.dumps(get_preset_raw(arguments.preset))))
    if arguments.alias:
        aliases.update(parse_alias_table(arguments.alias.read_text(encoding="utf-8-sig")))

    maximum = 0 if str(arguments.max_characters).casefold() == "all" else int(arguments.max_characters)

    try:
        nlp = load_pipeline()
        result = analyze_text(
            text,
            nlp,
            aliases=aliases,
            manual_names=_split(arguments.manual_names),
            excluded_names=_split(arguments.exclude),
            mode=arguments.mode,
            window_size=arguments.window,
            minimum_edge_weight=arguments.min_weight,
            minimum_mentions=arguments.min_mentions,
            maximum_characters=maximum,
            auto_merge=not arguments.no_auto_merge,
        )
    except (InputValidationError, ModelLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    arguments.outdir.mkdir(parents=True, exist_ok=True)
    positions = layout_graph(result.graph)
    figure = create_network_figure(result.graph, positions=positions)
    (arguments.outdir / "network.html").write_text(figure_to_html(figure), encoding="utf-8")
    _write_csv(arguments.outdir / "characters.csv", node_table(result.graph).to_dict(orient="records"))
    _write_csv(
        arguments.outdir / "relationships.csv",
        edge_table(result.graph).to_dict(orient="records"),
    )
    (arguments.outdir / "summary.json").write_text(
        json.dumps(_summary(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not arguments.no_png:
        (arguments.outdir / "network.png").write_bytes(
            render_png(result.graph, positions=positions)
        )
    (arguments.outdir / "aliases.json").write_text(aliases_to_json(aliases), encoding="utf-8")

    if not arguments.quiet:
        print()
        print(f"Characters: {result.graph.number_of_nodes()}   "
              f"Relationships: {result.graph.number_of_edges()}   "
              f"Mentions: {len(result.mentions)}   Units: {result.segment_count}")
        print(f"Analysed in {result.elapsed_seconds:.1f} s")
        print()
        print(f"{'#':>3}  {'character':32s} mentions  weighted degree")
        drawn = [
            (name, count)
            for name, count in result.character_counts.most_common()
            if name in result.graph
        ]
        for index, (name, count) in enumerate(drawn[:15], start=1):
            weighted = result.graph.nodes[name].get("weighted_degree", 0)
            print(f"{index:>3}  {name:32.32s} {count:8d}  {weighted:>15}")
        if len(drawn) > 15:
            print(f"     … {len(drawn) - 15} more characters in characters.csv")
        if result.merges:
            print()
            print(f"Automatic merges: {len(result.merges)} (see summary.json)")
        for warning in result.warnings:
            print(f"warning: {warning}")
        print()
        print(f"Wrote results to {arguments.outdir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
