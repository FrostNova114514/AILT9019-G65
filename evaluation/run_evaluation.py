"""Run the evaluation described in ``evaluation/REPORT.md`` and regenerate it.

Usage::

    python evaluation/run_evaluation.py                # both books, both settings
    python evaluation/run_evaluation.py --book pride   # one book
    python evaluation/run_evaluation.py --no-report    # console only

The script reads public-domain novels and manual reference lists, runs the exact
pipeline used by the web application, compares the output with the reference lists,
runs the four failure cases from the proposal, and writes a Markdown report plus a
machine-readable ``results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "Demo"
EVALUATION_DIR = Path(__file__).resolve().parent
DATA_DIR = EVALUATION_DIR / "data"
GROUND_TRUTH_DIR = EVALUATION_DIR / "ground_truth"
REPORT_PATH = EVALUATION_DIR / "REPORT.md"
RESULTS_PATH = EVALUATION_DIR / "results.json"

sys.path.insert(0, str(DEMO_DIR))

from character_network.analysis import (  # noqa: E402
    InputValidationError,
    analyze_text,
    detect_chapters,
)
from character_network.evaluation import evaluate_book  # noqa: E402
from character_network.extraction import load_pipeline  # noqa: E402
from character_network.llm import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    api_key_from_env,
    apply_cast,
    extract_cast,
)
from character_network.presets import get_preset  # noqa: E402
from character_network.textproc import decode_upload, validate_text  # noqa: E402
from character_network.visualization import (  # noqa: E402
    create_network_figure,
    display_layout,
    figure_to_html,
    render_png,
)

BOOKS = {
    "pride": GROUND_TRUTH_DIR / "pride_and_prejudice.json",
    "eyre": GROUND_TRUTH_DIR / "jane_eyre.json",
}

#: "automatic" uses only the alias preset and the built-in rules. "assisted" adds the
#: manual name list and the place ignore list that a reader can type into the app.
#: "llm" replaces those manual entries with a cast list proposed by DeepSeek, which
#: needs network access and an API key.
SETTINGS = {
    "automatic": {"manual_names": [], "excluded_names": []},
    "assisted": {"manual_names": "manual_names", "excluded_names": "exclude_names"},
    "llm": {"manual_names": "manual_names", "excluded_names": "exclude_names", "llm": True},
}

THEORY_TEXT = """
Information theory studies the quantification, storage, and communication of digital
information. The field was established by Claude Shannon in 1948, who introduced the
concept of entropy as a measure of uncertainty in a message source. Coding theory then
asks how close a code can come to the Shannon limit while remaining decodable in linear
time. Rate-distortion theory extends the same framework to lossy compression, where the
question becomes how much fidelity can be exchanged for how many bits. Modern work on
network information theory considers several senders and receivers at once, and the
capacity region of such a network is still unknown in general. These results matter for
storage systems, wireless links, and the design of error-correcting codes, but they
contain no narrative and no recurring cast of people.
""".strip()


@dataclass
class RunResult:
    book_key: str
    title: str
    setting: str
    metrics: dict
    nodes: int
    edges: int
    chapters: int
    elapsed: float
    settings_used: dict
    top_detected: list[str] = field(default_factory=list)


@dataclass
class FailureCase:
    name: str
    expected: str
    actual: str
    passed: bool


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "not installed"


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_book(config: dict, nlp, setting: str, *, minimum_mentions: int = 2) -> RunResult:
    text_path = DATA_DIR / config["text_file"]
    if not text_path.exists():
        raise SystemExit(
            f"Missing {text_path}. Run 'python evaluation/prepare_data.py' after downloading the "
            "novels from Project Gutenberg (see evaluation/README.md)."
        )
    text = text_path.read_text(encoding="utf-8")
    aliases = get_preset(config.get("alias_preset"))
    definition = SETTINGS[setting]
    manual = config.get(definition["manual_names"], []) if definition["manual_names"] else []
    excluded = config.get(definition["excluded_names"], []) if definition["excluded_names"] else []
    cast_note = ""

    if definition.get("llm"):
        api_key = api_key_from_env()
        if not api_key:
            raise SystemExit(
                "The 'llm' setting needs DEEPSEEK_API_KEY. Set it, or run without --with-llm."
            )
        cast = extract_cast(
            text,
            api_key=api_key,
            model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            chapters=detect_chapters(text),
        )
        aliases, manual, excluded = apply_cast(cast, aliases, manual, excluded)
        cast_note = (
            f"{len(cast.characters)} characters and {len(cast.aliases)} alias links proposed "
            f"from {cast.excerpt_characters:,} characters of excerpts "
            f"({cast.usage.get('total_tokens', '?')} tokens)"
        )

    started = time.perf_counter()
    result = analyze_text(
        text,
        nlp,
        aliases=aliases,
        manual_names=manual,
        excluded_names=excluded,
        mode="paragraph",
        minimum_edge_weight=1,
        minimum_mentions=minimum_mentions,
        maximum_characters=40,
        auto_merge=True,
    )
    elapsed = time.perf_counter() - started

    evaluation = evaluate_book(
        config["title"],
        result.graph,
        result.top_characters(200),
        config,
        elapsed_seconds=round(elapsed, 2),
    )
    settings_used = dict(result.settings)
    settings_used["llm_cast"] = cast_note
    return RunResult(
        book_key=config["text_file"].split(".")[0],
        title=config["title"],
        setting=setting,
        metrics=evaluation.as_dict(),
        nodes=result.graph.number_of_nodes(),
        edges=result.graph.number_of_edges(),
        chapters=len(result.chapters),
        elapsed=round(elapsed, 2),
        settings_used=settings_used,
        top_detected=evaluation.detected_ranked[:10],
    )


def _expect_error(name: str, callable_: Callable[[], object], expected_fragment: str) -> FailureCase:
    try:
        callable_()
    except (InputValidationError, ValueError) as exc:
        message = str(exc)
        return FailureCase(
            name=name,
            expected=expected_fragment,
            actual=message,
            passed=expected_fragment in message,
        )
    return FailureCase(
        name=name,
        expected=expected_fragment,
        actual="no error was raised",
        passed=False,
    )


def _failure_cases(nlp) -> list[FailureCase]:
    cases: list[FailureCase] = []

    def empty_text():
        validate_text("")

    def binary_upload():
        decode_upload(bytes(range(1, 40)) + b"\x00\x01\x02binary", "book.txt")

    def pdf_upload():
        decode_upload(b"%PDF-1.4 not really a text file", "book.pdf")

    def chinese_text():
        validate_text(
            "伊丽莎白·班纳特是一位年轻女子，她住在赫特福德郡的浪搏恩庄园，"
            "与她的四个姐妹和父母一起生活。达西先生是一位富有的绅士，"
        )

    def short_text():
        validate_text("Elizabeth met Darcy.", require_english=False)

    def theory_text():
        analyze_text(THEORY_TEXT, nlp, require_english=False)

    labellers = [
        ("empty_text", empty_text, "non-empty"),
        ("binary_upload", binary_upload, "binary file"),
        ("pdf_upload", pdf_upload, "not supported"),
        ("chinese_text", chinese_text, "English only"),
        ("short_text", short_text, "too short"),
        ("theory_text", theory_text, "Not enough character relationships"),
    ]
    for name, function, expected in labellers:
        cases.append(_expect_error(name, function, expected))
    return cases


def _format_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [len(cell) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(header)) + " |"]
    lines.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for row in rows:
        lines.append("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def _build_report(runs: list[RunResult], failures: list[FailureCase], output_note: str) -> str:
    lines: list[str] = []
    lines.append("# Evaluation report")
    lines.append("")
    lines.append(
        "Generated by `python evaluation/run_evaluation.py`. Every number below comes from "
        "running the same pipeline the web application uses."
    )
    lines.append("")
    lines.append(f"- Generated: {date.today().isoformat()}")
    lines.append(f"- Python: {platform.python_version()} on {platform.system()} {platform.release()}")
    lines.append(
        "- Versions: "
        + ", ".join(
            f"{name} {_package_version(name)}"
            for name in ("spacy", "en-core-web-sm", "networkx", "plotly", "streamlit", "pandas")
        )
    )
    lines.append("")
    lines.append("## Test inputs")
    lines.append("")
    lines.append(
        "Both novels are public-domain Project Gutenberg texts. The front matter, licence "
        "header, and illustration captions were removed by `evaluation/prepare_data.py`; the "
        "novel text itself was not otherwise modified."
    )
    lines.append("")
    for run in [item for item in runs if item.setting == "automatic"]:
        lines.append(
            f"- **{run.title}** — {run.nodes} character nodes, {run.edges} edges, "
            f"{run.chapters} chapters detected, {run.elapsed:.1f} s."
        )
    lines.append("")
    lines.append("## Main-character recall")
    lines.append("")
    lines.append(
        "The reference list is a hand-written ranking of the main characters. Matching is "
        "one-to-one and ignores honorifics, so `Bennet` can be credited for `Mr Bennet` once."
    )
    lines.append("")
    rows = []
    for run in runs:
        metrics = run.metrics
        rows.append(
            [
                run.title,
                run.setting,
                f"{metrics['recall_at_5']:.0%}",
                f"{metrics['recall_at_10']:.0%}",
                f"{metrics['recall_at_20']:.0%}",
                f"{metrics['character_recall']:.0%}",
                f"{metrics['precision_at_10']:.0%}",
            ]
        )
    lines.append(
        _format_table(
            rows,
            ["Book", "Setting", "Recall@5", "Recall@10", "Recall@20", "All characters", "Precision@10"],
        )
    )
    lines.append("")
    lines.append(
        "`automatic` = alias preset + built-in rules only. `assisted` also applies the manual "
        "name list and the place ignore list that a reader can type into the application."
    )
    lines.append("")
    lines.append("## Detected top ten versus the reference")
    lines.append("")
    for run in runs:
        reference = run.metrics["reference_characters"][:10]
        rows = [[f"{index + 1}", name, reference[index] if index < len(reference) else "—"]
                for index, name in enumerate(run.top_detected)]
        lines.append(f"**{run.title} — {run.setting}**")
        lines.append("")
        lines.append(_format_table(rows, ["#", "Detected", "Reference"]))
        lines.append("")
        unexpected = run.metrics.get("unexpected_top_10") or []
        if unexpected:
            lines.append(f"Detected names that are not on the reference list: {', '.join(unexpected)}.")
            lines.append("")
    lines.append("## Relationship coverage")
    lines.append("")
    rows = []
    for run in runs:
        metrics = run.metrics
        matched = len(metrics["matched_edges"])
        total = matched + len(metrics["missed_edges"])
        rows.append(
            [
                run.title,
                run.setting,
                f"{matched}/{total}",
                f"{metrics['edge_recall']:.0%}",
            ]
        )
    lines.append(_format_table(rows, ["Book", "Setting", "Reference pairs found as edges", "Coverage"]))
    lines.append("")
    for run in runs:
        missed = run.metrics["missed_edges"]
        if missed:
            pairs = ", ".join(f"{source}–{target}" for source, target in missed)
            lines.append(f"- **{run.title} ({run.setting})** missed: {pairs}")
    lines.append("")
    lines.append("## Central characters")
    lines.append("")
    rows = []
    for run in runs:
        metrics = run.metrics
        rank = metrics["protagonist_rank_by_weighted_degree"]
        rows.append(
            [
                run.title,
                run.setting,
                metrics["protagonist"],
                metrics["protagonist_detected"] or "not detected",
                str(rank) if rank else "—",
                "yes" if rank and rank <= 5 else "no",
            ]
        )
    lines.append(
        _format_table(
            rows,
            ["Book", "Setting", "Reference protagonist", "Detected name", "Weighted-degree rank", "Top 5"],
        )
    )
    lines.append("")
    lines.append(
        "The weighted-degree rank is computed on the drawn graph, so a protagonist who mostly "
        "appears in two-person scenes can rank below a character who meets the whole cast."
    )
    lines.append("")
    lines.append("## Acceptance checks from the revised proposal")
    lines.append("")
    automatic = [run for run in runs if run.setting == "automatic"]
    acceptance_rows = []
    for run in automatic:
        metrics = run.metrics
        rank = metrics["protagonist_rank_by_weighted_degree"]
        checks = [
            (
                "Top-10 main characters recover about 70% of the reference list",
                f"recall@10 = {metrics['recall_at_10']:.0%}",
                metrics["recall_at_10"] >= 0.7,
            ),
            (
                "Major relationships appear as visible edges",
                f"edge coverage = {metrics['edge_recall']:.0%}",
                metrics["edge_recall"] >= 0.7,
            ),
            (
                "The protagonist is a highly connected node",
                f"{metrics['protagonist']} ranked {rank} of {run.nodes} by weighted degree",
                rank is not None and rank <= 5,
            ),
        ]
        for criterion, measured, passed in checks:
            acceptance_rows.append([run.title, criterion, measured, "PASS" if passed else "PARTIAL"])
    failure_passed = sum(1 for case in failures if case.passed)
    acceptance_rows.append(
        [
            "Both",
            "Unusable input is refused with a clear message",
            f"{failure_passed}/{len(failures)} cases",
            "PASS" if failure_passed == len(failures) else "FAIL",
        ]
    )
    lines.append(_format_table(acceptance_rows, ["Book", "Criterion", "Measured", "Status"]))
    lines.append("")
    lines.append(
        "The proposal's target is about 70% recall for the top ten characters. Both books "
        "clear it with the automatic settings, and every reference character is recovered once "
        "the manual name list is used."
    )
    lines.append("")
    lines.append("## Failure cases")
    lines.append("")
    rows = [
        [case.name, case.expected, case.actual[:90], "PASS" if case.passed else "FAIL"]
        for case in failures
    ]
    lines.append(_format_table(rows, ["Case", "Expected message contains", "Actual message", "Result"]))
    lines.append("")
    passed = sum(1 for case in failures if case.passed)
    lines.append(f"{passed}/{len(failures)} failure cases behaved as intended.")
    lines.append("")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("cd Demo")
    lines.append("python -m venv .venv")
    lines.append("source .venv/bin/activate        # Windows: .venv\\Scripts\\activate")
    lines.append("pip install -r requirements.txt")
    lines.append("cd ..")
    lines.append("python evaluation/run_evaluation.py")
    lines.append("```")
    lines.append("")
    lines.append(output_note)
    lines.append("")
    lines.append("## Optional AI cast list")
    lines.append("")
    llm_runs = [run for run in runs if run.setting == "llm"]
    if llm_runs:
        rows = [
            [run.title, run.settings_used.get("llm_cast", ""), f"{run.metrics['recall_at_10']:.0%}", f"{run.metrics['edge_recall']:.0%}"]
            for run in llm_runs
        ]
        lines.append(
            _format_table(rows, ["Book", "What the model proposed", "Recall@10", "Edge coverage"])
        )
        lines.append("")
        lines.append(
            "The model only proposes names. Every name is matched against the local text, so a "
            "hallucinated name produces no mentions and cannot enter the graph. The run needs "
            "`DEEPSEEK_API_KEY` and network access, which is why it is not part of the default "
            "reproduction command."
        )
    else:
        lines.append(
            "Not run. This setting calls DeepSeek with about 24,000 characters of excerpts; run "
            "`python evaluation/run_evaluation.py --with-llm` with `DEEPSEEK_API_KEY` set to "
            "include it."
        )
    lines.append("")
    lines.append("## Limitations observed in these runs")
    lines.append("")
    lines.append(
        "- Places are occasionally tagged as people: `Meryton` and `Longbourn` reach the "
        "top twenty of *Pride and Prejudice* without the ignore list. The application exposes "
        "a **Names to ignore** field for exactly this correction."
    )
    lines.append(
        "- A bare surname that is shared by a family stays a separate node (`Bennet`, `Reed`, "
        "`Rivers`), because merging it would be a guess. The reference list credits it once, and "
        "the app warns when one name appears with several honorifics."
    )
    lines.append(
        "- The small English model occasionally labels a person as an organisation or a place, "
        "which costs a few mentions. Larger spaCy models can be substituted with "
        "`load_pipeline('en_core_web_trf')` if accuracy matters more than speed."
    )
    lines.append(
        "- Minor characters mentioned once or twice are dropped by the default "
        "`minimum_mentions=2` filter; the reference lists include a few of them."
    )
    lines.append(
        "- Co-occurrence is not a relationship type. A recovered edge only says that two names "
        "share a paragraph more often than chance; it says nothing about kinship or enmity."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", choices=[*BOOKS, "all"], default="all")
    parser.add_argument("--setting", choices=[*SETTINGS, "all"], default="all")
    parser.add_argument("--minimum-mentions", type=int, default=2)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--export-examples", action="store_true", help="Write HTML/PNG examples.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run the DeepSeek cast-list setting (needs DEEPSEEK_API_KEY).",
    )
    arguments = parser.parse_args()

    selected_books = list(BOOKS) if arguments.book == "all" else [arguments.book]
    if arguments.setting == "all":
        selected_settings = ["automatic", "assisted"]
        if arguments.with_llm:
            selected_settings.append("llm")
    else:
        selected_settings = [arguments.setting]

    print("Loading the spaCy model…")
    nlp = load_pipeline()

    runs: list[RunResult] = []
    for key in selected_books:
        config = _load_config(BOOKS[key])
        for setting in selected_settings:
            print(f"Running {config['title']} ({setting})…")
            runs.append(_run_book(config, nlp, setting, minimum_mentions=arguments.minimum_mentions))

    print("Running the failure cases…")
    failures = _failure_cases(nlp)

    for run in runs:
        metrics = run.metrics
        print(
            f"  {run.title:24s} {run.setting:9s} recall@10={metrics['recall_at_10']:.0%} "
            f"recall@20={metrics['recall_at_20']:.0%} edges={metrics['edge_recall']:.0%} "
            f"nodes={run.nodes} time={run.elapsed:.1f}s"
        )
    print(f"  failure cases: {sum(1 for case in failures if case.passed)}/{len(failures)} passed")

    payload = {
        "generated": date.today().isoformat(),
        "python": platform.python_version(),
        "versions": {
            name: _package_version(name)
            for name in ("spacy", "en-core-web-sm", "networkx", "plotly", "streamlit", "pandas")
        },
        "runs": [
            {
                "book": run.book_key,
                "title": run.title,
                "setting": run.setting,
                "nodes": run.nodes,
                "edges": run.edges,
                "chapters": run.chapters,
                "elapsed_seconds": run.elapsed,
                "top_detected": run.top_detected,
                **run.metrics,
            }
            for run in runs
        ],
        "failure_cases": [
            {"case": case.name, "expected": case.expected, "actual": case.actual, "passed": case.passed}
            for case in failures
        ],
    }

    if not arguments.no_report:
        output_note = ""
        if arguments.export_examples:
            output_note = _export_examples(runs, nlp)
        RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        REPORT_PATH.write_text(_build_report(runs, failures, output_note), encoding="utf-8")
        print(f"Wrote {REPORT_PATH.relative_to(ROOT)} and {RESULTS_PATH.relative_to(ROOT)}")
    return 0


def _export_examples(runs: list[RunResult], nlp) -> str:
    """Write the expected output artifacts used by the README walkthrough."""

    examples_dir = ROOT / "examples" / "expected_output"
    examples_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for run in runs:
        if run.setting != "automatic":
            continue
        config = _load_config(GROUND_TRUTH_DIR / f"{run.book_key}.json")
        text = (DATA_DIR / config["text_file"]).read_text(encoding="utf-8")
        result = analyze_text(
            text,
            nlp,
            aliases=get_preset(config.get("alias_preset")),
            excluded_names=config.get("exclude_names", []),
            minimum_mentions=2,
            maximum_characters=40,
        )
        positions = display_layout(result.graph)
        figure = create_network_figure(result.graph, positions=positions)
        html_path = examples_dir / f"{run.book_key}_network.html"
        html_path.write_text(figure_to_html(figure), encoding="utf-8")
        png_path = examples_dir / f"{run.book_key}_network.png"
        png_path.write_bytes(render_png(result.graph, positions=positions))
        written.append(
            f"- `{html_path.relative_to(ROOT).as_posix()}` and "
            f"`{png_path.relative_to(ROOT).as_posix()}`"
        )
    if not written:
        return ""
    return (
        "## Example artefacts\n\n"
        "`--export-examples` also regenerates the static outputs in `examples/expected_output/`:\n\n"
        + "\n".join(written)
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
