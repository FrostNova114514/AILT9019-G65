"""Core analysis tools for the Character Relationship Network application.

The public entry point is :func:`character_network.analysis.analyze_text`, which
turns an English text into a weighted NetworkX graph plus the mentions, merge audit
trail, and chapter structure that the interface displays.
"""

from .analysis import (
    AnalysisResult,
    InputValidationError,
    analyze_text,
    detect_chapters,
    slice_to_chapters,
)
from .extraction import DEFAULT_MODEL, Mention, load_pipeline
from .graph import build_graph, chapter_presence, count_cooccurrences, graph_metrics
from .names import (
    Merge,
    aliases_to_csv,
    aliases_to_json,
    normalize_surface,
    parse_alias_json,
    parse_alias_table,
    resolve_name,
    strip_titles,
)
from .presets import get_preset, preset_labels
from .textproc import Chapter, decode_upload, detect_language, validate_text
from .visualization import (
    chapter_heatmap,
    create_network_figure,
    edge_table,
    figure_to_html,
    node_table,
    render_png,
)

__all__ = [
    "AnalysisResult",
    "Chapter",
    "DEFAULT_MODEL",
    "InputValidationError",
    "Merge",
    "Mention",
    "aliases_to_csv",
    "aliases_to_json",
    "analyze_text",
    "build_graph",
    "chapter_heatmap",
    "chapter_presence",
    "count_cooccurrences",
    "create_network_figure",
    "decode_upload",
    "detect_chapters",
    "detect_language",
    "edge_table",
    "figure_to_html",
    "get_preset",
    "graph_metrics",
    "load_pipeline",
    "node_table",
    "normalize_surface",
    "parse_alias_json",
    "parse_alias_table",
    "preset_labels",
    "render_png",
    "resolve_name",
    "slice_to_chapters",
    "strip_titles",
    "validate_text",
]
