"""Streamlit interface for the Character Relationship Network.

Run with::

    streamlit run app.py

Everything runs locally by default: the text, the spaCy model, NetworkX, and the plots
stay on this machine. The optional AI cast list is the single exception, and it is off
unless the user turns it on; it sends a bounded excerpt to DeepSeek, never the whole
book, and the resulting names are verified against the local text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from character_network.analysis import (
    AnalysisResult,
    InputValidationError,
    analyze_text,
    detect_chapters,
    slice_to_chapters,
)
from character_network.extraction import ModelLoadError, load_pipeline
from character_network.graph import chapter_presence
from character_network.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMError,
    api_key_from_env,
    apply_cast,
    extract_cast,
    key_source,
)
from character_network.names import aliases_to_json, parse_alias_table
from character_network.presets import get_preset_raw, preset_labels
from character_network.textproc import decode_upload
from character_network.visualization import (
    chapter_heatmap,
    create_network_figure,
    display_layout,
    edge_table,
    figure_to_html,
    merge_table,
    node_table,
    render_png,
)

APP_DIRECTORY = Path(__file__).resolve().parent
SAMPLE_PATH = APP_DIRECTORY / "sample_data" / "sample_story.txt"
EXAMPLE_BOOK_PATH = APP_DIRECTORY / "sample_data" / "pride_and_prejudice_chapters_1_3.txt"
STYLE_PATH = APP_DIRECTORY / "assets" / "liquid_glass.css"

COLOR_CHOICES = {"Communities": "community", "Weighted degree": "degree"}

st.set_page_config(
    page_title="Character Relationship Network",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the English language model…")
def language_model():
    """Load the spaCy pipeline once per Streamlit process."""

    return load_pipeline()


def _state() -> dict:
    if "app_state" not in st.session_state:
        st.session_state.app_state = {
            "result": None,
            "figure": None,
            "positions": None,
            "error": None,
            "text_label": "",
        }
    return st.session_state.app_state


def _split_names(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").replace(";", ",").split(",") if item.strip()]


def _summary_payload(result: AnalysisResult) -> dict:
    """Build a JSON-serialisable summary without pandas/numpy scalar types."""

    characters = [
        {
            "character": name,
            "mentions": int(data.get("frequency", 0)),
            "connections": int(data.get("degree", 0)),
            "weighted_degree": int(data.get("weighted_degree", 0)),
            "betweenness": float(data.get("betweenness", 0.0)),
            "community": int(data.get("community", 0)) + 1,
        }
        for name, data in result.graph.nodes(data=True)
    ]
    characters.sort(key=lambda row: (-row["mentions"], row["character"]))
    relationships = [
        {
            "source": source,
            "target": target,
            "cooccurrence_count": int(data.get("weight", 0)),
            "example": data.get("example", ""),
        }
        for source, target, data in result.graph.edges(data=True)
    ]
    relationships.sort(key=lambda row: -row["cooccurrence_count"])
    return {
        "settings": result.settings,
        "metrics": result.metrics,
        "chapters": [
            {"number": chapter.number, "title": chapter.title} for chapter in result.chapters
        ],
        "characters": characters,
        "relationships": relationships,
        "auto_merges": [
            {"variant": merge.variant, "canonical": merge.canonical, "mentions": merge.mentions}
            for merge in result.merges
        ],
        "warnings": result.warnings,
        "elapsed_seconds": result.elapsed_seconds,
    }


state = _state()

st.markdown(
    """
    <section class="hero">
      <div class="hero-badge">Runs locally · no account · no API key</div>
      <h1>Character Relationship Network</h1>
      <p>Turn an English novel into an interactive map of who appears with whom.
      Upload a book, check the detected names, and read the network.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="notice"><span class="notice-icon">i</span><span><strong>Read the edges carefully:</strong>
    an edge means two names appear near each other in the text. It does not establish whether they are
    friends, relatives, enemies, or partners.</span></div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- options

with st.sidebar:
    st.header("Analysis settings")
    mode_label = st.radio(
        "Co-occurrence unit",
        ["Paragraphs", "Sliding token window"],
        help="Characters are connected when their names appear in the same unit.",
    )
    mode = "paragraph" if mode_label == "Paragraphs" else "window"
    window_size = st.slider(
        "Window size (tokens)",
        min_value=40,
        max_value=400,
        value=120,
        step=20,
        disabled=mode != "window",
        help="Only used with the sliding window: each window overlaps the next by half.",
    )
    minimum_edge_weight = st.slider(
        "Minimum co-occurrence count",
        1,
        10,
        1,
        help="Hide pairs that co-occur fewer times. Raises precision, lowers recall.",
    )
    minimum_mentions = st.slider(
        "Minimum mentions per character",
        1,
        10,
        2,
        help="Names mentioned fewer times are usually NER noise. Set to 1 to keep every name.",
    )
    character_limit_label = st.selectbox(
        "Characters shown",
        ["10", "20", "40", "60", "100", "All detected"],
        index=2,
        help="'All detected' is the complete network the proposal asks for; smaller limits are easier to read.",
    )
    maximum_characters = 0 if character_limit_label == "All detected" else int(character_limit_label)
    color_label = st.radio("Node colour", list(COLOR_CHOICES), help="Community = detected clusters.")
    show_labels = st.toggle("Show character names", value=True)
    st.divider()
    st.caption(
        "Defaults give a readable graph. Lower the thresholds to see more names, "
        "raise them to remove weak connections."
    )

# --------------------------------------------------------------------------- input

input_tab, names_tab, guide_tab = st.tabs(["1 · Add text", "2 · Names and aliases", "3 · How to use"])

current_text = ""
current_label = ""
chapters: list = []

with input_tab:
    upload_column, paste_column = st.columns([0.9, 1.1], gap="large")
    with upload_column:
        st.subheader("Upload a book")
        uploaded = st.file_uploader(
            "English plain text (.txt)", type=["txt", "text", "md"], key="uploader"
        )
        st.caption("UTF-8 plain text · up to 2,000,000 characters · files stay on this machine")
        button_columns = st.columns(2)
        if button_columns[0].button("Load short sample", width="stretch"):
            st.session_state["pasted_text"] = SAMPLE_PATH.read_text(encoding="utf-8")
            st.session_state["alias_text"] = aliases_to_json(get_preset_raw("Pride and Prejudice"))
            st.session_state["excluded_names_input"] = "Netherfield, Longbourn, Meryton"
            st.session_state["text_label"] = "Built-in short sample"
            st.rerun()
        if button_columns[1].button(
            "Load novel example", width="stretch", disabled=not EXAMPLE_BOOK_PATH.exists()
        ):
            st.session_state["pasted_text"] = EXAMPLE_BOOK_PATH.read_text(encoding="utf-8")
            st.session_state["alias_text"] = aliases_to_json(get_preset_raw("Pride and Prejudice"))
            st.session_state["excluded_names_input"] = "Netherfield, Longbourn, Meryton"
            st.session_state["text_label"] = "Pride and Prejudice, chapters 1–3"
            st.rerun()

    with paste_column:
        st.subheader("Or paste text")
        pasted = st.text_area(
            "English text",
            key="pasted_text",
            height=250,
            placeholder="Paste several paragraphs or a whole novel chapter here…",
            label_visibility="collapsed",
        )

    uploaded_text = ""
    if uploaded is not None:
        try:
            uploaded_text = decode_upload(uploaded.getvalue(), uploaded.name)
        except InputValidationError as exc:
            st.error(str(exc))

    current_text = uploaded_text or pasted
    current_label = (
        uploaded.name if uploaded_text else st.session_state.get("text_label", "Pasted text")
    )

    if current_text.strip():
        chapters = detect_chapters(current_text)
        st.markdown("#### Reading scope")
        if len(chapters) >= 2:
            st.caption(
                f"{len(chapters)} chapters detected. Choose a range to follow the story "
                "chapter by chapter, or keep the whole book."
            )
            chapter_range = st.slider(
                "Chapters to analyse",
                min_value=1,
                max_value=len(chapters),
                value=(1, len(chapters)),
                help="Chapter headings such as 'CHAPTER IV' are detected automatically.",
                key=f"chapter_range_{len(chapters)}",
            )
        else:
            chapter_range = (1, 1)
            st.caption("No chapter headings detected, so the whole text will be analysed.")
        st.caption(f"{len(current_text):,} characters · about {len(current_text.split()):,} words")
    else:
        chapter_range = (1, 1)

with names_tab:
    with st.expander("AI cast list (optional) — recover characters that spaCy misses"):
        st.markdown(
            "The local model misses minor characters and sometimes tags a person as a place. "
            "This option asks DeepSeek for a **cast list only** (names, aliases, and places to "
            "ignore). Counting, merging, thresholds, and drawing stay local and deterministic."
        )
        ai_enabled = st.toggle(
            "Use DeepSeek to propose characters, aliases, and place names",
            value=bool(api_key_from_env()),
            key="ai_enabled",
        )
        if ai_enabled:
            api_key = st.text_input(
                "DeepSeek API key",
                value=api_key_from_env(),
                type="password",
                key="ai_key",
                help=(
                    "Reads DEEPSEEK_API_KEY, Demo/.streamlit/secrets.toml, or Demo/.env when set. "
                    "The key stays in this browser session and is never written to the repository."
                ),
            )
            source = key_source()
            if source:
                st.caption(f"Using the key from `{source}`.")
            st.caption(
                "Sends about 24,000 characters of evenly spaced excerpts (roughly 7k tokens, "
                "well under one US cent) to DeepSeek. The full book is never uploaded, and every "
                "returned name is checked against the local text before it reaches the graph."
            )
            if st.button("Find characters with AI", width="stretch", key="ai_run"):
                if not current_text.strip():
                    st.error("Add a book first: the AI needs excerpts to read.")
                elif not api_key.strip():
                    st.error(
                        "Paste a DeepSeek API key, or set DEEPSEEK_API_KEY before starting the app."
                    )
                else:
                    try:
                        with st.spinner("Asking DeepSeek for the cast list…"):
                            cast = extract_cast(
                                current_text,
                                api_key=api_key,
                                base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
                                model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
                                chapters=detect_chapters(current_text),
                            )
                        base_aliases = parse_alias_table(
                            st.session_state.get("alias_text", "{}")
                        )
                        merged, manual, excluded = apply_cast(
                            cast,
                            base_aliases,
                            _split_names(st.session_state.get("manual_names_input", "")),
                            _split_names(st.session_state.get("excluded_names_input", "")),
                        )
                        st.session_state["alias_text"] = aliases_to_json(merged)
                        st.session_state["manual_names_input"] = ", ".join(manual)
                        st.session_state["excluded_names_input"] = ", ".join(excluded)
                        st.session_state["ai_summary"] = (
                            f"Added {len(cast.characters)} characters, {len(cast.aliases)} alias "
                            f"links and {len(cast.places)} place names from {cast.model} "
                            f"({cast.excerpt_characters:,} characters sent)."
                        )
                        st.rerun()
                    except (LLMError, ValueError) as exc:
                        st.error(str(exc))
        if st.session_state.get("ai_summary"):
            st.success(st.session_state["ai_summary"])
            st.caption(
                "The fields below show the merged result. Review them, then press "
                "**Generate relationship network**."
            )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Alias table")
        preset_label = st.selectbox(
            "Start from a known book",
            ["None", *preset_labels()],
            help="Loads a small hand-written table for a common test book.",
        )
        if st.button("Load selected preset into the editor", width="stretch", disabled=preset_label == "None"):
            st.session_state["alias_text"] = aliases_to_json(get_preset_raw(preset_label))
            st.rerun()
        if "alias_text" not in st.session_state:
            st.session_state["alias_text"] = "{}"
        alias_text = st.text_area(
            "Variants → canonical name (JSON, or one 'variant,canonical' pair per line)",
            key="alias_text",
            height=260,
            help="Explicit mappings are applied before anything else. Keep them conservative.",
        )
        alias_upload = st.file_uploader("…or upload an alias table (.json / .csv / .txt)", type=["json", "csv", "txt"])
        auto_merge = st.toggle(
            "Merge unambiguous partial names automatically",
            value=True,
            help=(
                "Renames 'Darcy' to 'Fitzwilliam Darcy' only when exactly one longer name "
                "contains it. Ambiguous surnames such as 'Bennet' are never merged, and every "
                "merge is listed in the results."
            ),
        )
        st.caption("Automatic merging is a rule, not clustering: one candidate or nothing.")

    with right:
        st.subheader("Manual corrections")
        manual_raw = st.text_input(
            "Additional names",
            placeholder="Charlotte Lucas, Colonel Fitzwilliam",
            help="Comma-separated names to find even when spaCy misses them.",
            key="manual_names_input",
        )
        excluded_raw = st.text_input(
            "Names to ignore",
            placeholder="London, English",
            help="Comma-separated names that should not appear in the graph.",
            key="excluded_names_input",
        )
        st.markdown("#### Downloads for reuse")
        st.caption(
            "The alias table and the last result can be exported as files from the results "
            "section below."
        )
        with st.expander("Accepted alias formats"):
            st.markdown(
                """
                ```json
                {"Lizzy": "Elizabeth Bennet", "Mr. Darcy": "Fitzwilliam Darcy"}
                ```

                ```text
                Lizzy,Elizabeth Bennet
                Mr. Darcy,Fitzwilliam Darcy
                ```
                """
            )

with guide_tab:
    st.markdown(
        """
        ### How to use this tool

        1. **Add text.** Upload a UTF-8 `.txt` book, paste chapters directly, or press
           **Load novel example** for *Pride and Prejudice* chapters 1–3.
        2. **Check the reading scope.** If chapter headings are found, pick the chapters
           you are reading this week.
        3. **Review names and aliases.** Keep the alias table small and explicit. Use
           **Additional names** for characters spaCy missed, and **Names to ignore** for
           places or headings.
        4. **Adjust the settings** in the sidebar. Paragraph mode is the default; a sliding
           window connects characters who share a local passage.
        5. **Generate the network**, then hover over a node or an edge, read the tables, and
           download the HTML, PNG, or CSV files.

        ### What the numbers mean

        | Element | Meaning |
        | --- | --- |
        | Node size | How often the character is mentioned |
        | Node colour | Community (cluster of frequently co-occurring characters) or weighted degree |
        | Edge width | How often the pair appears in the same paragraph or window |
        | Betweenness | How often the character sits on the shortest path between others |

        ### When the app refuses

        Empty text, binary files, non-English text, text that is too short, and texts with
        no repeated character pairs are rejected with a message instead of a meaningless
        graph. That behaviour is deliberate and tested.

        ### Known limitations

        - Person-name recognition can miss minor characters or tag places as people.
        - One node can accidentally contain two people who share a surname and honorific.
        - Co-occurrence is not a relationship type: a strong edge says nothing about
          friendship, kinship, or enmity.
        """
    )

# --------------------------------------------------------------------------- run

st.divider()
run_column, note_column = st.columns([0.32, 0.68])
with run_column:
    generate = st.button("Generate relationship network", type="primary", width="stretch")
with note_column:
    if current_label:
        st.caption(f"Ready to analyse: **{current_label}** ({len(current_text):,} characters)")

if generate:
    state.update({"result": None, "figure": None, "positions": None, "error": None, "text_label": current_label})
    try:
        text_to_use = current_text
        if len(chapters) >= 2:
            text_to_use = slice_to_chapters(current_text, chapters, chapter_range[0], chapter_range[1])

        alias_source = alias_text
        if alias_upload is not None:
            alias_source = alias_upload.getvalue().decode("utf-8-sig", errors="replace")
        aliases = parse_alias_table(alias_source)

        with st.spinner("Detecting characters, counting co-occurrences, and drawing the network…"):
            nlp = language_model()
            result = analyze_text(
                text_to_use,
                nlp,
                aliases=aliases,
                manual_names=_split_names(manual_raw),
                excluded_names=_split_names(excluded_raw),
                mode=mode,
                window_size=window_size,
                minimum_edge_weight=minimum_edge_weight,
                minimum_mentions=minimum_mentions,
                maximum_characters=maximum_characters,
                auto_merge=auto_merge,
            )
            positions = display_layout(result.graph)
            figure = create_network_figure(
                result.graph,
                positions=positions,
                color_by=COLOR_CHOICES[color_label],
                show_labels=show_labels,
            )
        state.update(
            {
                "result": result,
                "figure": figure,
                "positions": positions,
                "error": None,
                "text_label": current_label,
            }
        )
    except (InputValidationError, ModelLoadError, ValueError) as exc:
        state.update({"result": None, "figure": None, "positions": None, "error": str(exc)})
    except Exception:  # pragma: no cover - last-resort guard for the demo
        state.update(
            {
                "result": None,
                "figure": None,
                "positions": None,
                "error": "The analysis could not be completed. Check the input and settings, then try again.",
            }
        )

# --------------------------------------------------------------------------- results

if state["error"]:
    st.error(state["error"])

result: AnalysisResult | None = state["result"]
if result is not None and state["figure"] is not None:
    graph = result.graph
    characters = node_table(graph)
    relationships = edge_table(graph)

    st.subheader("Relationship network")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Characters", graph.number_of_nodes())
    metric_columns[1].metric("Relationships", graph.number_of_edges())
    metric_columns[2].metric("Name mentions", len(result.mentions))
    metric_columns[3].metric("Text units", result.segment_count)
    metric_columns[4].metric("Analysis time", f"{result.elapsed_seconds:.1f} s")

    if result.chapters:
        st.caption(
            f"Chapters analysed: **{result.chapters[0].title}** → **{result.chapters[-1].title}** "
            f"({len(result.chapters)} detected in this selection)"
        )

    for warning in result.warnings:
        st.warning(warning)

    if result.settings.get("auto_merge") and result.merges:
        with st.expander(f"Automatic name merges applied ({len(result.merges)})"):
            st.dataframe(merge_table(result.merges), hide_index=True, width="stretch")
            st.caption("Turn the merge toggle off to compare the unmerged network.")

    st.plotly_chart(
        state["figure"],
        width="stretch",
        config={"scrollZoom": True, "displaylogo": False},
    )

    table_left, table_right = st.columns(2, gap="large")
    with table_left:
        st.markdown("#### Characters")
        st.dataframe(characters, hide_index=True, width="stretch")
    with table_right:
        st.markdown("#### Strongest co-occurrences")
        st.dataframe(relationships, hide_index=True, width="stretch")

    if result.chapters and len(result.chapters) >= 2:
        st.markdown("#### Character presence by chapter")
        names, labels, matrix = chapter_presence(result.chapters, result.mentions, character_limit=25)
        if names:
            st.plotly_chart(
                chapter_heatmap(names, labels, matrix),
                width="stretch",
                config={"displaylogo": False},
            )
            st.caption(
                "Read this like a cast list per chapter: a bright cell means the character is "
                "active there. It is a quick way to see when someone enters or leaves the story."
            )

    st.markdown("#### Download the result")
    download_columns = st.columns(5)
    html = figure_to_html(state["figure"])
    download_columns[0].download_button(
        "Graph (HTML)",
        data=html,
        file_name="character_network.html",
        mime="text/html",
        width="stretch",
    )
    try:
        png = render_png(
            graph,
            positions=state["positions"],
            color_by=COLOR_CHOICES[color_label],
        )
        download_columns[1].download_button(
            "Graph (PNG)",
            data=png,
            file_name="character_network.png",
            mime="image/png",
            width="stretch",
        )
    except Exception as exc:  # pragma: no cover - only when matplotlib is missing
        download_columns[1].caption(f"PNG unavailable: {exc}")

    download_columns[2].download_button(
        "Characters (CSV)",
        data=characters.to_csv(index=False).encode("utf-8"),
        file_name="characters.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[3].download_button(
        "Relationships (CSV)",
        data=relationships.to_csv(index=False).encode("utf-8"),
        file_name="relationships.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[4].download_button(
        "Summary (JSON)",
        data=json.dumps(_summary_payload(result), indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="analysis_summary.json",
        mime="application/json",
        width="stretch",
    )

    with st.expander("Recognised name list (for manual correction)"):
        st.caption(
            "Copy any name into **Names to ignore** in the *Names and aliases* tab, "
            "or add a missing character to **Additional names**."
        )
        recognised = pd.DataFrame(
            {
                "character": [name for name, _ in result.character_counts.most_common()],
                "mentions": [count for _, count in result.character_counts.most_common()],
            }
        )
        st.dataframe(recognised, hide_index=True, width="stretch", height=260)

    with st.expander("Interpretation and limitations"):
        st.markdown(
            """
            - Edge weight is the number of selected text units in which both names occur.
            - A central node appears with many others; it is not automatically the most
              important character.
            - Named-entity recognition can miss characters or mistake places for people.
            - Explicit aliases and automatic partial-name merges can be wrong; the merge
              audit trail above shows every automatic decision.
            - Nothing is uploaded: the text is processed in this local process only.
            """
        )

st.caption(
    "AILT9019 Group 65 · local processing · no text is sent to a remote service "
    "· edge weights describe co-occurrence only"
)
