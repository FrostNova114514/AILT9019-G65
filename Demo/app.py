"""Streamlit interface for the Character Relationship Network demo."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import spacy
import streamlit as st

from character_network.analysis import (
    InputValidationError,
    analyze_text,
    parse_alias_json,
)
from character_network.visualization import (
    create_network_figure,
    edge_table,
    node_table,
)


APP_DIRECTORY = Path(__file__).resolve().parent
SAMPLE_PATH = APP_DIRECTORY / "sample_data" / "sample_story.txt"
DEFAULT_ALIASES = {
    "Lizzy": "Elizabeth Bennet",
    "Elizabeth": "Elizabeth Bennet",
    "Miss Bennet": "Elizabeth Bennet",
    "Mrs. Bennet": "Mrs Bennet",
    "Jane": "Jane Bennet",
    "Miss Jane Bennet": "Jane Bennet",
    "Bingley": "Charles Bingley",
    "Mr. Bingley": "Charles Bingley",
    "Mr. Darcy": "Fitzwilliam Darcy",
    "Darcy": "Fitzwilliam Darcy",
}


st.set_page_config(
    page_title="Character Relationship Network",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #f7f8fc; }
      .block-container { max-width: 1320px; padding-top: 2.2rem; }
      .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(125deg, #17223b 0%, #334e8a 55%, #6941c6 100%);
        box-shadow: 0 18px 45px rgba(23, 34, 59, 0.16);
        margin-bottom: 1.25rem;
      }
      .hero h1 { margin: 0 0 .35rem 0; font-size: 2.25rem; }
      .hero p { margin: 0; max-width: 850px; opacity: .9; font-size: 1.03rem; }
      .notice {
        border-left: 4px solid #f0a43c;
        background: #fff8e9;
        color: #5f4318;
        border-radius: 8px;
        padding: .8rem 1rem;
        margin-bottom: 1rem;
      }
      div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e6e9f2;
        padding: .8rem 1rem;
        border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <h1>Character Relationship Network</h1>
      <p>Turn English long-form text into an interactive map of character co-occurrences. Everything runs locally in this demo.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="notice"><strong>Interpret carefully:</strong> an edge means that two names occur near each other in the text. It does not establish whether they are friends, relatives, enemies, or romantic partners.</div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the English language model…")
def load_language_model():
    """Load the small English spaCy pipeline once per Streamlit process."""

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "The spaCy English model is missing. Run: python -m spacy download en_core_web_sm"
        ) from exc


def decode_upload(uploaded_file) -> str:
    """Decode a Streamlit upload as UTF-8 text with a useful error."""

    if uploaded_file is None:
        return ""
    if not uploaded_file.name.lower().endswith(".txt"):
        raise InputValidationError("Only plain-text (.txt) files are supported.")
    try:
        return uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputValidationError(
            "The uploaded file is not valid UTF-8 text. Save it as UTF-8 and try again."
        ) from exc


def comma_separated_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""

with st.sidebar:
    st.header("Analysis settings")
    mode_label = st.radio(
        "Co-occurrence unit",
        ["Paragraphs", "Sliding token window"],
        help="Characters are connected when they appear in the same selected unit.",
    )
    mode = "paragraph" if mode_label == "Paragraphs" else "window"
    window_size = st.slider(
        "Window size (tokens)",
        min_value=40,
        max_value=300,
        value=120,
        step=10,
        disabled=mode != "window",
    )
    minimum_edge_weight = st.slider(
        "Minimum co-occurrence count", 1, 10, 1
    )
    maximum_characters = st.slider(
        "Maximum characters shown", 5, 60, 30, step=5
    )
    st.caption(
        "Higher thresholds produce a cleaner graph but can remove less frequent characters."
    )

input_tab, options_tab, guide_tab = st.tabs(
    ["1 · Add text", "2 · Names and aliases", "How to use"]
)

with input_tab:
    upload_column, paste_column = st.columns([0.85, 1.15], gap="large")
    with upload_column:
        st.subheader("Upload a book")
        uploaded = st.file_uploader("English plain text", type=["txt"])
        st.caption("UTF-8 `.txt` files only · Maximum 2,000,000 characters")
        if st.button("Load built-in sample", width="stretch"):
            st.session_state.sample_text = SAMPLE_PATH.read_text(encoding="utf-8")
            st.rerun()
    with paste_column:
        st.subheader("Or paste text")
        pasted = st.text_area(
            "English text",
            key="sample_text",
            height=230,
            placeholder="Paste several paragraphs from an English text here…",
            label_visibility="collapsed",
        )

with options_tab:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Alias table")
        alias_raw = st.text_area(
            "JSON mapping from a name variant to one canonical name",
            value=json.dumps(DEFAULT_ALIASES, indent=2),
            height=240,
        )
        st.caption(
            "Aliases are deliberately explicit to avoid incorrectly merging different characters."
        )
    with right:
        st.subheader("Manual corrections")
        manual_raw = st.text_input(
            "Additional names",
            placeholder="Charlotte Lucas, Colonel Fitzwilliam",
            help="Comma-separated exact names to find even if spaCy misses them.",
        )
        excluded_raw = st.text_input(
            "Names to exclude",
            placeholder="London, English",
            help="Comma-separated detected names that should not appear in the graph.",
        )

with guide_tab:
    st.markdown(
        """
        1. Upload a UTF-8 English `.txt` file or paste several paragraphs.
        2. Optionally edit the alias table or add/remove names.
        3. Choose paragraph or sliding-window analysis in the sidebar.
        4. Select **Generate relationship network**.
        5. Pan, zoom, and hover over the graph, then download the results if needed.

        For best results, use narrative prose with recurring named characters. Empty input,
        non-English text, and texts without enough character relationships are rejected safely.
        """
    )

st.divider()
generate = st.button(
    "Generate relationship network",
    type="primary",
    width="stretch",
)

if generate:
    try:
        uploaded_text = decode_upload(uploaded)
        text = uploaded_text if uploaded_text.strip() else pasted
        aliases = parse_alias_json(alias_raw)
        nlp = load_language_model()
        with st.spinner("Detecting characters and building the network…"):
            result = analyze_text(
                text,
                nlp,
                aliases=aliases,
                manual_names=comma_separated_names(manual_raw),
                excluded_names=comma_separated_names(excluded_raw),
                mode=mode,
                window_size=window_size,
                minimum_edge_weight=minimum_edge_weight,
                maximum_characters=maximum_characters,
            )
            st.session_state.analysis_result = result
    except (InputValidationError, RuntimeError) as exc:
        st.session_state.pop("analysis_result", None)
        st.error(str(exc))
    except Exception:
        st.session_state.pop("analysis_result", None)
        st.error(
            "The analysis could not be completed. Check the input and settings, then try again."
        )

result = st.session_state.get("analysis_result")
if result is not None:
    graph = result.graph
    figure = create_network_figure(graph)
    characters = node_table(graph)
    relationships = edge_table(graph)

    st.subheader("Relationship network")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Characters", graph.number_of_nodes())
    metric_columns[1].metric("Relationships", graph.number_of_edges())
    metric_columns[2].metric("Name mentions", len(result.mentions))
    metric_columns[3].metric("Text units", result.segment_count)

    st.plotly_chart(
        figure,
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

    st.markdown("#### Download results")
    html = figure.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Interactive graph (HTML)",
        data=html,
        file_name="character_relationship_network.html",
        mime="text/html",
        width="stretch",
    )
    download_columns[1].download_button(
        "Character table (CSV)",
        data=characters.to_csv(index=False).encode("utf-8"),
        file_name="characters.csv",
        mime="text/csv",
        width="stretch",
    )
    download_columns[2].download_button(
        "Relationship table (CSV)",
        data=relationships.to_csv(index=False).encode("utf-8"),
        file_name="relationships.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Interpretation and limitations"):
        st.markdown(
            """
            - Edge weight is a count of co-occurrence within the selected text unit.
            - A central node is frequently connected in this text; it is not automatically the most important character.
            - Named-entity recognition can miss characters or mistake other proper nouns for people.
            - Explicit alias mappings can improve results, but an incorrect mapping can merge different characters.
            """
        )

st.caption(
    "AILT9019 Group 65 · Local processing · No user text is intentionally stored or uploaded"
)
