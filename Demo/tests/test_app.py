from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not app.exception
    return app


def test_app_starts_without_errors_and_shows_instructions():
    app = _app()
    assert not app.exception
    text = " ".join(str(block.value) for block in app.markdown)
    assert "How to use" in text
    assert "does not establish" in text


def test_sample_runs_end_to_end_and_exports_result():
    app = _app()
    next(button for button in app.button if button.label == "Load short sample").click().run()
    assert not app.exception
    assert len(app.text_area[0].value) > 100

    next(button for button in app.button if button.label == "Generate relationship network").click().run()

    assert not app.exception
    assert not app.error
    labels = {metric.label for metric in app.metric}
    assert labels == {"Characters", "Relationships", "Name mentions", "Text units", "Analysis time"}
    assert len(app.dataframe) >= 2
    download_labels = {button.label for button in app.get("download_button")}
    assert "Graph (HTML)" in download_labels
    assert "Graph (PNG)" in download_labels
    assert "Characters (CSV)" in download_labels
    assert "Summary (JSON)" in download_labels


def test_empty_input_reports_a_clear_error():
    app = _app()
    generate = next(
        button for button in app.button if button.label == "Generate relationship network"
    )
    generate.click().run()
    assert not app.exception
    assert app.error
    assert "non-empty" in app.error[0].value.lower()
    assert not app.metric


def test_theory_style_text_is_refused_instead_of_drawn():
    app = _app()
    app.text_area[0].set_value(
        "Information theory studies the quantification, storage, and communication of "
        "digital information. The field was established by Claude Shannon in 1948, who "
        "introduced entropy as a measure of uncertainty in a message source. Coding "
        "theory asks how close a code can come to the Shannon limit while remaining "
        "decodable in linear time. Rate-distortion theory extends the framework to lossy "
        "compression, where the question becomes how much fidelity is exchanged for how "
        "many bits. Modern work on network information theory considers several senders "
        "and receivers at once, and the capacity region of such a network is unknown."
    ).run()
    next(button for button in app.button if button.label == "Generate relationship network").click().run()
    assert not app.exception
    assert app.error
    assert "Not enough character relationships" in app.error[0].value
    assert not app.metric


def test_example_book_button_loads_chapters_and_alias_preset():
    app = _app()
    next(button for button in app.button if button.label == "Load novel example").click().run()
    assert not app.exception
    pasted = app.text_area[0].value
    assert "Chapter I" in pasted
    assert "Lizzy" in app.text_area[1].value  # alias preset loaded into the editor
    assert any("61 chapters detected" in str(caption.value) or "chapters detected" in str(caption.value)
               for caption in app.caption)
