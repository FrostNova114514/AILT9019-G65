from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_sample_generates_complete_result():
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
    assert not app.exception

    load_sample = next(
        button for button in app.button if button.label == "Load built-in sample"
    )
    load_sample.click().run()
    assert len(app.text_area[0].value) > 100

    generate = next(
        button
        for button in app.button
        if button.label == "Generate relationship network"
    )
    generate.click().run()

    assert not app.exception
    assert not app.error
    assert {metric.label for metric in app.metric} == {
        "Characters",
        "Relationships",
        "Name mentions",
        "Text units",
    }
    assert len(app.dataframe) == 2
    assert len(app.get("download_button")) == 3
