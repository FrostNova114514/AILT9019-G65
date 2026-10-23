# Character Relationship Network Demo

This directory contains the working local web application described in the Group 65 revised proposal.

## Requirements

- Python 3.11–3.13 is recommended.
- Internet access is needed only during installation.
- No API key or cloud account is required.

## Installation

From this `Demo` directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `python3.12` is not installed, use a supported Python 3.11–3.13 executable available on the computer.

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`, and open it in the default browser.
The included configuration binds the server to `localhost`, so uploaded text is not exposed as a service on the local network.

## Test

```bash
source .venv/bin/activate
pytest -q
```

## Demo flow

1. Select **Load built-in sample**, or upload/paste another English text.
2. Review the alias table and optional manual name corrections.
3. Choose paragraph or sliding-window analysis.
4. Select **Generate relationship network**.
5. Inspect the graph and tables, then export HTML or CSV files.

## Interpretation

An edge means that two detected names occurred in the same paragraph or token window. Its weight is the number of such co-occurrences. It does not identify the semantic nature of a relationship.

The application processes input locally and does not intentionally upload or retain the submitted text.
