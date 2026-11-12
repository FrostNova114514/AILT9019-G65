# Character Relationship Network — application

The working local web app for Group 65. It turns an English book into a weighted
co-occurrence network of its characters, with chapter handling, a static PNG export, and
a headless CLI for reproducible runs.

## Requirements

- Python 3.11–3.13 (tested with 3.13.5)
- About 400 MB of disk after installation (spaCy model plus Plotly)
- Internet only during installation; the app itself never sends text anywhere

## Install

From this `Demo` directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Streamlit prints a local address (normally `http://localhost:8501`) and opens it. The
included `.streamlit/config.toml` binds the server to `localhost`, so nothing is exposed
on the local network.

## Run without a browser

`cli.py` runs the same pipeline and writes the same artifacts:

```bash
python cli.py --input sample_data/pride_and_prejudice_chapters_1_3.txt \
    --preset "Pride and Prejudice" \
    --exclude "Netherfield, Longbourn, Meryton" \
    --outdir output
```

| Option | Meaning |
| --- | --- |
| `--input` | UTF-8 plain-text file to analyse |
| `--preset` | Built-in alias table (`Pride and Prejudice`, `Jane Eyre`) |
| `--alias` | Your own alias table (`.json` or `.csv`) |
| `--chapters 1-6` | Analyse a chapter range only |
| `--mode window --window 200` | Co-occurrence by sliding token window |
| `--min-weight`, `--min-mentions`, `--max-characters` | Graph thresholds; `--max-characters all` keeps everyone |
| `--manual-names`, `--exclude` | Comma-separated corrections |
| `--no-auto-merge`, `--no-png` | Turn off the automatic partial-name merge or the PNG |
| `--llm-cast` | Ask DeepSeek for a cast list first (needs `--llm-key` or `DEEPSEEK_API_KEY`) |

Outputs: `network.html`, `network.png`, `characters.csv`, `relationships.csv`,
`summary.json`, `aliases.json`.

## Optional AI cast list

spaCy's small model misses minor characters and sometimes tags a person as a place. The
app can ask DeepSeek for a **cast list only** — canonical names, aliases, and places to
ignore — and then run the normal local pipeline on it.

**Set the key once, in a file.** Copy `Demo/.streamlit/secrets.example.toml` to
`Demo/.streamlit/secrets.toml` and fill in your key:

```toml
DEEPSEEK_API_KEY = "sk-…"
```

That file is gitignored, and both the app and `cli.py` read it automatically, so no
command ever contains the key. Two alternatives work too: the `DEEPSEEK_API_KEY`
environment variable (takes priority), or pasting the key into the app's password field
for one session.

```bash
streamlit run app.py
python cli.py --input book.txt --preset "Pride and Prejudice" --llm-cast --outdir output
```

In the app: *Names and aliases* → expand **AI cast list** → **Find characters with AI**.
The alias table, the additional-names field, and the ignore list are filled in, and you
can edit them before pressing Generate.

What leaves the machine: about 24,000 characters of evenly spaced excerpts (roughly 7k
tokens, under one US cent) and nothing else.

Why the list is safe to trust: the model only proposes names. Every name is matched
against the local text, so a name that is not in the book produces no mentions and
cannot appear in the graph. Measured effect on the evaluation: recall@10 rises from 80%
to 90% (*Pride and Prejudice*) and from 70% to 80% (*Jane Eyre*).

If the API is unreachable, the app shows the error and keeps the local result; remove
`--llm-cast` to stay offline.

## Test

```bash
python -m pytest -q
```

110 test cases (107 test functions) cover input decoding and validation, name normalization and merging,
segmentation, graph construction, the non-overlapping layout, the plotting functions,
the evaluation metrics, the optional cast-list parser, the CLI, and the Streamlit UI
(through `streamlit.testing`).

## Module map

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface: input, names and aliases, results, downloads |
| `cli.py` | Headless runner writing HTML/PNG/CSV/JSON |
| `character_network/textproc.py` | Decoding, validation, language check, paragraphs, chapters, windows |
| `character_network/names.py` | Normalization, alias table parsing, title rules, conservative merges |
| `character_network/extraction.py` | spaCy PERSON detection, honorific re-attachment, filters |
| `character_network/graph.py` | Co-occurrence counting, graph construction, metrics, layout |
| `character_network/visualization.py` | Canvas layout without overlaps, Plotly figure, matplotlib PNG, chapter heatmap, tables |
| `character_network/analysis.py` | Pipeline orchestration, `analyze_text`, `AnalysisResult` |
| `character_network/evaluation.py` | Recall@k, edge coverage, name matching |
| `character_network/llm.py` | Optional DeepSeek cast list (excerpt sampling, JSON parsing, merge rules) |
| `character_network/presets.py` | Alias tables for the two test novels |

## Troubleshooting

**`OSError: [E050] Can't find model 'en_core_web_sm'`**
The wheel install failed or pip skipped it. Run
`python -m spacy download en_core_web_sm`, or reinstall with
`python -m pip install --force-reinstall -r requirements.txt`.

**`streamlit: command not found`**
The virtual environment is not active. Activate it (`.venv\Scripts\activate` on Windows)
and run `python -m streamlit run app.py`.

**The graph is unreadable**
Raise *Minimum co-occurrence count*, raise *Minimum mentions per character*, or lower
*Characters shown*. `--max-characters all` draws everyone, which is rarely readable for a
whole novel. Node circles never overlap: the layout reserves space for each marker and
pushes neighbours apart, and labels move to the side that stays clear.

**The AI cast list fails**
`The API key was rejected` means the key is wrong or revoked — generate a new one.
`Could not reach the API` means no network; turn the toggle off, the local pipeline does
not need it. Rate limits reset after a moment.

**Two characters share one node**
Honorifics are stripped before counting, so a bare surname (`Bennet`, `Reed`) can absorb
an unnamed relative. Add an explicit alias such as `{"Bennet": "Mr Bennet"}` and re-run.
Conversely, if one person is split, use the *Additional names* field.

**The app refuses my text**
Supported input is UTF-8 (also UTF-16 and Windows-1252) plain text between 40 and
2,000,000 characters, in English, with recurring named people. PDF, Word, and binary
files are rejected on purpose.
