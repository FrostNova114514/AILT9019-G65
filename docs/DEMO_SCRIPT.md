# 60-second recording plan (mid-term check)

The mid-term check asks for one real input, the system processing it, and the result in
60 seconds or less, with narration or captions. This is a shot list that fits.

## Before recording

```bash
cd Demo
source .venv/bin/activate
streamlit run app.py
```

- Open `http://localhost:8501` and close other tabs, notifications, and any window that
  shows personal files.
- Do **not** show API keys, e-mail, or the OneDrive path bar.
- The first analysis loads the spaCy model (about 1–2 s). Do a warm-up run once and keep
  the browser tab open: the model is cached per process (`st.cache_resource`), so the
  recorded run starts immediately.
- Window size 1280 × 720 or larger, browser zoom 100%.

## Shot list

| Time | Action | Caption / narration |
| --- | --- | --- |
| 0:00–0:06 | Show the app start page, sidebar visible | "A local tool that turns an English novel into one character network." |
| 0:06–0:14 | Click **Load novel example** in *1 · Add text* | "Pride and Prejudice, chapters one to three, from the bundled example." |
| 0:14–0:20 | Point at the reading scope slider and the *Names and aliases* tab | "Chapter headings are detected, and the alias table maps Lizzy to Elizabeth Bennet." |
| 0:20–0:24 | Click **Generate relationship network** | "One click: person names, co-occurrence counts, graph." |
| 0:24–0:40 | Hover one node, then one thick edge; show the quotation in the tooltip | "Node size is mentions, edge width is co-occurrence, and every edge keeps one sentence you can check." |
| 0:40–0:50 | Scroll to the tables, then the chapter heatmap | "Characters and relationships as tables, and a per-chapter presence map." |
| 0:50–0:56 | Hover the download buttons, then show the PNG open | "Exports as a standalone HTML page, a PNG, CSV tables, and JSON." |
| 0:56–1:00 | Cut to `evaluation/REPORT.md` table | "On two full novels: 80% and 70% recall for the top ten characters, 100% and 80% of the expected relationships." |

## Exact numbers to quote

From a fresh run of the bundled example (13 characters, 34 relationships): Charles
Bingley 23 mentions, Mr Bennet 16, Elizabeth Bennet 12, Mrs Bennet 8, Fitzwilliam Darcy 6.

From the full-book evaluation: *Pride and Prejudice* recall@10 80%, edge coverage 100%;
*Jane Eyre* recall@10 70%, edge coverage 80%.

## If something goes wrong on camera

- Empty graph → raise *Minimum mentions per character* back to 2 and re-generate; the
  ignore list may have removed a real character.
- A main character is missing → open **AI cast list**, press **Find characters with AI**
  (with `DEEPSEEK_API_KEY` set), review the filled fields, and generate again. Without a
  network, skip this: the local pipeline still works.
- Model error → the banner says to run `python -m spacy download en_core_web_sm`.
- Anything else → stop, restart `streamlit run app.py`, and record again from 0:00.

## Optional extra shot (5 s, if there is room)

Open *Names and aliases*, point at the **AI cast list** expander and say: "If a character
is missing, the model proposes a cast list, and every name is still verified against the
text." Never show the API key on screen.

## Cheat line

If narration is hard to time, use only these four captions, one per shot:

1. "Upload or paste an English book — locally, no API."
2. "Alias table and chapter scope keep the result honest."
3. "Co-occurrence graph with one checkable sentence per edge."
4. "HTML, PNG, CSV — and a reproducible evaluation on two novels."
