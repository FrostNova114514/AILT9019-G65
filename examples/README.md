# Example input and expected output

This folder is the checkable evidence that the core path runs: one real input file and
the files the pipeline produces from it.

## Input

`Demo/sample_data/pride_and_prejudice_chapters_1_3.txt` — *Pride and Prejudice*,
chapters I–III (18,965 characters, about 3,400 words), copied from the public-domain
Project Gutenberg text (ebook 1342) with the illustration captions removed.

## Regenerate the output

```bash
cd Demo
python cli.py \
  --input sample_data/pride_and_prejudice_chapters_1_3.txt \
  --preset "Pride and Prejudice" \
  --exclude "Netherfield, Longbourn, Meryton" \
  --outdir ../examples/expected_output/pride_chapters_1_3
```

The same result is produced in the web app: **Load novel example** → **Generate
relationship network** (the example button fills in the alias table and the ignore list).

## Expected output

`pride_chapters_1_3/` contains:

| File | What it is |
| --- | --- |
| `network.html` | Standalone interactive graph (Plotly inlined, opens offline) |
| `network.png` | Static 200 dpi image with the same layout |
| `characters.csv` | Character, mentions, connections, weighted degree, betweenness, community |
| `relationships.csv` | Edge list with co-occurrence counts and one example quotation per pair |
| `summary.json` | Settings, metrics, chapters, merges, warnings, full tables |
| `aliases.json` | The alias table that was applied |

Printed summary for this input:

```text
Characters: 13   Relationships: 34   Mentions: 101   Units: 85
Analysed in 0.4 s

  #  character                        mentions  weighted degree
  1  Charles Bingley                        23               26
  2  Mr Bennet                              16               12
  3  Elizabeth Bennet                       12               17
  4  Mrs Bennet                              8               10
  5  Mrs Long                                6                1
  6  Jane Bennet                             6                9
  7  Fitzwilliam Darcy                       6               10
  8  Mary Bennet                             4                5
  9  Lady Lucas                              3                6
 10  Kitty Bennet                            3                2
 11  Sir William Lucas                       2                4
 12  Mrs Hurst                               2                5
 13  Caroline Bingley                        2                9
```

Bingley leads on mentions in these three chapters because chapter II is almost all
talk *about* him; Elizabeth overtakes him across the whole novel (see
`evaluation/REPORT.md`).

## Full-book output

`pride_and_prejudice_network.*` and `jane_eyre_network.*` are the same two files
regenerated for both complete novels by:

```bash
python evaluation/run_evaluation.py --export-examples
```

The HTML files inline Plotly (about 5 MB each) so they open without internet access.
