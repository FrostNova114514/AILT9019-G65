# Evaluation

Two English novels, a hand-written reference list per novel, and one script that
reproduces every number in [REPORT.md](REPORT.md).

## What is measured

| Metric | Definition |
| --- | --- |
| Recall@5 / @10 / @20 | Share of the reference top-N characters that also appear in the detected top-N |
| Character recall | Share of all reference characters that are detected at all |
| Precision@10 | Share of the detected top ten that are on the reference list |
| Edge coverage | Share of the reference relationship pairs that exist as graph edges |
| Protagonist rank | Position of the reference protagonist by weighted degree |
| Failure cases | Six unusable inputs that must be refused with a clear message |

Matching is one-to-one and ignores honorifics, so a bare `Bennet` can be credited for
`Mr Bennet` exactly once and never twice.

## Two settings

- **automatic** — the alias preset for the book plus the built-in rules. This is what a
  user gets after choosing a preset and pressing Generate.
- **assisted** — the same run plus the *Additional names* list and the *Names to ignore*
  list from the reference file. This is what the fields in the *Names and aliases* tab
  are for.

Comparing the two shows what the manual correction step changes: in *Jane Eyre* it moves
two more main characters into the top five and removes the single false positive from the
top ten. It does not change recall@10, because that limit comes from ranking order rather
than from a missed name.

## Data

| File | Source | Size |
| --- | --- | --- |
| `data/pride_and_prejudice.txt` | Project Gutenberg ebook 1342 | 680 KB, 61 chapters |
| `data/jane_eyre.txt` | Project Gutenberg ebook 1260 | 1.0 MB, 37 chapters |
| `ground_truth/pride_and_prejudice.json` | Hand-written reference list | 20 characters, 20 pairs |
| `ground_truth/jane_eyre.json` | Hand-written reference list | 20 characters, 20 pairs |

Both novels are in the public domain. `prepare_data.py` removes the Gutenberg licence
header and footer, the editor's introduction, and the 1894 plate captions (which inject
a publisher name into the text). The prose itself is untouched.

If the `*_raw.txt` downloads are missing, fetch them again:

```bash
curl -L -o evaluation/data/pride_and_prejudice_raw.txt https://www.gutenberg.org/files/1342/1342-0.txt
curl -L -o evaluation/data/jane_eyre_raw.txt          https://www.gutenberg.org/files/1260/1260-0.txt
python evaluation/prepare_data.py
```

## Run it

```bash
python evaluation/run_evaluation.py                 # both books, both settings
python evaluation/run_evaluation.py --book pride    # one book
python evaluation/run_evaluation.py --export-examples
```

The script writes `REPORT.md` and `results.json` next to itself and needs no network
access once the model is installed.

## Current result (short version)

| Book | Setting | Recall@10 | Edge coverage |
| --- | --- | --- | --- |
| Pride and Prejudice | automatic | 80% | 100% |
| Jane Eyre | automatic | 70% | 80% |
| Both | failure cases | — | 6/6 refused |

The proposal's target was about 70% recall for the top ten characters; both books meet
it without any manual name entry. Full tables, the detected lists, and the missed pairs
are in [REPORT.md](REPORT.md).
