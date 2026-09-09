# CLAUDE.md — B-NER

Last updated: 2026-09-09

Implementation and dataset code for the paper *"B-NER: A Novel Bangla Named
Entity Recognition Dataset with Largest Entities and Its Baseline Evaluation"*.
The repo is a notebook-based research project; there is no application code,
package, or test suite.

## Project structure

```
data/                        DVC pointers (.dvc) for the B-NER corpus — run `dvc pull`
model/                       DVC pointers for the fine-tuned NER model + tokenizer
eda_cleaning.ipynb           Corpus cleaning and exploratory analysis
dataset_tokens_prep.ipynb    Tokenisation / dataset assembly for training
dataset_PCA_overlapping.ipynb  PCA + KMeans analysis of entity overlap (reads .xlsx)
train_ner.ipynb              Fine-tunes BertForTokenClassification, seqeval metrics
ner_demo.ipynb               Loads the trained model and runs inference
requirements.txt             Direct runtime dependencies (see convention below)
env.yml                      LEGACY Python 3.8 conda export from 2021 — unmaintained
```

## Setup

- Python **3.10+** (required by transformers 5.x / torch 2.x).
- `dvc pull` to fetch data and model artifacts.
- `pip install -r requirements.txt`
- `pip install jupyterlab` — deliberately **not** in requirements.txt.

## Dependency convention (important)

`requirements.txt` lists **only packages the notebooks directly import**, with
`>=` lower bounds set to the oldest version carrying no known advisory.

Do **not** regenerate it with `pip freeze`. The file was originally a full
freeze of a 2021 conda environment: it pinned ~120 packages, most of them
transitive (aiohttp, urllib3, pyarrow) or local Jupyter tooling (tornado,
mistune, jupyter-server, nbconvert), plus `nltk` and `spacy` that no notebook
imports. Every one of those entered GitHub's dependency graph and generated
Dependabot alerts — 235 open alerts by 2026-09.

The invariant: **a package belongs in `requirements.txt` only if a notebook
imports it.** Transitive dependencies resolve themselves and stay patched;
dev/runtime tooling is installed separately.

## API notes

These notebooks were written against transformers 4.15 / datasets 1.17. Two
APIs changed with the supported versions and are already updated in-tree:

- `datasets.load_metric` was removed in `datasets>=3.0` → use
  `evaluate.load("seqeval")` (hence `evaluate` in requirements.txt).
- `TrainingArguments(evaluation_strategy=...)` was renamed to `eval_strategy`
  in `transformers>=4.46` and removed in 5.x.

Stored cell *outputs* still show the old calls — those are historical execution
records from 2021 and are intentionally left untouched.
