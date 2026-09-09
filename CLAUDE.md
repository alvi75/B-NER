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
imports it *or needs it at runtime*.** Transitive dependencies resolve
themselves and stay patched; dev/runtime tooling is installed separately.

The "or needs it at runtime" half is not optional — three entries here are
never `import`ed by any notebook and all three are required:

| Package | Why it is needed, despite no import |
| --- | --- |
| `accelerate` (via `transformers[torch]`) | `TrainingArguments.__post_init__` raises `ImportError` without it on transformers 5.x. Not a base dep of the transformers wheel — only under `extra == "torch"`. |
| `openpyxl` | engine for `pd.read_excel` on `.xlsx` in `dataset_PCA_overlapping.ipynb` |
| `seqeval` | `evaluate.load("seqeval")` imports it inside `compute()` |

An audit that greps `import` lines will miss all three. Check keyword arguments
and input types too.

## API notes

These notebooks were written against transformers 4.15 / datasets 1.17 /
pandas 1.3 / sklearn 1.0 (2021). Six APIs broke on the supported versions. All
six are fixed in-tree; each was reproduced as a real exception before fixing:

| Where | Was | Now | Because |
| --- | --- | --- | --- |
| `train_ner.ipynb` c0 | `datasets.load_metric` | `evaluate.load("seqeval")` | removed in `datasets>=3.0` |
| `train_ner.ipynb` c28 | `evaluation_strategy=` | `eval_strategy=` | renamed in `transformers>=4.46` |
| `train_ner.ipynb` c29 | `Trainer(tokenizer=)` | `processing_class=` | removed in `transformers` 5.x — no `**kwargs` to absorb it |
| `train_ner.ipynb` c8/11/12 | `load_dataset(use_auth_token=)` | kwarg dropped | removed in `datasets>=3.0`; use `token=` if auth is needed |
| `eda_cleaning.ipynb` c1 | `.fillna(method="ffill")` | `.ffill()` | `method=` removed in pandas 3.x |
| `dataset_PCA_overlapping.ipynb` c4/c6 | `.todense()` | `.toarray()` | `.todense()` returns `np.matrix`, which sklearn>=1.2 rejects with `TypeError` |

### Reproducibility warnings

Two defaults changed *silently* — they do not error, they return different
numbers than the paper:

- **`KMeans` `n_init` went 10 → 1** in sklearn 1.4. Now pinned explicitly as
  `KMeans(n_clusters=2, n_init=10, random_state=42)` in
  `dataset_PCA_overlapping.ipynb` to restore the old behaviour *and* make it
  deterministic (the original passed no `random_state`).
- **The `Trainer` default optimizer changed** from HF's own `AdamW` (4.15) to
  `adamw_torch_fused` (5.x). Same seed and data will produce different weights
  and a different F1. Pass `optim="adamw_torch"` if the paper's number must
  reproduce. **Not** changed in-tree — it is a research decision, not a bug.

### Known broken, not fixable here

`train_ner.ipynb` c27 loads `neuralspace-reverie/indic-transformers-bn-bert`.
That checkpoint now returns **HTTP 401** from the Hub and appears withdrawn
(`?author=neuralspace-reverie` returns `[]`; a control model returns 200).
Training cannot run until it is mirrored or substituted. Unrelated to the
dependency work.

Stored cell *outputs* still show the old calls — those are historical execution
records from 2021 and are intentionally left untouched.

## Secrets

`train_ner.ipynb` previously hardcoded a Hugging Face token in
`load_dataset(..., use_auth_token='...')` across cells 8, 11 and 12. It was
committed in `edf7962` and public. The kwarg is now removed, but **removal does
not revoke it** — it remains in git history and must be revoked at
huggingface.co/settings/tokens. Never commit a token; use the `HF_TOKEN`
environment variable, which `huggingface_hub` reads automatically.
