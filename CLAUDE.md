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

- Python **3.12+**. transformers 5.x / torch 2.x only need 3.10, but numpy 2.5
  needs 3.12 and pandas 3.0 / scikit-learn 1.9 / matplotlib 3.11 need 3.11 — on
  3.10 the `>=` bounds silently resolve to an older, untested stack.
- `dvc pull` to fetch data and model artifacts.
- `pip install -r requirements.txt`
- `pip install jupyterlab` — deliberately **not** in requirements.txt.

## Dependency convention (important)

`requirements.txt` lists **only what the project needs at runtime**. Where a
`>=` bound is given it is the oldest version carrying no known advisory;
unbounded entries have no advisories on record.

Do **not** regenerate it with `pip freeze`. The file was originally a full
freeze of a 2021 conda environment: it pinned 127 packages, most of them
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
| `accelerate` (via `transformers[torch]`) | raises `ImportError` without it on transformers 5.x (raised in `_setup_devices`, reached from `TrainingArguments.__post_init__` via `self.device`). Not a base dep of the transformers wheel — only under `extra == "torch"`. |
| `openpyxl` | engine for `pd.read_excel` on `.xlsx` in `dataset_PCA_overlapping.ipynb` |
| `seqeval` | the evaluate seqeval module does `from seqeval.metrics import ...` at module level, so it is needed at `evaluate.load()` time |

An audit that greps `import` lines will miss all three. Check keyword arguments
and input types too.

## API notes

These notebooks were written against transformers 4.15 / datasets 1.17 /
pandas 1.3 / sklearn 1.0 (2021). Six APIs broke on the supported versions. All
six are fixed in-tree; each was reproduced as a real exception before fixing:

| Where | Was | Now | Because |
| --- | --- | --- | --- |
| `train_ner.ipynb` c0 | `datasets.load_metric` | `evaluate.load("seqeval")` | removed in `datasets>=3.0` |
| `train_ner.ipynb` c28 | `evaluation_strategy=` | `eval_strategy=` | `eval_strategy` added in 4.41; `evaluation_strategy` kept as a deprecated alias, removed in 5.0 |
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

### Pre-existing bugs (NOT caused by the dependency migration)

Verified identical in the original 2023 commits and at HEAD — these have never
worked in this repo layout. Left unfixed deliberately: each depends on how you
actually run the notebooks, which is your call, not a migration decision.

1. **Every data/model path is one level too high.** All five notebooks reference
   `./../data/...` and `./../model/...` — **33 references** — but `data/` and
   `model/` sit at the repo root *beside* the notebooks, so `./../` resolves
   outside the repo. Reproduced: `OSError: Repo id must be in the form
   'repo_name' or 'namespace/repo_name': './../model/ner_tokenizer'`. Fix is to
   drop one `../` throughout, if you intend notebooks to run from the repo root.
   Combined with the empty `.dvc/config`, no notebook can currently reach real
   data.
2. **`np.split` on a DataFrame returns ndarrays.** `dataset_tokens_prep.ipynb`
   c13 does `np.split(data.sample(...), [...])`, then c15 calls
   `.reset_index(drop=True)` on the results →
   `AttributeError: 'numpy.ndarray' object has no attribute 'reset_index'`.
   pandas does not implement `__array_function__`, so this never worked as
   written. Use `.iloc[]` slices instead.
3. **`ner_demo.ipynb` hardcodes `.to("cuda")`** (c3, c5) →
   `AssertionError: Torch not compiled with CUDA enabled` on any CPU-only host.
   Guard with `"cuda" if torch.cuda.is_available() else "cpu"`.

Also informational: `BertTokenizerFast` and `DistilBertTokenizerFast` are now
bare aliases of the slow classes (`tokenization_bert.py:140`,
`BertTokenizerFast = BertTokenizer`) — fast/slow were unified in transformers
5.x. `is_fast` is still `True`, so behaviour is unaffected; only `isinstance`
checks would notice.

### Known broken, not fixable here

`train_ner.ipynb` c27 loads `neuralspace-reverie/indic-transformers-bn-bert`.
That checkpoint now returns **HTTP 401** from the Hub and appears withdrawn
(`?author=neuralspace-reverie` returns `[]`; a control model returns 307 then
200 on redirect, so the 401 is real and not a transport failure).
Training cannot run until it is mirrored or substituted. Unrelated to the
dependency work.

One stored cell *output* still shows a pre-fix call (`train_ner.ipynb` c0,
`load_metric`) — a historical execution record from 2021, intentionally left
untouched. No other output references a pre-fix API.

## Repo hygiene

- **`.dvc/config` is empty** — no DVC remote is defined, so `dvc pull` cannot
  resolve anything. The 17 `.dvc` pointers carry only md5/size/path. A remote
  must be added before any notebook can run on real data.
- A root `.gitignore` was added covering `.env*`, `*.pem`, `*.key`,
  `credentials.json` and `.ipynb_checkpoints/`. There was none before, which is
  how the token below got committed. Consider enabling GitHub secret scanning
  with push protection, and `nbstripout` as a pre-commit hook.

## Secrets

**UNRESOLVED — requires action by the repo owner.**

`train_ner.ipynb` hardcoded a credential in `load_dataset(..., use_auth_token='...')`
across cells 8, 11 and 12. Removed from the working tree in `84050f8`.

**The provider is NOT confirmed.** The value is 40 characters of pure lowercase
hex with no prefix. That is *not* the Hugging Face format — HF tokens have been
`hf_`-prefixed since 2022. 40-lowercase-hex matches a **legacy GitHub personal
access token** and a **Weights & Biases API key**; `env.yml` pinned
`wandb==0.12.2`, so a W&B key was plausibly in use. It was identified as "an HF
token" purely from the variable name — do not repeat that inference.

Revoke in all three consoles rather than trying to identify it first
(revocation is free and idempotent):

- `github.com/settings/tokens` (classic PATs — the closest format match)
- `wandb.ai/settings`
- `huggingface.co/settings/tokens`

**Removing it from HEAD did nothing for the exposure.** The repo is public. The
value sits in 3 blobs of `train_ner.ipynb` reachable from `HEAD`, and both
`edf7962` (first leak, 2023-02-24) and `8f6ab20` return **HTTP 200** to an
unauthenticated GitHub API request today. Exposure is ~1,293 days and ongoing,
so assume it has been harvested by automated scanners.

Full remediation, in order:

1. Revoke/rotate in all three consoles. Do this first.
2. Audit each account's security log for use during the exposure window.
3. Purge history (`git filter-repo --replace-text`, then force-push).
4. Ask GitHub Support to garbage-collect the unreferenced objects — after a
   force-push a commit stays fetchable by SHA until they do. Step 3 is
   cosmetic without this.
5. Prevent recurrence: the root `.gitignore` added in this pass, plus GitHub
   secret scanning with push protection and an `nbstripout` pre-commit hook.

Never commit a token. Use an environment variable (`HF_TOKEN` is read
automatically by `huggingface_hub`).

Also disclosed, low severity: `env.yml:187` carries
`prefix: /Users/mdmmn378/miniconda3/envs/hawker`, exposing a third party's
macOS username and an unrelated internal project name. Scrub it if history is
rewritten in step 3.
