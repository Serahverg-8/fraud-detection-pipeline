# Fraud Detection Pipeline

A lightweight, end-to-end fraud detection system built on the [IEEE-CIS Fraud
Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection): data →
features → model → serving → monitoring.

Built incrementally as a portfolio project, one milestone at a time. The goal
is to demonstrate judgment across the full ML lifecycle — not just squeeze
out the best leaderboard score.

## Why this project

Adjacent to real payments/fraud ML work. Class imbalance is severe (~3.5%
fraud rate), so evaluation focuses on **precision, recall, and PR-AUC** —
accuracy is close to meaningless here.

## Status

- [x] Milestone 1 — Baseline (EDA + plain model, no feature engineering)
- [x] Milestone 2 — Feature engineering (group aggregations vs. baseline PR-AUC)
- [ ] Milestone 3 — Experiment tracking
- [ ] Milestone 4 — Serving (FastAPI scoring endpoint)
- [ ] Milestone 5 — Monitoring (drift check)
- [ ] Milestone 6 — Containerize (stretch)

See [plan.md](plan.md) for the full milestone breakdown and working rhythm.

## Approach & what didn't work

_(Updated as milestones progress — this section is meant to be honest about
tradeoffs, not just report final numbers.)_

### Milestone 1 — Baseline

**Approach:** XGBoost on the raw transaction + identity tables, joined on
`TransactionID`. No feature engineering — just enough preprocessing to make
the columns usable:
- Dropped the ~9 `id_*` columns that were >99% missing (per EDA)
- Log-transformed `TransactionAmt` (heavily right-skewed)
- Integer-coded all categorical columns rather than one-hot encoding —
  `card1` alone has 13k+ unique values, so one-hot would have exploded the
  feature space for no benefit at the baseline stage
- **Time-based train/val split** (80/20, sorted by `TransactionDT`), not
  random — avoids leaking future transactions into training and mimics how
  the model would actually be scored in production
- `scale_pos_weight` to compensate for the ~3.5% fraud rate

**Results** (held-out validation, later in time than training):
- PR-AUC: **0.513**
- ROC-AUC: 0.902 (reported for reference only — see below)
- At the default 0.5 threshold: 70% recall, 23% precision on the fraud class

**What didn't work / tradeoffs:**
- The gap between PR-AUC (0.51) and ROC-AUC (0.90) is the imbalance problem
  made visible — ROC-AUC looks strong mostly because the negative class is
  huge, so it's not a reliable signal of how useful the model actually is.
  PR-AUC is the number to track going forward.
- 23% precision at 70% recall means ~3 out of 4 flagged transactions would
  be false alarms at this threshold — not production-usable as-is. Milestone
  2's feature engineering and a proper threshold-selection pass (rather than
  the default 0.5) are the next levers, not more model tuning on this same
  feature set.
- No hyperparameter tuning was done on purpose — the point of a baseline is
  a fair comparison point for future feature work, not the best possible
  score on this feature set.

### Milestone 2 — Feature engineering

**Approach:** added 6 leakage-safe per-card aggregation features
(`src/features.py`), grouped by `card1` (the closest thing to a card
identifier in the anonymized data):
- `card1_count_expanding`, `card1_amt_mean_expanding`, `card1_amt_std_expanding`
  — stats over *all prior* transactions for that card
- `card1_count_24h`, `card1_amt_mean_24h` — stats over the *trailing 24h* only
- `amt_to_card1_mean_ratio` — current amount ÷ that card's running mean (an
  "is this transaction out of pattern for this card?" signal)

**Leakage safety** was the main design constraint, not the feature list
itself: every stat is computed from transactions strictly *before* the
current one (via `shift`/`rolling(closed='left')`), and features are
computed on the full time-sorted dataset *before* the train/val split — a
validation-row is allowed to see history from earlier training rows (that's
how the model would actually be used in production), but never the
reverse. Covered by 7 unit tests in `tests/test_features.py`, including an
explicit test that mutating a later transaction's data doesn't change an
earlier transaction's computed features.

**Results** (`python -m src.train`, same time-based split as Milestone 1):
- PR-AUC: 0.513 → **0.520**
- ROC-AUC: 0.902 → 0.900 (essentially flat, as expected — see Milestone 1
  notes on why ROC-AUC is a poor discriminator of quality here)
- Recall at threshold 0.5 ticked up slightly (70% → 71%), precision flat (23%)

**What didn't work / tradeoffs:**
- The PR-AUC gain (+0.007) is real but modest — a single grouping key
  (`card1`) and a narrow feature set moves the needle less than expected
  going in. This tracks with fraud-detection literature: aggregation
  features tend to help more in combination with several grouping keys
  (e.g. card × merchant, card × device) than any one key alone.
  Worth revisiting with `addr1` or a card+`ProductCD` combination if there's
  a follow-up pass, but out of scope for keeping this milestone's feature
  set small enough to cleanly attribute the PR-AUC change to.
- Computing exact leakage-safe rolling stats via `groupby().apply()` (used
  for the 24h window, to keep row alignment provably correct rather than
  relying on `groupby().rolling()`'s output ordering) is a few seconds
  slower than a fully vectorized approach. At 590k rows it's a non-issue
  (~2.5s total); would need revisiting at meaningfully larger scale.

### Feature pruning (quick detour, not a formal milestone)

`notebooks/02_feature_importance.ipynb` checked the trained model for dead
weight: 56 features the model never split on, plus 71 `V`-columns highly
correlated (>0.95) with another `V`-column. Dropping all 116 (`python -m
src.train --prune`) held PR-AUC essentially flat (0.520 → 0.521) with 27%
fewer features — a nice confirmation that a chunk of the anonymized
`V`-columns really are redundant, not independent signal.

## Repo structure

```
fraud-detection-pipeline/
├── README.md
├── plan.md
├── data/              # raw + processed data (gitignored)
├── notebooks/         # exploratory work — EDA, experiments
├── src/
│   ├── features.py    # feature engineering
│   ├── train.py       # training script
│   ├── serve.py       # FastAPI scoring endpoint
│   └── monitor.py     # drift check
├── models/            # saved model artifacts (gitignored)
├── tests/             # unit tests (pytest)
├── experiments.csv    # run log (or mlruns/ if using MLflow)
├── Dockerfile          # stretch goal
└── requirements.txt
```

## How to run

### Setup

```bash
conda env create -f environment.yml
conda activate fraud-detection

# macOS only: XGBoost needs the OpenMP runtime
brew install libomp
```

### Data

Download the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection/data)
(requires a free Kaggle account and accepting the competition rules), then
unzip into `data/raw/`:

```
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
├── test_identity.csv
└── sample_submission.csv
```

`data/` is gitignored — the raw CSVs (~1.3GB total) are not committed.

### Training

```bash
python -m src.train              # with Milestone 2 engineered features
python -m src.train --baseline   # Milestone 1 baseline, for comparison
```

Prints PR-AUC/ROC-AUC/precision/recall on a held-out (time-based)
validation split, and saves the model to `models/features_xgb.json` (or
`models/baseline_xgb.json` with `--baseline`) — both gitignored.

### Tests

```bash
python -m pytest tests/ -v
```
