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
- [ ] Milestone 2 — Feature engineering (group aggregations vs. baseline PR-AUC)
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
python src/train.py
```

Trains the baseline XGBoost model, prints PR-AUC/ROC-AUC/precision/recall
on a held-out (time-based) validation split, and saves the model to
`models/baseline_xgb.json` (gitignored).
