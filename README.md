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

- [ ] Milestone 1 — Baseline (EDA + plain model, no feature engineering)
- [ ] Milestone 2 — Feature engineering (group aggregations vs. baseline PR-AUC)
- [ ] Milestone 3 — Experiment tracking
- [ ] Milestone 4 — Serving (FastAPI scoring endpoint)
- [ ] Milestone 5 — Monitoring (drift check)
- [ ] Milestone 6 — Containerize (stretch)

See [plan.md](plan.md) for the full milestone breakdown and working rhythm.

## Approach & what didn't work

_(Updated as milestones progress — this section is meant to be honest about
tradeoffs, not just report final numbers.)_

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

_(To be filled in once Milestone 1's training script lands.)_
