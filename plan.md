# Fraud Detection Pipeline — Project Plan

## Goal

A lightweight but complete end-to-end ML system: data → features → model →
serving → monitoring. Built incrementally, one milestone per week, using the
IEEE-CIS Fraud Detection dataset (real e-commerce transaction data from a
payments company).

This is a portfolio project — the point is to demonstrate judgment across the
full ML lifecycle, not just model accuracy.

## Why this project

Directly adjacent to real payments/fraud ML work. Built to be honest about
tradeoffs (documented in the README as it evolves) rather than polished for
show.

## Repo structure

```
fraud-detection-pipeline/
├── README.md              # problem, approach, results, how to run
├── plan.md                # this file
├── data/                  # raw + processed (or a fetch script)
├── notebooks/              # exploratory work — EDA, experiments
├── src/
│   ├── features.py         # feature engineering
│   ├── train.py             # training script
│   ├── serve.py             # FastAPI scoring endpoint
│   └── monitor.py           # drift check
├── models/                 # saved model artifacts
├── experiments.csv          # or mlruns/ if using MLflow
├── Dockerfile               # stretch goal
└── requirements.txt
```

## Milestones

1. **Baseline** — minimal EDA, a plain model (logistic regression or basic
   XGBoost), running end to end with no feature engineering yet. Goal:
   something runs, not correctness.
2. **Feature engineering** — group aggregations (e.g. average transaction
   amount per card over a rolling window). Compare PR-AUC against baseline.
3. **Experiment tracking** — every run's params/metrics logged (CSV or local
   MLflow).
4. **Serving** — a small FastAPI endpoint that scores one transaction.
5. **Monitoring** — a basic drift check on one feature's distribution over
   time.
6. **Containerize (stretch)** — a Dockerfile for reproducibility outside the
   dev machine.

## Working rhythm

- One milestone worked on per Thursday session (~30–60 min), most milestones
  will span more than one week.
- Commit + update README after each session, even if the milestone isn't
  finished — partial progress with a note beats a silent gap.
- Log Kaggle-inspired techniques tried (and whether they helped) directly in
  the README's "approach" section as they're added.

## Status

- [ ] Milestone 1 — Baseline
- [ ] Milestone 2 — Feature engineering
- [ ] Milestone 3 — Experiment tracking
- [ ] Milestone 4 — Serving
- [ ] Milestone 5 — Monitoring
- [ ] Milestone 6 — Containerize (stretch)
