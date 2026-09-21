# Fraud Detection Pipeline — Project Plan

## Goal

A lightweight but complete end-to-end ML system: data → features → model →
tuning → serving → monitoring. Built incrementally, one milestone per week,
using the IEEE-CIS Fraud Detection dataset (real e-commerce transaction data
from a payments company).

This is a portfolio project — the point is to demonstrate judgment across the
full ML lifecycle, not just model accuracy. Revised after Milestone 2 to also
target an actual Kaggle leaderboard submission, not just a documented score.

## Why this project

Directly adjacent to real payments/fraud ML work. Built to be honest about
tradeoffs (documented in the README as it evolves) rather than polished for
show.

## Repo structure

```
fraud-detection-pipeline/
├── README.md              # problem, approach, results, how to run
├── plan.md                # this file
├── data/                  # raw + processed (gitignored)
├── notebooks/              # exploratory work — EDA, experiments
├── src/
│   ├── features.py         # feature engineering
│   ├── train.py             # training script
│   ├── tune.py               # hyperparameter tuning (Milestone 4)
│   ├── predict.py            # test-set predictions for Kaggle submission (Milestone 5)
│   ├── serve.py             # FastAPI scoring endpoint (Milestone 6)
│   └── monitor.py           # drift check (Milestone 7)
├── models/                 # saved model artifacts (gitignored)
├── mlflow.db               # MLflow run history (gitignored)
├── Dockerfile               # Milestone 8
└── requirements.txt
```

## Milestones

1. **Baseline** — minimal EDA, a plain model (logistic regression or basic
   XGBoost), running end to end with no feature engineering yet. Goal:
   something runs, not correctness.
2. **Feature engineering** — group aggregations (e.g. average transaction
   amount per card over a rolling window). Compare PR-AUC against baseline.
3. **MLflow experiment tracking** — local MLflow set up, logging params/
   metrics/artifacts for every run going forward (including backfilling the
   Milestone 1/2 results as a starting history).
4. **Model experimentation** — systematic tuning (Optuna) of XGBoost,
   LightGBM, and CatBoost on the Milestone 2 feature set, plus TabM (a
   parameter-efficient deep learning model — genuine tree-vs-deep-learning
   comparison, unlike TabPFN/TabFM/TabICL which don't fit this dataset's
   scale, see README). Every trial logged to MLflow. Pick the best run.
5. **Kaggle submission** — generate predictions on the competition's
   `test_transaction.csv` with the best tuned model and submit for an actual
   leaderboard score.
6. **Serving** — a small FastAPI endpoint that scores one transaction, using
   the best model from Milestone 4/5.
7. **Monitoring** — a basic drift check on one feature's distribution over
   time.
8. **Containerize** — a Dockerfile for reproducibility outside the dev
   machine (promoted from a stretch goal — needed for the finished pipeline).

## Working rhythm

- One milestone worked on per Thursday session (~30–60 min), most milestones
  will span more than one week.
- Commit + update README after each session, even if the milestone isn't
  finished — partial progress with a note beats a silent gap.
- Log Kaggle-inspired techniques tried (and whether they helped) directly in
  the README's "approach" section as they're added.
- Keep the codebase and process lightweight — this is iterative, first-pass
  work with expected retries, not a production system. Don't over-build.

## Status

- [x] Milestone 1 — Baseline (PR-AUC 0.513)
- [x] Milestone 2 — Feature engineering (PR-AUC 0.520; 0.521 after a feature
      pruning detour — see README)
- [ ] Milestone 3 — MLflow experiment tracking
- [ ] Milestone 4 — Model experimentation (tuning + TabM)
- [ ] Milestone 5 — Kaggle submission
- [ ] Milestone 6 — Serving
- [ ] Milestone 7 — Monitoring
- [ ] Milestone 8 — Containerize
