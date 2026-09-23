# Fraud Detection Pipeline

A lightweight, end-to-end fraud detection system built on the [IEEE-CIS Fraud
Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection): data →
features → tuning → serving → monitoring, ending in an actual Kaggle
leaderboard submission.

Built incrementally as a portfolio project, one milestone at a time. The goal
is to demonstrate judgment across the full ML lifecycle, not just chase a
leaderboard number — but a real submission at the end is still the honest
way to prove the model works.

## Why this project

Adjacent to real payments/fraud ML work. Class imbalance is severe (~3.5%
fraud rate), so evaluation focuses on **precision, recall, and PR-AUC** —
accuracy is close to meaningless here.

## Status

- [x] Milestone 1 — Baseline (EDA + plain model, no feature engineering)
- [x] Milestone 2 — Feature engineering (group aggregations vs. baseline PR-AUC)
- [x] Milestone 3 — MLflow experiment tracking
- [x] Milestone 4 — Model experimentation (XGBoost/LightGBM/CatBoost tuning + TabM)
- [x] Milestone 5 — Kaggle submission (Private LB 0.9261 ROC-AUC)
- [ ] Milestone 6 — Serving (FastAPI scoring endpoint)
- [ ] Milestone 7 — Monitoring (drift check)
- [ ] Milestone 8 — Containerize

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

### Milestone 3 — MLflow experiment tracking

Every `python -m src.train` run now logs its params (run type, hyperparams,
`scale_pos_weight`, feature count) and metrics (PR-AUC, ROC-AUC, fraud
precision/recall) to a local MLflow instance (`sqlite:///mlflow.db` — MLflow
3.x deprecated the plain-file `mlruns/` backend, so SQLite is the current
recommended local setup). Backfilled the 3 existing runs (baseline,
features, features+pruned) as a starting history. View with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Milestone 4 — Model experimentation

**XGBoost tuning done:** 30-trial Optuna search (`python -m src.tune`) over
`max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`,
`subsample`, `colsample_bytree` on the pruned feature set. Every trial
logged to MLflow (`run_type=tuning_xgboost`).

- PR-AUC: 0.521 → **0.578** (best trial: `max_depth=10, learning_rate=0.15,
  n_estimators=437, min_child_weight=9, subsample=0.73,
  colsample_bytree=0.92`)
- **Didn't fully explore:** the best trials consistently pushed toward
  `max_depth=10`, the upper edge of the search space — there's likely more
  headroom with deeper trees. Not re-run with a wider range this pass
  (diminishing time-value for a portfolio project vs. moving on to
  LightGBM/CatBoost/TabM), but noted honestly rather than presenting 0.578
  as a ceiling.

**LightGBM tuning done:** same 30-trial Optuna pattern (`python -m src.tune
--model lightgbm`), search space over `num_leaves`, `learning_rate`,
`n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`.

- PR-AUC: **0.597** — better than tuned XGBoost (0.578). ROC-AUC: 0.913.
  (best trial: `num_leaves=251, learning_rate=0.091, n_estimators=496,
  min_child_samples=64, subsample=0.83, colsample_bytree=0.89`)
- Same pattern as XGBoost: best trial landed at `num_leaves=251`, right at
  the search space's upper bound (255) — likely more headroom, not
  re-explored this pass for the same reason as above.

**CatBoost tuning done:** same pattern (`python -m src.tune --model
catboost`), search space over `depth`, `learning_rate`, `n_estimators`,
`l2_leaf_reg`, `subsample`.

- PR-AUC: **0.519** — notably *worse* than both XGBoost (0.578) and
  LightGBM (0.597), despite `depth` also hitting the search space's upper
  bound (10). ROC-AUC: 0.910.
- **Likely cause, not just bad luck:** CatBoost's actual strength is its
  native categorical handling (ordered target encoding on raw category
  values), but our shared `preprocess()` step already integer-codes every
  categorical column before any model sees it — so CatBoost was fed the
  same flattened numeric features as XGBoost/LightGBM and never got to use
  the one thing it's specifically good at. A fairer CatBoost comparison
  would pass it the raw categorical columns directly (via `cat_features`)
  instead of pre-encoded ones. Not redone this pass — noting it as the
  probable explanation rather than concluding CatBoost is simply worse
  here.

**TabM baseline done:** one run (`python -m src.tune_tabm`) with a fixed,
reasonable architecture (`n_blocks=3, d_block=256, dropout=0.1, k=8`) —
not a full Optuna sweep like the tree models got (see Milestone 4 design
note above), since the goal was checking whether a deep learning model is
competitive here at all before investing in tuning it further. Trained on
the Mac's M5 GPU via PyTorch's MPS backend, 15 epochs, ~3.5 min total.

- PR-AUC: **0.432**, ROC-AUC: 0.836 — the worst of the 4 models.
- **Training was noticeably unstable:** PR-AUC swung between 0.22 and 0.43
  across epochs rather than improving smoothly, and ROC-AUC actually
  *declined* over training (0.86 → 0.83). Likely causes: a fixed learning
  rate with no schedule or early stopping, and 15 epochs is a light
  training budget for a from-scratch deep model on this much data —
  neural tabular models typically need considerably more epochs and
  careful LR scheduling to converge well, which trees don't.
- Given the scope decision to do one baseline run rather than a full
  sweep, this result reads as "TabM needs real tuning investment to be
  competitive here," not "TabM doesn't work for fraud detection."
  Honest as-is: it's not the model we'd pick without further work.

### Milestone 4 summary

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| LightGBM (tuned) | **0.597** | 0.913 |
| XGBoost (tuned) | 0.578 | 0.904 |
| CatBoost (tuned) | 0.519 | 0.910 |
| TabM (untuned baseline) | 0.432 | 0.836 |

**LightGBM is the winner** and will be the model carried into Milestone 5
(Kaggle submission).

### Milestone 5 — Kaggle submission

`python -m src.predict` retrains LightGBM (winning Milestone 4
hyperparams, pulled from MLflow rather than hardcoded) on **100% of the
labeled training data** — the validation split's job was model selection,
which is done — then scores the competition's actual test set.

Two correctness details that would have silently broken this without
catching them:
- **`test_identity.csv` uses hyphens** (`id-01`) **where
  `train_identity.csv` uses underscores** (`id_01`) — a known quirk of
  this competition's files. Without normalizing this, the merge would
  silently produce all-NaN identity columns for every test row instead of
  erroring.
- **Feature engineering and categorical encoding run on train+test
  combined**, not separately, before splitting back apart. Test
  transactions are chronologically after train in this dataset, so the
  card aggregation features need train's history to be correct; and
  categorical integer codes must be fit once across both, or train and
  test could silently disagree on what code N means for a given column.

Result: `submissions/submission.csv`, 506,691 predictions, mean predicted
probability 3.4% (close to the training set's 3.5% fraud rate — a good
calibration sanity check), no NaNs, no duplicate `TransactionID`s.

**Uploaded to Kaggle** (manually — the API's newer token format isn't
supported by the `kaggle`/`kagglehub` packages, same issue as the
original dataset download in Milestone 1):

- **Private LB (the actual final score): 0.9261 ROC-AUC**
- Public LB: 0.8868 ROC-AUC

For context: this competition's 1st place solution — an ensemble of
tuned XGBoost/LightGBM/CatBoost plus a "UID" reconstruction trick to
identify the real client behind anonymized transactions (see next
section) — scored 0.9459 private. Their *individual*, non-ensembled
models scored 0.93–0.94. Landing at 0.9261 with a single model, no UID
feature engineering, and a 30-trial tuning budget is a solid result for
where this project chose to stop — not top-leaderboard, but well within
range of a credible single-model submission.

### What we deliberately didn't do (and why)

Documented here rather than silently omitted, since the point of this
project is honest tradeoffs, not chasing the highest possible number:

- **UID reconstruction** (`card1` + `addr1` + normalized `D1` to
  reconstruct the real entity behind anonymized transactions) — the
  single biggest lever in the winning solutions. Would very likely close
  a meaningful chunk of the 0.02 gap to the top individual models. Skipped
  to keep this project's feature engineering scope to what we could
  clearly attribute PR-AUC changes to (Milestone 2's design goal), and
  because 0.926 was already a good enough result to stop at without it.
- **Ensembling** (blending XGBoost + LightGBM + CatBoost predictions) —
  the other big lever the winners used. Straightforward to add later if
  revisited; not done here since Milestone 4 was about comparing models
  individually, not combining them.
- **CatBoost with native categorical handling** (`cat_features`, instead
  of pre-integer-encoded columns) — noted in Milestone 4 as the likely
  fix for CatBoost's underperformance; not retried.

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
│   ├── tune.py        # hyperparameter tuning (XGBoost/LightGBM/CatBoost)
│   ├── tune_tabm.py   # TabM (deep learning) baseline
│   ├── predict.py     # Kaggle test-set predictions
│   ├── serve.py       # FastAPI scoring endpoint
│   └── monitor.py     # drift check
├── models/            # saved model artifacts (gitignored)
├── submissions/       # Kaggle submission CSVs (gitignored)
├── tests/             # unit tests (pytest)
├── mlflow.db          # MLflow run history (gitignored)
├── Dockerfile          # Milestone 8
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
