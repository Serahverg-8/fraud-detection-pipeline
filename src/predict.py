"""
Kaggle submission — Milestone 5.

Retrains the winning model (LightGBM, best hyperparams from Milestone 4's
Optuna sweep, pulled from MLflow) on 100% of the labeled training data,
then scores the competition's test set for submission.

Feature engineering and preprocessing run on train+test COMBINED (not
separately) before splitting back apart -- test transactions are
chronologically after train in this dataset, so the card aggregation
features need train's history to be correct, and categorical integer
codes must be fit once across both so train and test agree on what each
code means.

Usage:
    python -m src.predict
"""

import lightgbm as lgb
import mlflow
import pandas as pd

from src.features import add_card_aggregate_features
from src.train import DATA_DIR, MLFLOW_EXPERIMENT_NAME, drop_pruned_features, preprocess

SUBMISSION_PATH = "submissions/submission.csv"


def load_best_lgbm_params() -> dict:
    """Pull the winning LightGBM hyperparams from the Milestone 4 Optuna
    sweep already logged to MLflow, rather than hardcoding them here where
    they'd silently go stale if tuning is ever re-run.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        filter_string="params.run_type = 'tuning_lightgbm'",
        order_by=["metrics.pr_auc DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No tuning_lightgbm runs found in MLflow -- run `python -m src.tune --model lightgbm` first.")
    best = runs.iloc[0]
    return {
        "num_leaves": int(best["params.num_leaves"]),
        "learning_rate": float(best["params.learning_rate"]),
        "n_estimators": int(best["params.n_estimators"]),
        "min_child_samples": int(best["params.min_child_samples"]),
        "subsample": float(best["params.subsample"]),
        "colsample_bytree": float(best["params.colsample_bytree"]),
    }


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """test_identity.csv uses hyphens (id-01) where train_identity.csv uses
    underscores (id_01) -- a known quirk of this competition's files.
    Without this rename, merging train+test identity data silently produces
    all-NaN identity columns for every test row instead of erroring.
    """
    return df.rename(columns=lambda c: c.replace("-", "_"))


def load_combined_data() -> pd.DataFrame:
    train_tx = pd.read_csv(f"{DATA_DIR}/train_transaction.csv")
    train_id = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
    test_tx = pd.read_csv(f"{DATA_DIR}/test_transaction.csv")
    test_id = normalize_identity_columns(pd.read_csv(f"{DATA_DIR}/test_identity.csv"))

    train_tx = train_tx.copy()
    test_tx = test_tx.copy()
    train_tx["is_train"] = True
    test_tx["is_train"] = False

    combined_tx = pd.concat([train_tx, test_tx], ignore_index=True)
    combined_id = pd.concat([train_id, test_id], ignore_index=True)
    return combined_tx.merge(combined_id, on="TransactionID", how="left")


def main():
    print("Loading combined train+test data...")
    combined = load_combined_data()

    print("Adding per-card aggregation features (computed across train+test, time-ordered)...")
    combined = add_card_aggregate_features(combined)

    print("Preprocessing (categorical codes fit once across train+test)...")
    combined = preprocess(combined)
    combined = drop_pruned_features(combined)

    train_df = combined[combined["is_train"]]
    test_df = combined[~combined["is_train"]]

    drop_cols = ["isFraud", "TransactionID", "TransactionDT", "is_train"]
    feature_cols = [c for c in combined.columns if c not in drop_cols]

    X_train, y_train = train_df[feature_cols], train_df["isFraud"]
    X_test = test_df[feature_cols]

    print("Loading winning LightGBM hyperparams from MLflow...")
    params = load_best_lgbm_params()
    print(f"Params: {params}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"Training final model on all {len(X_train)} labeled rows...")
    model = lgb.LGBMClassifier(
        **params,
        subsample_freq=1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    print(f"Scoring {len(X_test)} test rows...")
    test_probs = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        "TransactionID": test_df["TransactionID"].astype(int),
        "isFraud": test_probs,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
