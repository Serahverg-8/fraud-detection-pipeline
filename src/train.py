"""
Training script — Milestones 1 & 2.

Trains an XGBoost classifier on the IEEE-CIS Fraud Detection dataset.
By default it includes the Milestone 2 per-card aggregation features
(src/features.py); pass --baseline to reproduce the Milestone 1 result
(no feature engineering) for comparison.

Usage:
    python src/train.py               # with engineered features
    python src/train.py --baseline    # Milestone 1 baseline, for comparison
"""

import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import add_card_aggregate_features

DATA_DIR = "data/raw"
DROPPED_FEATURES_PATH = "src/dropped_features.json"
MLFLOW_EXPERIMENT_NAME = "fraud-detection"

# Columns that were >90% missing in EDA (notebooks/01_eda.ipynb) — dropping
# for the baseline rather than imputing; revisit if they turn out to matter.
HIGH_MISSING_COLS = [
    "id_24", "id_25", "id_07", "id_08", "id_21", "id_26", "id_27", "id_23", "id_22",
]


def load_data():
    transaction = pd.read_csv(f"{DATA_DIR}/train_transaction.csv")
    identity = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
    df = transaction.merge(identity, on="TransactionID", how="left")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in HIGH_MISSING_COLS if c in df.columns]).copy()

    # log1p transform for the one clearly right-skewed numeric feature
    # (see EDA notebook — raw amount is heavily skewed)
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Everything non-numeric gets integer-coded rather than one-hot encoded.
    # XGBoost splits on integer codes just fine, and one-hot would explode
    # high-cardinality columns like card1 (13k+ unique values) into
    # thousands of sparse columns for no benefit at the baseline stage.
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("missing").astype("category").cat.codes

    return df


def drop_pruned_features(df: pd.DataFrame, dropped_path: str = DROPPED_FEATURES_PATH) -> pd.DataFrame:
    """Drop columns listed in a JSON file of low-value/redundant feature names
    (produced by notebooks/02_feature_importance.ipynb). Returns df unchanged
    if the file doesn't exist, and silently ignores listed names that aren't
    present as columns.
    """
    if not os.path.exists(dropped_path):
        return df
    with open(dropped_path) as f:
        dropped_cols = json.load(f)
    return df.drop(columns=[c for c in dropped_cols if c in df.columns])


def build_run_params(args, model_params: dict, scale_pos_weight: float, n_features: int) -> dict:
    """Decide what to log as MLflow params for this run, given the CLI flags
    that determine which variant (baseline / features / features_pruned) is
    being trained.
    """
    if args.baseline:
        run_type = "baseline"
    elif args.prune:
        run_type = "features_pruned"
    else:
        run_type = "features"
    return {
        "run_type": run_type,
        "scale_pos_weight": scale_pos_weight,
        "n_features": n_features,
        **model_params,
    }


def log_run(params: dict, metrics: dict) -> str:
    """Log a completed run's params and metrics to MLflow. Returns the run ID."""
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return run.info.run_id


def time_based_split(df: pd.DataFrame, val_frac: float = 0.2):
    # TransactionDT is seconds-since-a-reference-point, i.e. a proxy for
    # transaction order. Splitting on time (train = earlier, val = later)
    # instead of a random split avoids leaking future information into
    # training and better mimics how the model would actually be used —
    # scoring transactions it hasn't seen yet.
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Skip Milestone 2 feature engineering (reproduces the Milestone 1 baseline).",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Drop low-value/redundant features listed in src/dropped_features.json "
        "(produced by notebooks/02_feature_importance.ipynb).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading data...")
    df = load_data()

    if not args.baseline:
        print("Adding per-card aggregation features...")
        df = add_card_aggregate_features(df)

    print("Preprocessing...")
    df = preprocess(df)

    if args.prune:
        before = df.shape[1]
        df = drop_pruned_features(df)
        print(f"Pruned {before - df.shape[1]} low-value/redundant features.")

    train_df, val_df = time_based_split(df)
    print(f"Train: {train_df.shape}, Val: {val_df.shape}")

    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train, y_train = train_df[feature_cols], train_df["isFraud"]
    X_val, y_val = val_df[feature_cols], val_df["isFraud"]

    # scale_pos_weight compensates for the ~3.5% fraud rate by upweighting
    # the minority class during training — without it XGBoost would
    # happily predict "not fraud" for nearly everything and still get a
    # low loss, since negatives dominate.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model_params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
    model = xgb.XGBClassifier(
        **model_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
    )

    print("Training...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print("\nEvaluating on validation set (held-out, later in time)...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    pr_auc = average_precision_score(y_val, val_probs)
    roc_auc = roc_auc_score(y_val, val_probs)
    fraud_precision = precision_score(y_val, val_preds)
    fraud_recall = recall_score(y_val, val_preds)

    print(f"\nPR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}  (reported for reference — PR-AUC is the metric that matters here)")
    print("\nClassification report @ threshold 0.5:")
    print(classification_report(y_val, val_preds, target_names=["legit", "fraud"]))

    run_params = build_run_params(args, model_params, scale_pos_weight, len(feature_cols))
    run_metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fraud_precision": fraud_precision,
        "fraud_recall": fraud_recall,
    }
    run_id = log_run(run_params, run_metrics)
    print(f"Logged to MLflow, run_id={run_id}")

    if args.baseline:
        model_path = "models/baseline_xgb.json"
    elif args.prune:
        model_path = "models/features_pruned_xgb.json"
    else:
        model_path = "models/features_xgb.json"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
