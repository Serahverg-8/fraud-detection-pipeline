"""
Baseline training script — Milestone 1.

Trains a plain XGBoost classifier on the IEEE-CIS Fraud Detection dataset
with NO feature engineering: just enough preprocessing to make the raw
columns model-able (drop near-empty columns, encode categoricals as
integer codes, log-transform the one obvious numeric feature). The goal is
an honest end-to-end run to compare future feature-engineering work
against, not a tuned model.

Usage:
    python src/train.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

DATA_DIR = "data/raw"
MODEL_PATH = "models/baseline_xgb.json"

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
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    df = preprocess(df)

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

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
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

    print(f"\nPR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}  (reported for reference — PR-AUC is the metric that matters here)")
    print("\nClassification report @ threshold 0.5:")
    print(classification_report(y_val, val_preds, target_names=["legit", "fraud"]))

    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
