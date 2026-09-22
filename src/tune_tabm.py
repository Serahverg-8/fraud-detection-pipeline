"""
TabM baseline — Milestone 4.

TabM (https://github.com/yandex-research/tabm) is a parameter-efficient
ensemble-of-MLPs model -- unlike TabPFN/TabFM/TabICL (considered and
rejected, see README), it's trained normally via mini-batch gradient
descent, so it has no dataset-size ceiling.

Scope: ONE run with a reasonable default architecture, not a full Optuna
sweep like the GBT models got (see README for why) -- the goal here is to
see whether a deep learning model is competitive at all on this data
before investing in tuning it.

Usage:
    python -m src.tune_tabm
"""

import numpy as np
import pandas as pd
import tabm
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features import add_card_aggregate_features
from src.train import (
    MLFLOW_EXPERIMENT_NAME,
    drop_pruned_features,
    load_data,
    log_run,
    preprocess,
    time_based_split,
)

import mlflow

EPOCHS = 15
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
ARCHITECTURE = {"n_blocks": 3, "d_block": 256, "dropout": 0.1, "k": 8, "arch_type": "tabm"}


def identify_categorical_columns(raw_df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Which of feature_cols were originally object/str dtype in the raw
    (pre-preprocess) data -- these are the ones TabM should treat as
    categorical (via cat_cardinalities) rather than as plain numbers.
    """
    object_cols = set(raw_df.select_dtypes(include=["object", "str"]).columns)
    return [c for c in feature_cols if c in object_cols]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading and preparing data (pruned feature set)...")
    raw_df = load_data()
    raw_df = add_card_aggregate_features(raw_df)
    df = preprocess(raw_df)
    df = drop_pruned_features(df)

    train_df, val_df = time_based_split(df)
    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    categorical_cols = identify_categorical_columns(raw_df, feature_cols)
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    print(f"{len(numeric_cols)} numeric features, {len(categorical_cols)} categorical features")

    # Unlike trees, neural nets can't handle NaN or unscaled inputs --
    # impute (median) then standardize, both fit on train only to avoid
    # leaking validation statistics.
    imputer = SimpleImputer(strategy="median").fit(train_df[numeric_cols])
    scaler = StandardScaler().fit(imputer.transform(train_df[numeric_cols]))

    def to_tensors(split_df):
        x_num_imputed = imputer.transform(split_df[numeric_cols])
        x_num = torch.tensor(scaler.transform(x_num_imputed), dtype=torch.float32)
        x_cat = torch.tensor(split_df[categorical_cols].to_numpy(), dtype=torch.long)
        y = torch.tensor(split_df["isFraud"].to_numpy(), dtype=torch.float32)
        return x_num.to(device), x_cat.to(device), y.to(device)

    X_train_num, X_train_cat, y_train = to_tensors(train_df)
    X_val_num, X_val_cat, y_val = to_tensors(val_df)

    # Cardinalities are computed on the full (pre-split) df -- these are
    # category value ranges, not label information, so this is not leakage
    # the same way it wouldn't be leakage to know card1 ranges 0..13552.
    cat_cardinalities = [int(df[c].max()) + 1 for c in categorical_cols]

    model = tabm.TabM.make(
        n_num_features=len(numeric_cols),
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        **ARCHITECTURE,
    ).to(device)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=scale_pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n_train = len(y_train)
    best_pr_auc = -1.0
    best_metrics = None

    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            # TabM outputs (batch, k, 1) -- one prediction per ensemble
            # member; average over k to train them together (see module
            # docstring: this is what makes TabM a *parallel* ensemble).
            logits = model(X_train_num[idx], X_train_cat[idx]).squeeze(-1).mean(dim=1)
            loss = loss_fn(logits, y_train[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_num, X_val_cat).squeeze(-1).mean(dim=1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        y_val_np = y_val.cpu().numpy()

        pr_auc = average_precision_score(y_val_np, val_probs)
        roc_auc = roc_auc_score(y_val_np, val_probs)
        print(f"  epoch {epoch}: loss={loss.item():.4f}  PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_metrics = {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "fraud_precision": precision_score(y_val_np, val_preds),
                "fraud_recall": recall_score(y_val_np, val_preds),
            }
            torch.save(model.state_dict(), "models/tabm.pt")

    print(f"\nBest PR-AUC:  {best_metrics['pr_auc']:.4f}")
    print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")

    run_params = {
        "run_type": "tabm_baseline",
        "n_features": len(feature_cols),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        **ARCHITECTURE,
    }
    run_id = log_run(run_params, best_metrics)
    print(f"Logged to MLflow, run_id={run_id}")
    print("Best model saved to models/tabm.pt")


if __name__ == "__main__":
    main()
