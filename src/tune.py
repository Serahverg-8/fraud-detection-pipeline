"""
Hyperparameter tuning — Milestone 4 (XGBoost).

Runs an Optuna study over XGBoost hyperparameters on the Milestone 2
pruned feature set (our best result so far, PR-AUC 0.521). Every trial is
logged to MLflow (tagged run_type=tuning_xgboost) so trials can be sorted
and compared in the MLflow UI.

Usage:
    python -m src.tune
"""

import mlflow
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.features import add_card_aggregate_features
from src.train import (
    MLFLOW_EXPERIMENT_NAME,
    drop_pruned_features,
    load_data,
    log_run,
    preprocess,
    time_based_split,
)

N_TRIALS = 30


def suggest_xgb_params(trial: optuna.Trial) -> dict:
    """Map an Optuna trial to an XGBoost hyperparameter dict. Kept separate
    from the training/objective logic so the search space itself is
    testable without running a full study.
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading and preparing data (pruned feature set)...")
    df = load_data()
    df = add_card_aggregate_features(df)
    df = preprocess(df)
    df = drop_pruned_features(df)

    train_df, val_df = time_based_split(df)
    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train, y_train = train_df[feature_cols], train_df["isFraud"]
    X_val, y_val = val_df[feature_cols], val_df["isFraud"]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    best_state = {"pr_auc": -1.0, "model": None, "params": None}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        pr_auc = average_precision_score(y_val, val_probs)

        log_run(
            {"run_type": "tuning_xgboost", "scale_pos_weight": scale_pos_weight,
             "n_features": len(feature_cols), **params},
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc_score(y_val, val_probs),
                "fraud_precision": precision_score(y_val, val_preds),
                "fraud_recall": recall_score(y_val, val_preds),
            },
        )

        if pr_auc > best_state["pr_auc"]:
            best_state.update(pr_auc=pr_auc, model=model, params=params)

        return pr_auc

    print(f"Running {N_TRIALS} Optuna trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest PR-AUC: {best_state['pr_auc']:.4f}")
    print(f"Best params: {best_state['params']}")

    best_state["model"].save_model("models/tuned_xgb.json")
    print("Best model saved to models/tuned_xgb.json")


if __name__ == "__main__":
    main()
