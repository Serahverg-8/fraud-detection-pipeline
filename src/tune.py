"""
Hyperparameter tuning — Milestone 4.

Runs an Optuna study over hyperparameters on the Milestone 2 pruned
feature set (our best result so far without tuning, PR-AUC 0.521). Every
trial is logged to MLflow (tagged run_type=tuning_<model>) so trials can
be sorted and compared in the MLflow UI.

Usage:
    python -m src.tune                    # XGBoost (default)
    python -m src.tune --model lightgbm
"""

import argparse

import lightgbm as lgb
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


def suggest_lgbm_params(trial: optuna.Trial) -> dict:
    """LightGBM's equivalent search space -- num_leaves/min_child_samples
    stand in for XGBoost's max_depth/min_child_weight, same idea otherwise.
    """
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def build_xgb_model(params: dict, scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
    )


def build_lgbm_model(params: dict, scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **params,
        # subsample only takes effect with subsample_freq > 0 -- fixed at 1
        # (subsample every iteration) rather than tuned, to keep the search
        # space the same size as XGBoost's.
        subsample_freq=1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )


def fit_xgb_model(model, X_train, y_train, X_val, y_val) -> None:
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


def fit_lgbm_model(model, X_train, y_train, X_val, y_val) -> None:
    # LightGBM's sklearn API silences per-iteration logs via the
    # constructor's verbose=-1 (set in build_lgbm_model) rather than a
    # fit()-time argument.
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])


MODELS = {
    "xgboost": {
        "suggest": suggest_xgb_params, "build": build_xgb_model, "fit": fit_xgb_model, "save_ext": "json",
    },
    "lightgbm": {
        "suggest": suggest_lgbm_params, "build": build_lgbm_model, "fit": fit_lgbm_model, "save_ext": "txt",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="xgboost")
    args = parser.parse_args()
    model_spec = MODELS[args.model]

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

    best_state = {"pr_auc": -1.0, "roc_auc": None, "model": None, "params": None}

    def objective(trial: optuna.Trial) -> float:
        params = model_spec["suggest"](trial)
        model = model_spec["build"](params, scale_pos_weight)
        model_spec["fit"](model, X_train, y_train, X_val, y_val)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        pr_auc = average_precision_score(y_val, val_probs)
        roc_auc = roc_auc_score(y_val, val_probs)

        # PR-AUC is what we optimize for (the right metric given the class
        # imbalance), but ROC-AUC is printed alongside it because that's
        # the actual metric Kaggle scores submissions on -- worth watching
        # both in case they ever diverge meaningfully.
        print(f"  trial {trial.number}: PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")

        log_run(
            {"run_type": f"tuning_{args.model}", "scale_pos_weight": scale_pos_weight,
             "n_features": len(feature_cols), **params},
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "fraud_precision": precision_score(y_val, val_preds),
                "fraud_recall": recall_score(y_val, val_preds),
            },
        )

        if pr_auc > best_state["pr_auc"]:
            best_state.update(pr_auc=pr_auc, roc_auc=roc_auc, model=model, params=params)

        return pr_auc

    print(f"Running {N_TRIALS} Optuna trials ({args.model})...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest PR-AUC:  {best_state['pr_auc']:.4f}")
    print(f"Best ROC-AUC: {best_state['roc_auc']:.4f}  (same trial as best PR-AUC, not independently optimized)")
    print(f"Best params: {best_state['params']}")

    model_path = f"models/tuned_{args.model}.{model_spec['save_ext']}"
    best_state["model"].booster_.save_model(model_path) if args.model == "lightgbm" \
        else best_state["model"].save_model(model_path)
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()
