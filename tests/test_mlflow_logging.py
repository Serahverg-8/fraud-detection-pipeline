from argparse import Namespace

import mlflow

from src.train import build_run_params, log_run


def test_run_type_is_baseline_when_baseline_flag_set():
    args = Namespace(baseline=True, prune=False)
    params = build_run_params(args, model_params={}, scale_pos_weight=27.46, n_features=425)
    assert params["run_type"] == "baseline"


def test_run_type_is_features_pruned_when_prune_flag_set():
    args = Namespace(baseline=False, prune=True)
    params = build_run_params(args, model_params={}, scale_pos_weight=27.46, n_features=316)
    assert params["run_type"] == "features_pruned"


def test_run_type_is_features_when_neither_flag_set():
    args = Namespace(baseline=False, prune=False)
    params = build_run_params(args, model_params={}, scale_pos_weight=27.46, n_features=429)
    assert params["run_type"] == "features"


def test_build_run_params_includes_model_params_and_counts():
    args = Namespace(baseline=False, prune=False)
    params = build_run_params(
        args, model_params={"max_depth": 6, "n_estimators": 200}, scale_pos_weight=27.46, n_features=429
    )
    assert params["max_depth"] == 6
    assert params["n_estimators"] == 200
    assert params["n_features"] == 429
    assert params["scale_pos_weight"] == 27.46


def test_log_run_records_params_and_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow_test.db")
    mlflow.set_experiment("test-experiment")

    params = {"run_type": "baseline", "max_depth": 6}
    metrics = {"pr_auc": 0.513, "roc_auc": 0.902}

    run_id = log_run(params, metrics)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.params["run_type"] == "baseline"
    assert run.data.params["max_depth"] == "6"
    assert run.data.metrics["pr_auc"] == 0.513
    assert run.data.metrics["roc_auc"] == 0.902
