import optuna
import pytest

from src.tune import suggest_catboost_params, suggest_lgbm_params, suggest_xgb_params


def test_suggest_xgb_params_returns_expected_keys():
    trial = optuna.trial.FixedTrial({
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    })

    params = suggest_xgb_params(trial)

    assert params == {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }


def test_suggest_xgb_params_warns_on_out_of_range_value():
    # FixedTrial warns when a fixed value falls outside the distribution
    # passed to suggest_* -- this is really a test that our search space
    # bounds are what we think they are (catches an accidental typo in the
    # range, e.g. max_depth=60 instead of 6).
    trial = optuna.trial.FixedTrial({
        "max_depth": 999,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    })

    with pytest.warns(UserWarning, match="max_depth"):
        suggest_xgb_params(trial)


def test_suggest_lgbm_params_returns_expected_keys():
    trial = optuna.trial.FixedTrial({
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    })

    params = suggest_lgbm_params(trial)

    assert params == {
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }


def test_suggest_catboost_params_returns_expected_keys():
    trial = optuna.trial.FixedTrial({
        "depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "l2_leaf_reg": 5.0,
        "subsample": 0.8,
    })

    params = suggest_catboost_params(trial)

    assert params == {
        "depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "l2_leaf_reg": 5.0,
        "subsample": 0.8,
    }
