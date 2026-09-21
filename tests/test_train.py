import json

import pandas as pd

from src.train import drop_pruned_features


def test_drops_columns_listed_in_pruned_features_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    dropped_path = tmp_path / "dropped_features.json"
    dropped_path.write_text(json.dumps(["b"]))

    result = drop_pruned_features(df, dropped_path=str(dropped_path))

    assert list(result.columns) == ["a", "c"]


def test_returns_df_unchanged_when_dropped_features_file_missing(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    missing_path = tmp_path / "does_not_exist.json"

    result = drop_pruned_features(df, dropped_path=str(missing_path))

    assert list(result.columns) == ["a", "b"]


def test_ignores_listed_columns_that_are_not_present(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    dropped_path = tmp_path / "dropped_features.json"
    dropped_path.write_text(json.dumps(["b", "not_a_real_column"]))

    result = drop_pruned_features(df, dropped_path=str(dropped_path))

    assert list(result.columns) == ["a"]
