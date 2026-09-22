import pandas as pd

from src.tune_tabm import identify_categorical_columns


def test_identifies_object_dtype_columns_among_feature_cols():
    raw_df = pd.DataFrame({
        "TransactionAmt": [1.0, 2.0],
        "ProductCD": ["W", "H"],
        "card1": [111, 222],
        "P_emaildomain": ["gmail.com", None],
    })
    feature_cols = ["TransactionAmt", "ProductCD", "card1", "P_emaildomain"]

    result = identify_categorical_columns(raw_df, feature_cols)

    assert set(result) == {"ProductCD", "P_emaildomain"}


def test_ignores_object_columns_not_in_feature_cols():
    # e.g. columns dropped during pruning shouldn't show up even if they
    # were categorical in the raw data.
    raw_df = pd.DataFrame({
        "TransactionAmt": [1.0, 2.0],
        "DroppedCategoricalCol": ["a", "b"],
    })
    feature_cols = ["TransactionAmt"]

    result = identify_categorical_columns(raw_df, feature_cols)

    assert result == []
