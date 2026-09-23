import pandas as pd

from src.predict import normalize_identity_columns


def test_converts_hyphenated_id_columns_to_underscores():
    df = pd.DataFrame(columns=["TransactionID", "id-01", "id-02", "DeviceType"])

    result = normalize_identity_columns(df)

    assert list(result.columns) == ["TransactionID", "id_01", "id_02", "DeviceType"]


def test_leaves_already_underscored_columns_unchanged():
    df = pd.DataFrame(columns=["TransactionID", "id_01", "id_02", "DeviceType"])

    result = normalize_identity_columns(df)

    assert list(result.columns) == ["TransactionID", "id_01", "id_02", "DeviceType"]
