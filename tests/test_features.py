import pandas as pd

from src.features import add_card_aggregate_features


def _txns(rows):
    """rows: list of (TransactionID, card1, TransactionDT, TransactionAmt)."""
    return pd.DataFrame(
        rows, columns=["TransactionID", "card1", "TransactionDT", "TransactionAmt"]
    )


def test_first_transaction_for_a_card_gets_default_values():
    df = _txns([(1, "A", 0, 100.0)])

    result = add_card_aggregate_features(df)

    row = result.iloc[0]
    assert row["card1_count_expanding"] == 0
    assert pd.isna(row["card1_amt_mean_expanding"])
    assert pd.isna(row["card1_amt_std_expanding"])
    assert row["card1_count_24h"] == 0
    assert pd.isna(row["card1_amt_mean_24h"])
    assert pd.isna(row["amt_to_card1_mean_ratio"])


def test_expanding_count_increments_per_card_in_time_order():
    df = _txns([
        (1, "A", 0, 10.0),
        (2, "A", 100, 20.0),
        (3, "A", 200, 30.0),
    ])

    result = add_card_aggregate_features(df).sort_values("TransactionID")

    assert result["card1_count_expanding"].tolist() == [0, 1, 2]


def test_expanding_mean_only_uses_prior_transactions_not_current():
    df = _txns([
        (1, "A", 0, 10.0),
        (2, "A", 100, 20.0),
        (3, "A", 200, 30.0),
    ])

    result = add_card_aggregate_features(df).sort_values("TransactionID")

    # Row 3's expanding mean should be the average of rows 1 and 2 only (15.0),
    # never including its own amount (30.0).
    assert result.iloc[2]["card1_amt_mean_expanding"] == 15.0


def test_later_transaction_does_not_leak_into_earlier_transactions_features():
    df = _txns([
        (1, "A", 0, 10.0),
        (2, "A", 100, 20.0),
        (3, "A", 200, 30.0),
    ])
    baseline = add_card_aggregate_features(df).sort_values("TransactionID").reset_index(drop=True)

    df_mutated = df.copy()
    df_mutated.loc[df_mutated["TransactionID"] == 3, "TransactionAmt"] = 99999.0
    mutated = add_card_aggregate_features(df_mutated).sort_values("TransactionID").reset_index(drop=True)

    # Rows 1 and 2 come before row 3 in time, so changing row 3's amount
    # must not change their computed features.
    for col in [
        "card1_count_expanding", "card1_amt_mean_expanding", "card1_amt_std_expanding",
        "card1_count_24h", "card1_amt_mean_24h", "amt_to_card1_mean_ratio",
    ]:
        pd.testing.assert_series_equal(
            baseline.loc[:1, col], mutated.loc[:1, col], check_names=False
        )


def test_24h_window_excludes_transactions_older_than_24h():
    one_day = 24 * 60 * 60
    df = _txns([
        (1, "A", 0, 10.0),
        (2, "A", one_day + 1, 20.0),  # just over 24h after row 1
    ])

    result = add_card_aggregate_features(df).sort_values("TransactionID")

    # Row 2's 24h window should NOT include row 1 (it's outside the trailing 24h).
    assert result.iloc[1]["card1_count_24h"] == 0
    assert pd.isna(result.iloc[1]["card1_amt_mean_24h"])


def test_24h_window_includes_transactions_within_24h():
    df = _txns([
        (1, "A", 0, 10.0),
        (2, "A", 3600, 20.0),  # 1h after row 1
    ])

    result = add_card_aggregate_features(df).sort_values("TransactionID")

    assert result.iloc[1]["card1_count_24h"] == 1
    assert result.iloc[1]["card1_amt_mean_24h"] == 10.0


def test_different_cards_do_not_share_history():
    df = _txns([
        (1, "A", 0, 10.0),
        (2, "B", 100, 999.0),
        (3, "A", 200, 30.0),
    ])

    result = add_card_aggregate_features(df).sort_values("TransactionID")

    # Card A's third row should only see card A's history (row 1), not card B's.
    row3 = result[result["TransactionID"] == 3].iloc[0]
    assert row3["card1_count_expanding"] == 1
    assert row3["card1_amt_mean_expanding"] == 10.0
