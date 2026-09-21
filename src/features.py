"""
Feature engineering — Milestone 2.

Adds per-card group aggregation features to the transaction table. Every
feature is computed using only transactions that happened strictly BEFORE
the current one (never the current row itself, never a future row) — this
is what makes them safe to use without leaking information a real-time
scoring system wouldn't have had at the time of the transaction.
"""

import numpy as np
import pandas as pd

GROUP_COL = "card1"
AMT_COL = "TransactionAmt"
TIME_COL = "TransactionDT"
WINDOW_SECONDS = 24 * 60 * 60


def add_card_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with 6 leakage-safe per-card aggregate features added:

    - card1_count_expanding, card1_amt_mean_expanding, card1_amt_std_expanding:
      stats over ALL prior transactions for that card.
    - card1_count_24h, card1_amt_mean_24h: stats over the trailing 24h only.
    - amt_to_card1_mean_ratio: current amount / card1_amt_mean_expanding.

    Rows are returned in the same order as the input. A card's first
    transaction (no prior history) gets 0 for count features and NaN for
    mean/std/ratio features — XGBoost handles NaN natively, so these are
    left as-is rather than imputed.
    """
    original_index = df.index
    sorted_df = df.sort_values(TIME_COL)
    grouped = sorted_df.groupby(GROUP_COL)[AMT_COL]

    # Expanding stats: shift(1) so the current row's own amount is excluded —
    # expanding() alone would include the current row, which is the leak.
    shifted = grouped.shift(1)
    expanding_by_card = shifted.groupby(sorted_df[GROUP_COL])

    count_expanding = expanding_by_card.cumcount()
    # cumcount() counts the shifted (i.e. prior-only) rows seen so far per
    # group, which is exactly the count of prior transactions.

    # groupby(...).expanding()/.rolling() return results concatenated in
    # group order, not original row order — reindex back to sorted_df's
    # row order (a label-based lookup, safe regardless of internal order)
    # before using them.
    mean_expanding = (
        expanding_by_card.expanding().mean().droplevel(0).reindex(sorted_df.index)
    )
    std_expanding = (
        expanding_by_card.expanding().std().droplevel(0).reindex(sorted_df.index)
    )

    # 24h trailing window, per card. Computed via groupby+apply (rather than
    # groupby(...).rolling(...) directly) so each group's result keeps its
    # original row labels, making the final reindex unambiguous even if two
    # transactions share the same TransactionDT.
    def _rolling_24h(group: pd.DataFrame) -> pd.DataFrame:
        time_idx = pd.to_timedelta(group[TIME_COL], unit="s")
        s = group[AMT_COL].set_axis(time_idx)
        window = f"{WINDOW_SECONDS}s"
        # count uses min_periods=0 so "no prior transactions" reads as 0,
        # not NaN. mean keeps the default (NaN with no prior transactions —
        # there's no sensible average of zero values).
        count = s.rolling(window, closed="left", min_periods=0).count()
        mean = s.rolling(window, closed="left").mean()
        return pd.DataFrame(
            {"count_24h": count.to_numpy(), "mean_24h": mean.to_numpy()},
            index=group.index,
        )

    window_stats = sorted_df.groupby(GROUP_COL, group_keys=False)[
        [TIME_COL, AMT_COL]
    ].apply(_rolling_24h)
    window_stats = window_stats.reindex(sorted_df.index)

    result = sorted_df.copy()
    result["card1_count_expanding"] = count_expanding
    result["card1_amt_mean_expanding"] = mean_expanding
    result["card1_amt_std_expanding"] = std_expanding
    result["card1_count_24h"] = window_stats["count_24h"]
    result["card1_amt_mean_24h"] = window_stats["mean_24h"]
    result["amt_to_card1_mean_ratio"] = (
        result[AMT_COL] / result["card1_amt_mean_expanding"]
    )

    return result.loc[original_index]
