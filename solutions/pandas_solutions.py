"""Reference solutions for the pandas data-manipulation problem set."""

from __future__ import annotations

import pandas as pd


def _require_columns(dataframe: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise ValueError if dataframe is missing required columns."""
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def select_large_blueberries(
    berries: pd.DataFrame,
    minimum_size: float = 0.5,
) -> pd.DataFrame:
    """Select blueberries larger than minimum_size centimeters."""
    _require_columns(berries, {"type", "size"})

    mask = (berries["type"] == "blueberry") & (berries["size"] > minimum_size)

    return berries.loc[mask].copy()


def largest_berry_by_type(berries: pd.DataFrame) -> pd.DataFrame:
    """Return the largest berry row for each berry type.

    If there is a tie within a type, the first largest row in the original
    DataFrame order is returned.
    """
    _require_columns(berries, {"type", "size"})

    if berries.empty:
        return berries.copy()

    largest_indices = berries.groupby("type", sort=False)["size"].idxmax()

    return berries.loc[largest_indices].reset_index(drop=True)


def stock_price_day_after_max(stocks: pd.DataFrame) -> pd.DataFrame:
    """Return each stock's row on the observation after its maximum price.

    The "day after" is interpreted as the next recorded observation after the
    stock reaches its maximum price when rows are sorted by stock and day.

    Stocks whose maximum price occurs on their final recorded day are excluded.
    """
    _require_columns(stocks, {"stock", "day", "price"})

    sorted_stocks = stocks.sort_values(
        ["stock", "day"],
        kind="mergesort",
    ).reset_index(drop=True)

    selected_rows = []

    for _, group in sorted_stocks.groupby("stock", sort=False):
        max_position = group["price"].to_numpy().argmax()
        next_position = max_position + 1

        if next_position < len(group):
            selected_rows.append(group.iloc[next_position])

    if not selected_rows:
        return sorted_stocks.iloc[0:0].copy()

    return pd.DataFrame(selected_rows).reset_index(drop=True)
