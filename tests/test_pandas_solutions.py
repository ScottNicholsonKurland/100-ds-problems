import pandas as pd
import pytest

from solutions.pandas_solutions import (
    largest_berry_by_type,
    select_large_blueberries,
    stock_price_day_after_max,
)


def test_select_large_blueberries():
    berries = pd.DataFrame(
        {
            "type": ["blueberry", "blueberry", "raspberry", "blueberry"],
            "size": [0.4, 0.6, 0.8, 0.9],
            "id": [1, 2, 3, 4],
        },
    )

    expected = pd.DataFrame(
        {
            "type": ["blueberry", "blueberry"],
            "size": [0.6, 0.9],
            "id": [2, 4],
        },
        index=[1, 3],
    )

    pd.testing.assert_frame_equal(select_large_blueberries(berries), expected)


def test_select_large_blueberries_rejects_missing_columns():
    berries = pd.DataFrame({"type": ["blueberry"]})

    with pytest.raises(ValueError):
        select_large_blueberries(berries)


def test_largest_berry_by_type():
    berries = pd.DataFrame(
        {
            "type": ["blueberry", "raspberry", "blueberry", "raspberry"],
            "size": [0.4, 0.7, 0.6, 0.5],
            "id": [1, 2, 3, 4],
        },
    )

    expected = pd.DataFrame(
        {
            "type": ["blueberry", "raspberry"],
            "size": [0.6, 0.7],
            "id": [3, 2],
        },
    )

    pd.testing.assert_frame_equal(largest_berry_by_type(berries), expected)


def test_largest_berry_by_type_empty_dataframe():
    berries = pd.DataFrame({"type": [], "size": []})

    pd.testing.assert_frame_equal(largest_berry_by_type(berries), berries)


def test_stock_price_day_after_max():
    stocks = pd.DataFrame(
        {
            "stock": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "day": [1, 2, 3, 1, 2, 3, 1, 2],
            "price": [10, 20, 15, 5, 7, 9, 100, 90],
        },
    )

    expected = pd.DataFrame(
        {
            "stock": ["A", "C"],
            "day": [3, 2],
            "price": [15, 90],
        },
    )

    pd.testing.assert_frame_equal(stock_price_day_after_max(stocks), expected)


def test_stock_price_day_after_max_handles_unsorted_input():
    stocks = pd.DataFrame(
        {
            "stock": ["A", "A", "A"],
            "day": [3, 1, 2],
            "price": [15, 10, 20],
        },
    )

    expected = pd.DataFrame(
        {
            "stock": ["A"],
            "day": [3],
            "price": [15],
        },
    )

    pd.testing.assert_frame_equal(stock_price_day_after_max(stocks), expected)


def test_stock_price_day_after_max_returns_empty_if_max_is_final_observation():
    stocks = pd.DataFrame(
        {
            "stock": ["A", "A"],
            "day": [1, 2],
            "price": [10, 20],
        },
    )

    result = stock_price_day_after_max(stocks)

    assert result.empty
    assert list(result.columns) == ["stock", "day", "price"]
