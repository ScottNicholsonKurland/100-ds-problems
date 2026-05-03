import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from solutions.plotting import (
    plot_coordinate_grid,
    plot_linear_model_sample,
    plot_vectors,
)


def test_plot_linear_model_sample_returns_figure_and_axes():
    x = np.array([0, 1, 2, 3], dtype=float)
    y = np.array([1, 3, 5, 7], dtype=float)

    fig, ax = plot_linear_model_sample(
        x=x,
        y=y,
        intercept=1,
        slope=2,
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1

    plt.close(fig)


def test_plot_linear_model_sample_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        plot_linear_model_sample(
            x=np.array([0, 1, 2]),
            y=np.array([0, 1]),
            intercept=0,
            slope=1,
        )


def test_plot_coordinate_grid_returns_expected_shape_and_labels():
    fig, axes = plot_coordinate_grid(n_rows=2, n_cols=3)

    assert isinstance(fig, Figure)
    assert axes.shape == (2, 3)

    labels = [
        text.get_text()
        for ax in axes.ravel()
        for text in ax.texts
    ]

    assert labels == [
        "(0, 0)",
        "(0, 1)",
        "(0, 2)",
        "(1, 0)",
        "(1, 1)",
        "(1, 2)",
    ]

    plt.close(fig)


def test_plot_coordinate_grid_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError):
        plot_coordinate_grid(n_rows=0, n_cols=3)

    with pytest.raises(ValueError):
        plot_coordinate_grid(n_rows=2, n_cols=0)


def test_plot_vectors_returns_figure_and_axes():
    vector_pairs = [
        [(0, 1), (1, 0)],
        [(1, 1), (2, 2)],
        [(-1, 0), (0, -1)],
    ]

    fig, ax = plot_vectors(vector_pairs)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert len(ax.patches) == 3

    plt.close(fig)


def test_plot_vectors_rejects_bad_shape():
    with pytest.raises(ValueError):
        plot_vectors([(0, 1), (1, 0)])
