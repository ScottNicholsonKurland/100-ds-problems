"""Reference solutions for the plotting problem set."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray


def plot_linear_model_sample(
    x: ArrayLike,
    y: ArrayLike,
    intercept: float,
    slope: float,
) -> tuple[Figure, Axes]:
    """Plot sampled linear-model data with the true line overlaid.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib figure and axes objects.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape.")

    fig, ax = plt.subplots()

    ax.scatter(x_array, y_array, label="Sampled data")

    line_x = np.linspace(x_array.min(), x_array.max(), 100)
    line_y = intercept + slope * line_x
    ax.plot(line_x, line_y, label="True model")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sampled Data from a Simple Linear Model")
    ax.legend()

    return fig, ax


def plot_coordinate_grid(
    n_rows: int = 2,
    n_cols: int = 3,
) -> tuple[Figure, NDArray[np.object_]]:
    """Create a grid of plots labeled by their row and column coordinates."""
    if n_rows < 1:
        raise ValueError("n_rows must be positive.")

    if n_cols < 1:
        raise ValueError("n_cols must be positive.")

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)

    for row_index in range(n_rows):
        for col_index in range(n_cols):
            ax = axes[row_index, col_index]
            ax.text(
                0.5,
                0.5,
                f"({row_index}, {col_index})",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Plot Coordinate Grid")
    fig.tight_layout()

    return fig, axes


def plot_vectors(
    vector_pairs: Sequence[Sequence[Sequence[float]]],
) -> tuple[Figure, Axes]:
    """Plot vectors as arrows from starting coordinates to ending coordinates.

    Parameters
    ----------
    vector_pairs:
        Sequence shaped like ``(n_vectors, 2, 2)``. Each vector is represented as
        ``[[x_start, y_start], [x_end, y_end]]``.
    """
    vectors = np.asarray(vector_pairs, dtype=float)

    if vectors.ndim != 3 or vectors.shape[1:] != (2, 2):
        raise ValueError("vector_pairs must have shape (n_vectors, 2, 2).")

    starts = vectors[:, 0, :]
    ends = vectors[:, 1, :]
    deltas = ends - starts

    fig, ax = plt.subplots()

    for start, delta in zip(starts, deltas, strict=True):
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            length_includes_head=True,
            head_width=0.08,
        )

    all_points = vectors.reshape(-1, 2)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    x_margin = max(0.5, 0.1 * (x_max - x_min))
    y_margin = max(0.5, 0.1 * (y_max - y_min))

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Vector Plot")

    return fig, ax
