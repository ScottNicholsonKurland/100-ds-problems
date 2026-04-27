"""Reference solutions for the NumPy data-manipulation problem set."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def num_to_color(arr: NDArray[np.integer]) -> NDArray[np.str_]:
    """Replace 0 with 'red' and 1 with 'blue'."""
    arr = np.asarray(arr)

    if not np.isin(arr, [0, 1]).all():
        raise ValueError("Input array must contain only 0 and 1.")

    colors = np.array(["red", "blue"])
    return colors[arr]


def compute_part_mean(
    x: NDArray[np.number],
    b: NDArray[np.integer],
) -> dict[int, float]:
    """Compute means of x where b equals 0 and where b equals 1."""
    x = np.asarray(x)
    b = np.asarray(b)

    if x.shape != b.shape:
        raise ValueError("x and b must have the same shape.")

    if not np.isin(b, [0, 1]).all():
        raise ValueError("b must contain only 0 and 1.")

    return {
        0: float(x[b == 0].mean()),
        1: float(x[b == 1].mean()),
    }


def row_or_column_means(
    matrix: NDArray[np.number],
    label: Literal["row", "column"],
) -> NDArray[np.float64]:
    """Return row means or column means for a two-dimensional array."""
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")

    if label == "row":
        return matrix.mean(axis=1)

    if label == "column":
        return matrix.mean(axis=0)

    raise ValueError("label must be either 'row' or 'column'.")


def find_smallest_angle(
    x: NDArray[np.number],
    matrix: NDArray[np.number],
) -> NDArray[np.number]:
    """Return the row in matrix that forms the smallest angle with x."""
    x = np.asarray(x)
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if matrix.shape[1] != x.shape[0]:
        raise ValueError("matrix must have the same number of columns as len(x).")

    x_norm = np.linalg.norm(x)
    row_norms = np.linalg.norm(matrix, axis=1)

    if x_norm == 0 or np.any(row_norms == 0):
        raise ValueError("x and all matrix rows must be nonzero vectors.")

    cosine_similarities = matrix @ x / (row_norms * x_norm)
    return matrix[np.argmax(cosine_similarities)]


def offset_diagonals(n: int) -> NDArray[np.float64]:
    """Create an n-by-n matrix with ones above and below the main diagonal."""
    if n < 1:
        raise ValueError("n must be positive.")

    matrix = np.zeros((n, n))
    indices = np.arange(n - 1)

    matrix[indices, indices + 1] = 1
    matrix[indices + 1, indices] = 1

    return matrix


def cols_with_neg_value(matrix: NDArray[np.number]) -> NDArray[np.number]:
    """Return columns of matrix where at least one entry is negative."""
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")

    return matrix[:, np.any(matrix < 0, axis=0)]


def swap_rows(matrix: NDArray[np.number], i: int, j: int) -> None:
    """Swap rows i and j of matrix in place."""
    matrix[[i, j]] = matrix[[j, i]]


def checkerboard(n: int) -> NDArray[np.int_]:
    """Create an n-by-n checkerboard matrix with 1 in the top-left corner."""
    if n < 1:
        raise ValueError("n must be positive.")

    indices = np.indices((n, n)).sum(axis=0)
    return 1 - (indices % 2)


def can_multiply_in_order(*matrices: NDArray[np.number]) -> bool:
    """Return whether matrices can be multiplied in the supplied order."""
    if len(matrices) < 2:
        return True

    shapes = [np.asarray(matrix).shape for matrix in matrices]

    if any(len(shape) != 2 for shape in shapes):
        raise ValueError("All inputs must be two-dimensional matrices.")

    return all(
        left[1] == right[0] for left, right in zip(shapes, shapes[1:], strict=False)
    )


def cartesian_to_polar(points: NDArray[np.number]) -> NDArray[np.float64]:
    """Convert an array of 2D Cartesian points to polar coordinates."""
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2).")

    x_values = points[:, 0]
    y_values = points[:, 1]

    radii = np.sqrt(x_values**2 + y_values**2)
    angles = np.arctan2(y_values, x_values)

    return np.column_stack((radii, angles))


def project_orthogonal_to_vector(
    matrix: NDArray[np.number],
    vector: NDArray[np.number],
) -> NDArray[np.float64]:
    """Project 3D points onto the plane orthogonal to vector.

    The problem is under-specified because a 3D-to-2D projection requires
    choosing a coordinate basis for the target plane. This implementation
    chooses a deterministic orthonormal basis for that plane.
    """
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(vector, dtype=float)

    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("matrix must have shape (n, 3).")

    if vector.shape != (3,):
        raise ValueError("vector must have shape (3,).")

    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        raise ValueError("vector must be nonzero.")

    normal = vector / vector_norm

    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    basis_1 = helper - np.dot(helper, normal) * normal
    basis_1 = basis_1 / np.linalg.norm(basis_1)
    basis_2 = np.cross(normal, basis_1)

    projected_3d = matrix - np.outer(matrix @ normal, normal)

    return np.column_stack((projected_3d @ basis_1, projected_3d @ basis_2))


def pairwise_distances(
    first_points: NDArray[np.number],
    second_points: NDArray[np.number],
) -> NDArray[np.float64]:
    """Return pairwise Euclidean distances between two point arrays."""
    first_points = np.asarray(first_points, dtype=float)
    second_points = np.asarray(second_points, dtype=float)

    if first_points.ndim != 2 or second_points.ndim != 2:
        raise ValueError("Both inputs must be two-dimensional arrays.")

    if first_points.shape[1] != second_points.shape[1]:
        raise ValueError("Both inputs must have the same number of columns.")

    differences = first_points[:, np.newaxis, :] - second_points[np.newaxis, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))
