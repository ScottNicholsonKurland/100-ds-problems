import numpy as np
import pytest

from solutions.data_manipulation import (
    can_multiply_in_order,
    cartesian_to_polar,
    checkerboard,
    cols_with_neg_value,
    compute_part_mean,
    find_smallest_angle,
    num_to_color,
    offset_diagonals,
    pairwise_distances,
    project_orthogonal_to_vector,
    row_or_column_means,
    swap_rows,
)


def test_num_to_color():
    arr = np.array([0, 0, 1, 0, 1])

    expected = np.array(["red", "red", "blue", "red", "blue"])

    np.testing.assert_array_equal(num_to_color(arr), expected)


def test_num_to_color_rejects_invalid_values():
    with pytest.raises(ValueError):
        num_to_color(np.array([0, 1, 2]))


def test_compute_part_mean():
    x = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 1, 0, 0, 1])

    assert compute_part_mean(x, b) == {
        0: 3.5,
        1: pytest.approx(2.6666666666666665),
    }


def test_compute_part_mean_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compute_part_mean(np.array([1, 2]), np.array([0]))


def test_row_or_column_means():
    matrix = np.array([[0, 1], [2, 1]])

    np.testing.assert_allclose(row_or_column_means(matrix, "row"), np.array([0.5, 1.5]))
    np.testing.assert_allclose(
        row_or_column_means(matrix, "column"),
        np.array([1.0, 1.0]),
    )


def test_row_or_column_means_rejects_bad_label():
    matrix = np.array([[0, 1], [2, 1]])

    with pytest.raises(ValueError):
        row_or_column_means(matrix, "diagonal")


def test_find_smallest_angle():
    x = np.array([1, 0])
    matrix = np.array([[0, 1], [1, 1], [2, 0]])

    np.testing.assert_array_equal(find_smallest_angle(x, matrix), np.array([2, 0]))


def test_find_smallest_angle_rejects_zero_vector():
    with pytest.raises(ValueError):
        find_smallest_angle(np.array([0, 0]), np.array([[1, 0]]))


def test_offset_diagonals():
    expected = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=float,
    )

    np.testing.assert_array_equal(offset_diagonals(5), expected)


def test_cols_with_neg_value():
    matrix = np.array(
        [
            [1, -2, 3],
            [4, 5, -6],
            [7, 8, 9],
        ],
    )

    expected = np.array(
        [
            [-2, 3],
            [5, -6],
            [8, 9],
        ],
    )

    np.testing.assert_array_equal(cols_with_neg_value(matrix), expected)


def test_swap_rows():
    matrix = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ],
    )

    swap_rows(matrix, 0, 2)

    expected = np.array(
        [
            [5, 6],
            [3, 4],
            [1, 2],
        ],
    )

    np.testing.assert_array_equal(matrix, expected)


def test_checkerboard():
    expected = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ],
    )

    np.testing.assert_array_equal(checkerboard(5), expected)


def test_can_multiply_in_order():
    a = np.zeros((2, 3))
    b = np.zeros((3, 4))
    c = np.zeros((4, 5))
    d = np.zeros((6, 2))

    assert can_multiply_in_order(a, b, c)
    assert not can_multiply_in_order(a, d, b)


def test_cartesian_to_polar():
    points = np.array([[0, 1], [1, 0], [1, 1]])

    expected = np.array(
        [
            [1, np.pi / 2],
            [1, 0],
            [np.sqrt(2), np.pi / 4],
        ],
    )

    np.testing.assert_allclose(cartesian_to_polar(points), expected)


def test_project_orthogonal_to_vector_for_xy_plane():
    points = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
    )
    vector = np.array([0, 0, 1])

    expected = np.array(
        [
            [1, 2],
            [4, 5],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(project_orthogonal_to_vector(points, vector), expected)


def test_project_orthogonal_to_vector_rejects_zero_vector():
    with pytest.raises(ValueError):
        project_orthogonal_to_vector(np.array([[1, 2, 3]]), np.array([0, 0, 0]))


def test_pairwise_distances():
    first_points = np.array([[0, 0], [3, 4]])
    second_points = np.array([[0, 0], [6, 8]])

    expected = np.array(
        [
            [0, 10],
            [5, 5],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(pairwise_distances(first_points, second_points), expected)


def test_pairwise_distances_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        pairwise_distances(np.array([[0, 0]]), np.array([[0, 0, 0]]))
