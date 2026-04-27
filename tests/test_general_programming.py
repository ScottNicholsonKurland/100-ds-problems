from pathlib import Path

import pytest

from solutions.general_programming import (
    dict_to_list,
    differentiate_polynomial,
    either_vowel_flags,
    get_valid_passwords,
    group_words_by_first_letter,
    list_to_dict,
    merge_overlapping_sets,
    merge_word_files,
    polynomial_to_string,
    simplify_polynomial,
    solve_sudoku,
    split_into_words,
    transpose,
)


def test_dict_to_list():
    data = {"a": [1, 2, 3], "b": [3, 2, 1]}

    assert dict_to_list(data) == [
        {"a": 1, "b": 3},
        {"a": 2, "b": 2},
        {"a": 3, "b": 1},
    ]


def test_dict_to_list_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        dict_to_list({"a": [1, 2], "b": [1]})


def test_list_to_dict():
    rows = [{"a": 1, "b": 3}, {"a": 2, "b": 2}, {"a": 3, "b": 1}]

    assert list_to_dict(rows) == {"a": [1, 2, 3], "b": [3, 2, 1]}


def test_list_to_dict_empty_input():
    assert list_to_dict([]) == {}


def test_list_to_dict_rejects_inconsistent_keys():
    with pytest.raises(ValueError):
        list_to_dict([{"a": 1, "b": 2}, {"a": 3}])


def test_either_vowel_flags():
    assert either_vowel_flags(
        ["a", "b", "c", "d", "e"],
        ["v", "w", "x", "y", "z"],
    ) == [True, False, False, False, True]


def test_either_vowel_flags_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        either_vowel_flags(["a"], ["b", "c"])


def test_group_words_by_first_letter():
    text = "a special string bearing an important salutation"

    assert group_words_by_first_letter(text) == {
        "a": ["a", "an"],
        "b": ["bearing"],
        "i": ["important"],
        "s": ["special", "string", "salutation"],
    }


def test_merge_word_files(tmp_path: Path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    output_file = tmp_path / "output.txt"

    first_file.write_text("This\nis\na\nfile\nof\nwords\n", encoding="utf-8")
    second_file.write_text("And\nanother\nfile\n", encoding="utf-8")

    merge_word_files(first_file, second_file, output_file)

    assert output_file.read_text(encoding="utf-8") == (
        "And, This\n"
        "another, is\n"
        "a, file\n"
        "file\n"
        "of\n"
        "words\n"
    )


def test_transpose():
    assert transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
    ]


def test_transpose_rejects_ragged_matrix():
    with pytest.raises(ValueError):
        transpose([[1, 2], [3]])


def test_get_valid_passwords():
    possible_passwords = [
        "moshi",
        "m0shi",
        "mosh!",
        "m0sh!",
        "^^oshi",
        "^^0shi",
        "^^0sh!",
    ]

    assert get_valid_passwords(possible_passwords) == [
        "m0sh!",
        "^^0shi",
        "^^0sh!",
    ]


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ([1, 1, 1], "x^2 + x + 1"),
        ([2, -1, -2], "2x^2 - x - 2"),
        ([0, 9, -10], "9x - 10"),
        ([0, 0, 0], "0"),
        ([-1, 0, 1], "-x^2 + 1"),
        ([3], "3"),
    ],
)
def test_polynomial_to_string(coefficients, expected):
    assert polynomial_to_string(coefficients) == expected


def test_simplify_polynomial():
    assert simplify_polynomial([(1, 0), (1, 1), (1, 2)]) == [
        (1, 0),
        (1, 1),
        (1, 2),
    ]

    assert simplify_polynomial([(2, 0), (-2, 1), (2, 2), (-2, 2)]) == [
        (2, 0),
        (-2, 1),
    ]

    assert simplify_polynomial([(1, 0), (1, 1), (1, 0), (1, 1)]) == [
        (2, 0),
        (2, 1),
    ]


def test_simplify_polynomial_drops_zero_coefficients():
    assert simplify_polynomial([(2, 2), (-2, 2), (5, 0)]) == [(5, 0)]


def test_differentiate_polynomial():
    assert differentiate_polynomial([(1, 0), (1, 1), (1, 2)]) == [
        (1, 0),
        (2, 1),
    ]

    assert differentiate_polynomial([(2, 0), (-2, 1), (2, 2), (-2, 2)]) == [
        (-2, 0),
    ]

    assert differentiate_polynomial([(1, 0), (1, 1), (1, 0), (1, 1)]) == [
        (2, 0),
    ]


def test_split_into_words():
    language = {
        "number",
        "numbers",
        "ship",
        "ships",
        "hip",
        "hips",
        "swear",
        "wear",
    }

    assert split_into_words("number", language) == ["number"]

    assert split_into_words("numbership", language) == [
        "number ship",
        "numbers hip",
    ]

    assert split_into_words("numbershipswear", language) == [
        "number ship swear",
        "number ships wear",
        "numbers hip swear",
        "numbers hips wear",
    ]


def test_solve_sudoku():
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    expected = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]

    assert solve_sudoku(puzzle) == expected


def test_merge_overlapping_sets():
    original_sets = [{1, 2}, {3}, {2, 4}, {4, 6, 7}, {8}, {3, 9}]
    merged = merge_overlapping_sets(original_sets)

    assert {frozenset(group) for group in merged} == {
        frozenset({1, 2, 4, 6, 7}),
        frozenset({3, 9}),
        frozenset({8}),
    }

    assert original_sets == [{1, 2}, {3}, {2, 4}, {4, 6, 7}, {8}, {3, 9}]
