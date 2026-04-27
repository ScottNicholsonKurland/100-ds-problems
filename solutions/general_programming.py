"""Reference solutions for the General Programming problem set.

The functions in this module favor readable implementations and explicit edge
case handling over excessive cleverness.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

VOWELS = set("aeiouAEIOU")
PASSWORD_SYMBOLS = set("^!#$?-")


def dict_to_list(data: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Convert a dictionary of equal-length sequences into row dictionaries."""
    lengths = {len(values) for values in data.values()}
    if len(lengths) > 1:
        raise ValueError("All values must have the same length.")

    keys = list(data)
    return [
        dict(zip(keys, row_values, strict=True))
        for row_values in zip(*data.values(), strict=True)
    ]


def list_to_dict(rows: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert a list of row dictionaries into a dictionary of columns."""
    if not rows:
        return {}

    expected_keys = set(rows[0])
    output: dict[str, list[Any]] = {key: [] for key in rows[0]}

    for row in rows:
        if set(row) != expected_keys:
            raise ValueError("All dictionaries must contain the same keys.")
        for key, value in row.items():
            output[key].append(value)

    return output


def either_vowel_flags(left: Sequence[str], right: Sequence[str]) -> list[bool]:
    """Return True where either same-index character is a vowel."""
    if len(left) != len(right):
        raise ValueError("Input sequences must have the same length.")

    return [(a in VOWELS) or (b in VOWELS) for a, b in zip(left, right, strict=True)]


def group_words_by_first_letter(text: str) -> dict[str, list[str]]:
    """Group words from a string by their first letter."""
    grouped: dict[str, list[str]] = defaultdict(list)

    for word in text.split():
        grouped[word[0]].append(word)

    return dict(sorted(grouped.items()))


def merge_word_files(
    first_input: str | Path,
    second_input: str | Path,
    output_file: str | Path,
) -> None:
    """Merge two one-word-per-line files into alphabetized comma-paired lines.

    If one file is longer, the remaining single words are written one per line.
    """
    first_words = Path(first_input).read_text(encoding="utf-8").splitlines()
    second_words = Path(second_input).read_text(encoding="utf-8").splitlines()

    max_length = max(len(first_words), len(second_words))
    output_lines: list[str] = []

    for index in range(max_length):
        pair = []
        if index < len(first_words):
            pair.append(first_words[index])
        if index < len(second_words):
            pair.append(second_words[index])
        output_lines.append(", ".join(sorted(pair, key=str.casefold)))

    Path(output_file).write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def transpose(matrix: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """Transpose a rectangular list-of-lists matrix."""
    if not matrix:
        return []

    row_lengths = {len(row) for row in matrix}
    if len(row_lengths) > 1:
        raise ValueError("Matrix must be rectangular.")

    return [list(column) for column in zip(*matrix, strict=True)]


def get_valid_passwords(passwords: Iterable[str]) -> list[str]:
    """Return passwords containing at least one digit and one required symbol."""
    return [
        password
        for password in passwords
        if any(character.isdigit() for character in password)
        and any(character in PASSWORD_SYMBOLS for character in password)
    ]


def polynomial_to_string(coefficients: Sequence[int | float]) -> str:
    """Convert coefficients ordered high-to-low into a readable polynomial."""
    terms: list[tuple[str, str]] = []
    degree = len(coefficients) - 1

    for offset, coefficient in enumerate(coefficients):
        power = degree - offset

        if coefficient == 0:
            continue

        sign = "-" if coefficient < 0 else "+"
        absolute = abs(coefficient)

        if power == 0:
            body = f"{absolute:g}"
        elif power == 1:
            body = "x" if absolute == 1 else f"{absolute:g}x"
        else:
            body = f"x^{power}" if absolute == 1 else f"{absolute:g}x^{power}"

        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    output = f"-{first_body}" if first_sign == "-" else first_body

    for sign, body in terms[1:]:
        output += f" {sign} {body}"

    return output


def simplify_polynomial(
    polynomial: Iterable[tuple[int | float, int]],
) -> list[tuple[int | float, int]]:
    """Combine polynomial terms with like degree and drop zero coefficients.

    Polynomials are represented as ``(coefficient, degree)`` tuples.
    """
    terms: dict[int, int | float] = defaultdict(int)

    for coefficient, degree in polynomial:
        terms[degree] += coefficient

    return [
        (coefficient, degree)
        for degree, coefficient in sorted(terms.items())
        if coefficient != 0
    ]


def differentiate_polynomial(
    polynomial: Iterable[tuple[int | float, int]],
) -> list[tuple[int | float, int]]:
    """Differentiate and simplify a polynomial represented as term tuples."""
    derivative = [
        (coefficient * degree, degree - 1)
        for coefficient, degree in polynomial
        if degree > 0 and coefficient != 0
    ]

    return simplify_polynomial(derivative)


def split_into_words(text: str, language: set[str]) -> list[str]:
    """Return all possible segmentations of text into words from language."""
    memo: dict[str, list[list[str]]] = {"": [[]]}

    def segment(suffix: str) -> list[list[str]]:
        if suffix in memo:
            return memo[suffix]

        matches: list[list[str]] = []

        for word in sorted(language):
            if suffix.startswith(word):
                for rest in segment(suffix[len(word) :]):
                    matches.append([word, *rest])

        memo[suffix] = matches
        return matches

    return [" ".join(words) for words in segment(text)]


def solve_sudoku(grid: Sequence[Sequence[int]]) -> list[list[int]] | None:
    """Solve a 9x9 Sudoku puzzle with backtracking.

    Empty cells should be represented with 0. Returns a solved grid, or None if
    no solution exists.
    """
    board = [list(row) for row in grid]

    if len(board) != 9 or any(len(row) != 9 for row in board):
        raise ValueError("Sudoku grid must be 9x9.")

    def candidates(row: int, column: int) -> set[int]:
        used = set(board[row])
        used.update(board[r][column] for r in range(9))

        box_row = 3 * (row // 3)
        box_column = 3 * (column // 3)
        used.update(
            board[r][c]
            for r in range(box_row, box_row + 3)
            for c in range(box_column, box_column + 3)
        )

        return set(range(1, 10)) - used

    def find_empty_cell() -> tuple[int, int] | None:
        best_cell = None
        best_count = 10

        for row in range(9):
            for column in range(9):
                if board[row][column] == 0:
                    count = len(candidates(row, column))
                    if count < best_count:
                        best_cell = (row, column)
                        best_count = count

        return best_cell

    def backtrack() -> bool:
        cell = find_empty_cell()

        if cell is None:
            return True

        row, column = cell

        for value in sorted(candidates(row, column)):
            board[row][column] = value
            if backtrack():
                return True
            board[row][column] = 0

        return False

    return board if backtrack() else None


def merge_overlapping_sets(sets_: Iterable[set[Any]]) -> list[set[Any]]:
    """Merge sets sharing elements until all resulting sets are disjoint.

    The original set objects are not modified.
    """
    groups = [set(group) for group in sets_]
    changed = True

    while changed:
        changed = False
        merged_groups: list[set[Any]] = []

        while groups:
            current = groups.pop(0)
            overlapping = [group for group in groups if current & group]
            non_overlapping = [group for group in groups if not current & group]

            if overlapping:
                for group in overlapping:
                    current |= group
                changed = True

            merged_groups.append(current)
            groups = non_overlapping

        groups = merged_groups

    return groups
