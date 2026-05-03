from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from solutions.sql_queries import (
    CURRENT_CHECKOUT_USERS_SQL,
    OVERDUE_USERS_SQL,
    POSSIBLY_STOLEN_BOOKS_SUMMARY_SQL,
    POSSIBLY_STOLEN_USER_BOOK_PAIRS_SQL,
    RECENT_CHECKOUT_USERS_SQL,
)


@pytest.fixture
def connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.executescript(
        """
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            join_date TEXT NOT NULL,
            branch_id INTEGER NOT NULL,
            name TEXT NOT NULL
        );

        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            author_id INTEGER NOT NULL,
            genre_id INTEGER NOT NULL,
            publish_date TEXT NOT NULL,
            name TEXT NOT NULL
        );

        CREATE TABLE checkouts (
            user_id INTEGER NOT NULL,
            book_id INTEGER NOT NULL,
            checkout_time TEXT NOT NULL,
            return_time TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (book_id) REFERENCES books (book_id)
        );
        """,
    )

    conn.executemany(
        """
        INSERT INTO users (user_id, join_date, branch_id, name)
        VALUES (?, ?, ?, ?);
        """,
        [
            (1, "2025-01-10", 10, "Alice"),
            (2, "2025-02-15", 10, "Bob"),
            (3, "2025-03-20", 20, "Chen"),
            (4, "2025-04-25", 20, "Diana"),
        ],
    )

    conn.executemany(
        """
        INSERT INTO books (book_id, author_id, genre_id, publish_date, name)
        VALUES (?, ?, ?, ?, ?);
        """,
        [
            (101, 1001, 501, "2020-01-01", "SQL Basics"),
            (102, 1002, 502, "2021-01-01", "Python Patterns"),
            (103, 1003, 503, "2022-01-01", "Lost Analytics"),
            (104, 1004, 504, "2023-01-01", "Recent Returns"),
        ],
    )

    conn.executemany(
        """
        INSERT INTO checkouts (user_id, book_id, checkout_time, return_time)
        VALUES (?, ?, ?, ?);
        """,
        [
            (1, 101, "2026-04-20 09:00:00", "2026-04-25 10:00:00"),
            (2, 102, "2026-04-28 11:00:00", None),
            (3, 103, "2026-03-20 12:00:00", None),
            (4, 104, "2026-04-02 13:00:00", "2026-04-10 14:00:00"),
            (1, 103, "2026-02-01 08:00:00", None),
            (2, 103, "2026-01-15 08:00:00", None),
        ],
    )

    try:
        yield conn
    finally:
        conn.close()


def rows_as_dicts(cursor: sqlite3.Cursor) -> list[dict[str, object]]:
    return [dict(row) for row in cursor.fetchall()]


def test_recent_checkout_users(connection: sqlite3.Connection):
    rows = rows_as_dicts(
        connection.execute(
            RECENT_CHECKOUT_USERS_SQL,
            {"as_of_date": "2026-05-03"},
        ),
    )

    assert rows == [{"name": "Alice"}, {"name": "Bob"}]


def test_current_checkout_users(connection: sqlite3.Connection):
    rows = rows_as_dicts(connection.execute(CURRENT_CHECKOUT_USERS_SQL))

    assert rows == [{"name": "Alice"}, {"name": "Bob"}, {"name": "Chen"}]


def test_overdue_users(connection: sqlite3.Connection):
    rows = rows_as_dicts(
        connection.execute(
            OVERDUE_USERS_SQL,
            {"as_of_date": "2026-05-03"},
        ),
    )

    assert rows == [{"name": "Alice"}, {"name": "Bob"}, {"name": "Chen"}]


def test_possibly_stolen_books_summary(connection: sqlite3.Connection):
    rows = rows_as_dicts(
        connection.execute(
            POSSIBLY_STOLEN_BOOKS_SUMMARY_SQL,
            {"as_of_date": "2026-05-03"},
        ),
    )

    assert rows == [
        {
            "book_name": "Lost Analytics",
            "overdue_checkout_count": 3,
            "oldest_checkout_date": "2026-01-15",
        },
    ]


def test_possibly_stolen_user_book_pairs(connection: sqlite3.Connection):
    rows = rows_as_dicts(
        connection.execute(
            POSSIBLY_STOLEN_USER_BOOK_PAIRS_SQL,
            {"as_of_date": "2026-05-03"},
        ),
    )

    assert rows == [
        {
            "user_name": "Alice",
            "book_name": "Lost Analytics",
            "checkout_date": "2026-02-01",
        },
        {
            "user_name": "Bob",
            "book_name": "Lost Analytics",
            "checkout_date": "2026-01-15",
        },
        {
            "user_name": "Chen",
            "book_name": "Lost Analytics",
            "checkout_date": "2026-03-20",
        },
    ]
