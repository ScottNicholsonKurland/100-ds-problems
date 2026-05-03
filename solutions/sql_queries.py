"""Reference SQL queries for the SQL problem set.

The queries are written to run against SQLite in the test suite. They use
named parameters so tests can provide a stable "as of" date.
"""

RECENT_CHECKOUT_USERS_SQL = """
SELECT DISTINCT
    u.name
FROM users AS u
INNER JOIN checkouts AS c
    ON u.user_id = c.user_id
WHERE DATE(c.checkout_time) >= DATE(:as_of_date, '-1 month')
ORDER BY u.name;
"""

CURRENT_CHECKOUT_USERS_SQL = """
SELECT DISTINCT
    u.name
FROM users AS u
INNER JOIN checkouts AS c
    ON u.user_id = c.user_id
WHERE c.return_time IS NULL
ORDER BY u.name;
"""

OVERDUE_USERS_SQL = """
SELECT DISTINCT
    u.name
FROM users AS u
INNER JOIN checkouts AS c
    ON u.user_id = c.user_id
WHERE c.return_time IS NULL
  AND DATE(c.checkout_time) < DATE(:as_of_date, '-1 month')
ORDER BY u.name;
"""

POSSIBLY_STOLEN_BOOKS_SUMMARY_SQL = """
SELECT
    b.name AS book_name,
    COUNT(*) AS overdue_checkout_count,
    MIN(DATE(c.checkout_time)) AS oldest_checkout_date
FROM books AS b
INNER JOIN checkouts AS c
    ON b.book_id = c.book_id
WHERE c.return_time IS NULL
  AND DATE(c.checkout_time) < DATE(:as_of_date, '-1 month')
GROUP BY
    b.book_id,
    b.name
ORDER BY
    overdue_checkout_count DESC,
    book_name;
"""

POSSIBLY_STOLEN_USER_BOOK_PAIRS_SQL = """
SELECT
    u.name AS user_name,
    b.name AS book_name,
    DATE(c.checkout_time) AS checkout_date
FROM users AS u
INNER JOIN checkouts AS c
    ON u.user_id = c.user_id
INNER JOIN books AS b
    ON c.book_id = b.book_id
WHERE c.return_time IS NULL
  AND DATE(c.checkout_time) < DATE(:as_of_date, '-1 month')
ORDER BY
    u.name,
    b.name,
    checkout_date;
"""
