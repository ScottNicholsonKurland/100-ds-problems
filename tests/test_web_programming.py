import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from solutions.web_programming import (
    ocarina_world_record_progression,
    parse_duration_to_seconds,
    parse_redbubble_creators,
    plot_world_record_progression,
    read_super_metroid_leaderboard,
    redbubble_creators,
)


REDBUBBLE_HTML = """
<html>
  <body>
    <article data-testid="product-card">
      <a href="/people/artist_one/shop">artist_one</a>
      <span>zelda sticker</span>
    </article>
    <article data-testid="product-card" data-artist-name="artist_two">
      <span>zelda shirt</span>
    </article>
    <article data-testid="product-card">
      <a href="/people/artist_three/shop">artist_three</a>
      <span>eno sticker</span>
    </article>
  </body>
</html>
"""


def test_parse_redbubble_creators():
    assert parse_redbubble_creators(REDBUBBLE_HTML) == [
        "artist_one",
        "artist_two",
        "artist_three",
    ]


def test_parse_redbubble_creators_filters_by_type():
    assert parse_redbubble_creators(REDBUBBLE_HTML, type="shirt") == [
        "artist_two",
    ]


def test_redbubble_creators_fetches_multiple_pages():
    pages = {
        1: """
        <article data-testid="product-card">
          <a href="/people/page_one_artist/shop">page_one_artist</a>
          <span>zelda sticker</span>
        </article>
        """,
        2: """
        <article data-testid="product-card">
          <a href="/people/page_two_artist/shop">page_two_artist</a>
          <span>zelda sticker</span>
        </article>
        """,
    }

    fetched_urls = []

    def fake_fetch(url: str) -> str:
        fetched_urls.append(url)

        if "page=1" in url:
            return pages[1]

        if "page=2" in url:
            return pages[2]

        raise AssertionError(f"Unexpected URL: {url}")

    assert redbubble_creators("zelda", pages=2, fetch_text=fake_fetch) == [
        "page_one_artist",
        "page_two_artist",
    ]
    assert len(fetched_urls) == 2


def test_redbubble_creators_rejects_bad_pages():
    with pytest.raises(ValueError):
        redbubble_creators("zelda", pages=0, fetch_text=lambda url: "")


def test_parse_duration_to_seconds():
    assert parse_duration_to_seconds("12:34") == pytest.approx(754)
    assert parse_duration_to_seconds("1:02:03.5") == pytest.approx(3723.5)


def test_read_super_metroid_leaderboard_from_html():
    html = """
    <table>
      <tr>
        <th>Rank</th>
        <th>Player</th>
        <th>Time</th>
      </tr>
      <tr>
        <td>1</td>
        <td>Alice</td>
        <td>41:20</td>
      </tr>
      <tr>
        <td>2</td>
        <td>Bob</td>
        <td>42:05.5</td>
      </tr>
    </table>
    """

    result = read_super_metroid_leaderboard(html=html)

    expected = pd.DataFrame(
        {
            "rank": [1, 2],
            "player": ["Alice", "Bob"],
            "time": ["41:20", "42:05.5"],
            "time_seconds": [2480.0, 2525.5],
        },
    )

    pd.testing.assert_frame_equal(result, expected)


def test_ocarina_world_record_progression():
    leaderboard_html = """
    <html>
      <body>
        <a href="/runners/alice">Alice</a>
        <a href="/runners/bob">Bob</a>
      </body>
    </html>
    """

    runner_pages = {
        "http://zeldaspeedruns.com/runners/alice": """
        <table>
          <tr><th>Date</th><th>Time</th></tr>
          <tr><td>2020-01-01</td><td>10:00</td></tr>
          <tr><td>2020-03-01</td><td>9:50</td></tr>
        </table>
        """,
        "http://zeldaspeedruns.com/runners/bob": """
        <table>
          <tr><th>Date</th><th>Time</th></tr>
          <tr><td>2020-02-01</td><td>9:55</td></tr>
          <tr><td>2020-04-01</td><td>9:40</td></tr>
        </table>
        """,
    }

    result = ocarina_world_record_progression(
        leaderboard_html=leaderboard_html,
        runner_pages=runner_pages,
    )

    expected = pd.DataFrame(
        {
            "player": ["Alice", "Bob", "Alice", "Bob"],
            "date": [
                pd.Timestamp("2020-01-01").date(),
                pd.Timestamp("2020-02-01").date(),
                pd.Timestamp("2020-03-01").date(),
                pd.Timestamp("2020-04-01").date(),
            ],
            "time": ["10:00", "9:55", "9:50", "9:40"],
            "time_seconds": [600.0, 595.0, 590.0, 580.0],
            "world_record_seconds": [600.0, 595.0, 590.0, 580.0],
        },
    )

    pd.testing.assert_frame_equal(result, expected)


def test_plot_world_record_progression_returns_figure_and_axes():
    progression = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2020-01-01").date(),
                pd.Timestamp("2020-02-01").date(),
            ],
            "world_record_seconds": [600.0, 595.0],
        },
    )

    fig, ax = plot_world_record_progression(progression)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "World record time (seconds)"
    assert len(ax.lines) == 1

    plt.close(fig)


def test_plot_world_record_progression_rejects_missing_columns():
    with pytest.raises(ValueError):
        plot_world_record_progression(pd.DataFrame({"date": []}))
