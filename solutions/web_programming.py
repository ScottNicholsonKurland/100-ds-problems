"""Reference solutions for the web-programming problem set.

The live websites in the original problems may change over time. These
solutions separate fetching from parsing so the parser logic can be tested with
static HTML fixtures instead of brittle network-dependent tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from urllib.parse import quote_plus, urljoin

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from matplotlib.axes import Axes
from matplotlib.figure import Figure


FetchText = Callable[[str], str]


def _default_fetch_text(url: str) -> str:
    """Fetch a URL and return response text."""
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "100-ds-problems educational scraper"},
    )
    response.raise_for_status()
    return response.text


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    """Return unique strings while preserving first-seen order."""
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def _text(element) -> str:
    """Return collapsed visible text for a BeautifulSoup element."""
    return " ".join(element.get_text(" ", strip=True).split())


def _extract_redbubble_creator_from_href(href: str) -> str | None:
    """Extract a Redbubble creator handle from a creator/shop URL."""
    marker = "/people/"

    if marker not in href:
        return None

    tail = href.split(marker, maxsplit=1)[1]
    creator = tail.split("/", maxsplit=1)[0].strip()

    return creator or None


def parse_redbubble_creators(
    html: str,
    type: str | None = None,  # noqa: A002 - matches original problem wording.
) -> list[str]:
    """Parse creator names from a Redbubble-like search result page.

    The parser looks for creator metadata attributes first, then falls back to
    Redbubble-style ``/people/<creator>/...`` links.
    """
    soup = BeautifulSoup(html, "html.parser")
    creators = []

    cards = soup.select(
        "[data-testid*='product'], [class*='product'], article, li, div"
    )

    for card in cards:
        card_text = _text(card).lower()

        if type is not None and type.lower() not in card_text:
            continue

        for attr in ("data-artist-name", "data-creator", "data-seller-name"):
            creator = card.get(attr)

            if creator:
                creators.append(str(creator).strip())

        for link in card.find_all("a", href=True):
            creator = _extract_redbubble_creator_from_href(link["href"])

            if creator is not None:
                creators.append(creator)

    return _unique_preserve_order([creator for creator in creators if creator])


def redbubble_creators(
    search_string: str,
    type: str | None = None,  # noqa: A002 - matches original problem wording.
    pages: int = 1,
    fetch_text: FetchText | None = None,
) -> list[str]:
    """Return Redbubble creators from one or more search result pages.

    Parameters
    ----------
    search_string:
        Search query, such as ``"zelda"``.
    type:
        Optional product category filter, such as ``"sticker"`` or ``"shirt"``.
    pages:
        Number of search result pages to fetch.
    fetch_text:
        Optional dependency-injected fetch function for tests.
    """
    if not search_string.strip():
        raise ValueError("search_string must not be empty.")

    if pages < 1:
        raise ValueError("pages must be positive.")

    fetch = fetch_text or _default_fetch_text
    creators = []

    for page in range(1, pages + 1):
        url = (
            "https://www.redbubble.com/shop/"
            f"?query={quote_plus(search_string)}&page={page}"
        )

        html = fetch(url)
        creators.extend(parse_redbubble_creators(html, type=type))

    return _unique_preserve_order(creators)


def parse_duration_to_seconds(duration: str) -> float:
    """Convert a speedrun duration string to seconds.

    Supports ``M:SS``, ``H:MM:SS``, and optional decimal seconds.
    """
    parts = duration.strip().split(":")

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Could not parse duration: {duration!r}")


def _parse_first_html_table(html: str) -> list[dict[str, str]]:
    """Parse the first HTML table into a list of row dictionaries."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    if table is None:
        raise ValueError("HTML does not contain a table.")

    rows = table.find_all("tr")

    if not rows:
        raise ValueError("HTML table contains no rows.")

    headers = [
        _text(cell).strip().lower().replace(" ", "_")
        for cell in rows[0].find_all(["th", "td"])
    ]

    if not headers:
        raise ValueError("HTML table contains no header row.")

    records = []

    for row in rows[1:]:
        cells = [_text(cell) for cell in row.find_all(["td", "th"])]

        if not cells:
            continue

        records.append(dict(zip(headers, cells, strict=False)))

    return records


def _find_column(record: dict[str, str], candidates: Sequence[str]) -> str:
    """Find a likely column name from candidate substrings."""
    for column in record:
        normalized = column.lower()

        for candidate in candidates:
            if candidate in normalized:
                return column

    raise ValueError(f"Could not find a column matching: {candidates}")


def read_super_metroid_leaderboard(
    html: str | None = None,
    fetch_text: FetchText | None = None,
    url: str = "http://deertier.com/Leaderboard/AnyPercentRealTime",
) -> pd.DataFrame:
    """Read a Super Metroid leaderboard page into a DataFrame."""
    if html is None:
        fetch = fetch_text or _default_fetch_text
        html = fetch(url)

    records = _parse_first_html_table(html)

    if not records:
        return pd.DataFrame(columns=["rank", "player", "time", "time_seconds"])

    sample = records[0]
    rank_column = _find_column(sample, ("rank", "#"))
    player_column = _find_column(sample, ("player", "runner", "name"))
    time_column = _find_column(sample, ("time",))

    parsed_records = []

    for record in records:
        parsed_records.append(
            {
                "rank": int(record[rank_column]),
                "player": record[player_column],
                "time": record[time_column],
                "time_seconds": parse_duration_to_seconds(record[time_column]),
            }
        )

    return pd.DataFrame(parsed_records)


def _parse_runner_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract unique runner names and profile links from a leaderboard page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for link in soup.find_all("a", href=True):
        player = _text(link)
        href = link["href"]

        if not player:
            continue

        absolute_url = urljoin(base_url, href)
        links.append((player, absolute_url))

    return _unique_preserve_order(links)


def parse_speedrun_history_table(html: str, player: str) -> pd.DataFrame:
    """Parse a runner history table with date and time columns."""
    records = _parse_first_html_table(html)

    if not records:
        return pd.DataFrame(columns=["player", "date", "time", "time_seconds"])

    sample = records[0]
    date_column = _find_column(sample, ("date", "submitted"))
    time_column = _find_column(sample, ("time",))

    parsed_records = []

    for record in records:
        date_value = pd.to_datetime(record[date_column], errors="coerce")

        if pd.isna(date_value):
            continue

        parsed_records.append(
            {
                "player": player,
                "date": date_value.date(),
                "time": record[time_column],
                "time_seconds": parse_duration_to_seconds(record[time_column]),
            }
        )

    return pd.DataFrame(parsed_records)


def ocarina_world_record_progression(
    leaderboard_html: str,
    runner_pages: dict[str, str] | None = None,
    fetch_text: FetchText | None = None,
    base_url: str = "http://zeldaspeedruns.com/leaderboards/oot/any",
) -> pd.DataFrame:
    """Build the historical world-record progression from runner pages.

    ``runner_pages`` maps absolute runner URLs to static HTML. Tests should use
    this argument instead of live network calls.
    """
    fetch = fetch_text or _default_fetch_text
    pages_by_url = runner_pages or {}
    runner_frames = []

    for player, runner_url in _parse_runner_links(leaderboard_html, base_url):
        html = pages_by_url.get(runner_url)

        if html is None:
            html = fetch(runner_url)

        runner_frames.append(parse_speedrun_history_table(html, player=player))

    if not runner_frames:
        return pd.DataFrame(
            columns=["date", "player", "time", "time_seconds", "world_record_seconds"]
        )

    runs = pd.concat(runner_frames, ignore_index=True)
    runs = runs.sort_values(["date", "time_seconds"]).reset_index(drop=True)

    best_time = float("inf")
    record_rows = []

    for row in runs.to_dict("records"):
        if row["time_seconds"] < best_time:
            best_time = row["time_seconds"]
            row["world_record_seconds"] = best_time
            record_rows.append(row)

    return pd.DataFrame(record_rows)


def plot_world_record_progression(
    progression: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Plot a world-record progression DataFrame."""
    required_columns = {"date", "world_record_seconds"}

    if not required_columns <= set(progression.columns):
        raise ValueError(
            "progression must contain date and world_record_seconds columns."
        )

    fig, ax = plt.subplots()

    ax.plot(
        pd.to_datetime(progression["date"]),
        progression["world_record_seconds"],
        marker="o",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("World record time (seconds)")
    ax.set_title("World Record Progression")
    fig.autofmt_xdate()

    return fig, ax
