"""Reference solutions for the probability problem set."""

from __future__ import annotations

from math import comb, factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray


def simulate_diamond_after_star_redraws(
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> float:
    """Estimate probability that the final hand contains a diamond.

    Deck interpretation:

    - 60 total cards
    - 3 diamond cards
    - 3 star cards
    - 54 ordinary cards

    Process interpretation:

    - Draw an initial hand of 5 cards.
    - If the hand contains star cards, discard all star cards.
    - For each discarded star card, draw 3 replacement/additional cards.
    - Repeat until the hand contains no star cards.

    Return the simulated probability that the final hand contains at least
    one diamond card.
    """
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")

    rng = np.random.default_rng(seed)
    successes = 0

    ordinary = 0
    diamond = 1
    star = 2

    base_deck = np.array([ordinary] * 54 + [diamond] * 3 + [star] * 3)

    for _ in range(n_simulations):
        deck = base_deck.copy()
        rng.shuffle(deck)

        hand = list(deck[:5])
        deck_position = 5

        while star in hand:
            star_count = sum(card == star for card in hand)
            hand = [card for card in hand if card != star]

            draw_count = min(3 * star_count, len(deck) - deck_position)
            hand.extend(deck[deck_position : deck_position + draw_count])
            deck_position += draw_count

        if diamond in hand:
            successes += 1

    return successes / n_simulations


def sample_exponential_inverse_cdf(
    rate: float,
    size: int | tuple[int, ...],
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Sample from an exponential distribution using inverse transform sampling.

    If U ~ Uniform(0, 1), then

        X = -log(1 - U) / rate

    follows an exponential distribution with the supplied rate parameter.
    """
    if rate <= 0:
        raise ValueError("rate must be positive.")

    rng = np.random.default_rng(seed)
    uniform_samples = rng.uniform(0.0, 1.0, size=size)

    return -np.log1p(-uniform_samples) / rate


def generate_simple_linear_data(
    x: ArrayLike,
    intercept: float,
    slope: float,
    residual_sd: float,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate y values from a simple linear model.

    Model:

        y = intercept + slope * x + epsilon

    where epsilon is Gaussian noise with mean 0 and standard deviation
    residual_sd.
    """
    if residual_sd < 0:
        raise ValueError("residual_sd must be nonnegative.")

    x_array = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)

    residuals = rng.normal(
        loc=0.0,
        scale=residual_sd,
        size=x_array.shape,
    )

    return intercept + slope * x_array + residuals


def _stirling_second_kind(n_items: int, n_groups: int) -> int:
    """Return Stirling number of the second kind S(n_items, n_groups)."""
    if n_items < 0 or n_groups < 0:
        raise ValueError("n_items and n_groups must be nonnegative.")

    if n_items == n_groups == 0:
        return 1

    if n_items == 0 or n_groups == 0:
        return 0

    return sum(
        (-1) ** (n_groups - j)
        * comb(n_groups, j)
        * j**n_items
        for j in range(n_groups + 1)
    ) // factorial(n_groups)


def joke_unique_count_likelihood(
    total_jokes: int,
    n_visits: int,
    n_unique_seen: int,
) -> float:
    """Compute likelihood of seeing n_unique_seen jokes in n_visits visits.

    Assumes each website visit independently samples one joke uniformly from
    total_jokes possible jokes.

    The probability of seeing exactly k unique jokes in m visits from N total
    possible jokes is:

        S(m, k) * falling_factorial(N, k) / N**m

    where S(m, k) is a Stirling number of the second kind.
    """
    if total_jokes < 1:
        raise ValueError("total_jokes must be positive.")

    if n_visits < 1:
        raise ValueError("n_visits must be positive.")

    if not 1 <= n_unique_seen <= n_visits:
        raise ValueError("n_unique_seen must be between 1 and n_visits.")

    if total_jokes < n_unique_seen:
        return 0.0

    stirling = _stirling_second_kind(n_visits, n_unique_seen)

    falling_factorial = 1
    for value in range(total_jokes, total_jokes - n_unique_seen, -1):
        falling_factorial *= value

    return stirling * falling_factorial / total_jokes**n_visits


def estimate_total_jokes_mle(
    n_visits: int,
    n_unique_seen: int,
    max_jokes: int = 1_000,
) -> int:
    """Estimate total joke count by maximum likelihood.

    Raises ValueError when every visit produced a unique joke, because then
    the likelihood increases toward the upper search limit rather than giving
    a meaningful finite MLE.
    """
    if max_jokes < n_unique_seen:
        raise ValueError("max_jokes must be at least n_unique_seen.")

    if n_unique_seen == n_visits:
        raise ValueError(
            "No finite MLE is available when every observed joke is unique.",
        )

    candidates = range(n_unique_seen, max_jokes + 1)
    likelihoods = [
        joke_unique_count_likelihood(
            total_jokes=candidate,
            n_visits=n_visits,
            n_unique_seen=n_unique_seen,
        )
        for candidate in candidates
    ]

    best_index = int(np.argmax(likelihoods))

    return n_unique_seen + best_index
