import numpy as np
import pytest

from solutions.probability import (
    estimate_total_jokes_mle,
    generate_simple_linear_data,
    joke_unique_count_likelihood,
    sample_exponential_inverse_cdf,
    simulate_diamond_after_star_redraws,
)


def test_simulate_diamond_after_star_redraws_returns_reasonable_estimate():
    estimate = simulate_diamond_after_star_redraws(
        n_simulations=5_000,
        seed=2026,
    )

    assert 0.20 < estimate < 0.35


def test_simulate_diamond_after_star_redraws_rejects_nonpositive_simulations():
    with pytest.raises(ValueError):
        simulate_diamond_after_star_redraws(n_simulations=0)


def test_sample_exponential_inverse_cdf_shape_and_mean():
    samples = sample_exponential_inverse_cdf(
        rate=2.0,
        size=100_000,
        seed=2026,
    )

    assert samples.shape == (100_000,)
    assert np.all(samples >= 0)
    assert samples.mean() == pytest.approx(0.5, abs=0.02)


def test_sample_exponential_inverse_cdf_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        sample_exponential_inverse_cdf(rate=0.0, size=10)


def test_generate_simple_linear_data_without_noise():
    x = np.array([0, 1, 2, 3])

    y = generate_simple_linear_data(
        x=x,
        intercept=10,
        slope=2,
        residual_sd=0,
        seed=2026,
    )

    np.testing.assert_allclose(y, np.array([10, 12, 14, 16], dtype=float))


def test_generate_simple_linear_data_rejects_negative_residual_sd():
    with pytest.raises(ValueError):
        generate_simple_linear_data(
            x=np.array([1, 2, 3]),
            intercept=0,
            slope=1,
            residual_sd=-1,
        )


def test_joke_unique_count_likelihood_for_four_visits_three_unique():
    likelihood = joke_unique_count_likelihood(
        total_jokes=5,
        n_visits=4,
        n_unique_seen=3,
    )

    assert likelihood == pytest.approx(0.576)


def test_estimate_total_jokes_mle_for_four_visits_three_unique():
    assert (
        estimate_total_jokes_mle(
            n_visits=4,
            n_unique_seen=3,
            max_jokes=100,
        )
        == 5
    )


def test_estimate_total_jokes_mle_rejects_all_unique_case():
    with pytest.raises(ValueError):
        estimate_total_jokes_mle(
            n_visits=4,
            n_unique_seen=4,
            max_jokes=100,
        )
