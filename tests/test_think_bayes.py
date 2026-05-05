from math import comb, exp, isinf

import pandas as pd
import pytest

from solutions.think_bayes import (
    bayes_factor,
    beta_binomial_posterior,
    beta_binomial_predictive,
    binomial_likelihood_grid,
    condition_joint,
    credible_interval,
    diagnostic_test_posterior,
    distribution_of_maximum,
    distribution_of_sum,
    expected_loss_table,
    gamma_poisson_posterior,
    gamma_poisson_predictive_mean,
    joint_prior_independent,
    make_uniform_prior,
    marginal_likelihood,
    marginalize_joint,
    minimum_expected_loss_action,
    mixture_pmf,
    normalize_weights,
    odds_to_probability,
    posterior_predictive_binary,
    posterior_predictive_normal,
    posterior_predictive_poisson,
    posterior_summary,
    probability_to_odds,
    sequential_updates,
    update_binomial_grid,
    update_mark_recapture_grid,
    update_normal_mean_grid,
    update_pmf,
    update_poisson_rate_grid,
)


def assert_pmf_close(result, expected):
    assert set(result) == set(expected)

    for key, probability in expected.items():
        assert result[key] == pytest.approx(probability)


def test_normalize_weights():
    result = normalize_weights({"a": 1, "b": 2})

    assert_pmf_close(result, {"a": 1 / 3, "b": 2 / 3})


def test_normalize_weights_rejects_zero_total():
    with pytest.raises(ValueError):
        normalize_weights({"a": 0, "b": 0})


def test_normalize_weights_rejects_negative_weight():
    with pytest.raises(ValueError):
        normalize_weights({"a": 1, "b": -1})


def test_make_uniform_prior_keeps_unique_hypotheses_in_order():
    result = make_uniform_prior(["red", "blue", "red"])

    assert list(result) == ["red", "blue"]
    assert_pmf_close(result, {"red": 0.5, "blue": 0.5})


def test_make_uniform_prior_rejects_empty_hypotheses():
    with pytest.raises(ValueError):
        make_uniform_prior([])


def test_marginal_likelihood():
    prior = {"bowl_1": 0.5, "bowl_2": 0.5}
    likelihoods = {"bowl_1": 0.75, "bowl_2": 0.5}

    assert marginal_likelihood(prior, likelihoods) == pytest.approx(0.625)


def test_marginal_likelihood_rejects_missing_hypothesis():
    with pytest.raises(ValueError):
        marginal_likelihood({"a": 1.0}, {})


def test_update_pmf():
    prior = {"bowl_1": 0.5, "bowl_2": 0.5}
    likelihoods = {"bowl_1": 0.75, "bowl_2": 0.5}

    result = update_pmf(prior, likelihoods)

    assert_pmf_close(result, {"bowl_1": 0.6, "bowl_2": 0.4})


def test_update_pmf_rejects_zero_evidence():
    with pytest.raises(ValueError):
        update_pmf({"a": 1.0}, {"a": 0.0})


def test_probability_to_odds_and_back():
    odds = probability_to_odds(0.2)

    assert odds == pytest.approx(0.25)
    assert odds_to_probability(odds) == pytest.approx(0.2)
    assert isinf(probability_to_odds(1.0))
    assert odds_to_probability(float("inf")) == pytest.approx(1.0)


def test_odds_to_probability_rejects_negative_odds():
    with pytest.raises(ValueError):
        odds_to_probability(-1)


def test_diagnostic_test_posterior():
    result = diagnostic_test_posterior(
        prevalence=0.01,
        sensitivity=0.99,
        false_positive_rate=0.05,
    )

    assert result == pytest.approx(1 / 6)


def test_diagnostic_test_posterior_rejects_impossible_positive_test():
    with pytest.raises(ValueError):
        diagnostic_test_posterior(
            prevalence=0,
            sensitivity=1,
            false_positive_rate=0,
        )


def test_bayes_factor():
    assert bayes_factor(0.75, 0.5) == pytest.approx(1.5)


def test_bayes_factor_rejects_zero_second_likelihood():
    with pytest.raises(ValueError):
        bayes_factor(0.75, 0)


def test_binomial_likelihood_grid():
    result = binomial_likelihood_grid([0, 0.5, 1], successes=1, failures=1)

    assert_pmf_close(result, {0.0: 0.0, 0.5: 0.5, 1.0: 0.0})


def test_binomial_likelihood_grid_rejects_bad_probability():
    with pytest.raises(ValueError):
        binomial_likelihood_grid([-0.1, 0.5], successes=1, failures=1)


def test_update_binomial_grid():
    prior = {0.25: 1 / 3, 0.5: 1 / 3, 0.75: 1 / 3}

    result = update_binomial_grid(prior, successes=2, failures=1)

    assert_pmf_close(result, {0.25: 0.15, 0.5: 0.4, 0.75: 0.45})


def test_posterior_summary():
    result = posterior_summary({0.2: 0.25, 0.8: 0.75}, threshold=0.5)

    assert result["mean"] == pytest.approx(0.65)
    assert result["map"] == pytest.approx(0.8)
    assert result["probability_at_or_below"] == pytest.approx(0.25)


def test_credible_interval():
    posterior = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}

    assert credible_interval(posterior, mass=0.8) == (1, 4)


def test_credible_interval_rejects_bad_mass():
    with pytest.raises(ValueError):
        credible_interval({1: 1.0}, mass=0)


def test_posterior_predictive_binary():
    assert posterior_predictive_binary({0.2: 0.25, 0.8: 0.75}) == pytest.approx(0.65)


def test_posterior_predictive_binary_rejects_non_probability_hypothesis():
    with pytest.raises(ValueError):
        posterior_predictive_binary({2: 1.0})


def test_beta_binomial_posterior_and_predictive():
    posterior = beta_binomial_posterior(
        alpha=2,
        beta=3,
        successes=4,
        failures=1,
    )

    assert posterior == {"alpha": 6.0, "beta": 4.0}
    assert beta_binomial_predictive(2, 3, successes=4, failures=1) == pytest.approx(0.6)


def test_beta_binomial_posterior_rejects_bad_prior_parameter():
    with pytest.raises(ValueError):
        beta_binomial_posterior(alpha=0, beta=3, successes=1, failures=1)


def test_gamma_poisson_posterior_and_predictive_mean():
    posterior = gamma_poisson_posterior(
        shape=2,
        rate=1,
        counts=[3, 1],
        exposures=[1, 2],
    )

    assert posterior == {"shape": 6.0, "rate": 4.0}
    assert gamma_poisson_predictive_mean(6, 4, exposure=2) == pytest.approx(3)


def test_gamma_poisson_posterior_rejects_bad_exposures():
    with pytest.raises(ValueError):
        gamma_poisson_posterior(
            shape=2,
            rate=1,
            counts=[3, 1],
            exposures=[1],
        )


def test_update_poisson_rate_grid():
    posterior = update_poisson_rate_grid({1: 0.5, 2: 0.5}, counts=[0])
    expected_rate_1 = exp(-1) / (exp(-1) + exp(-2))

    assert posterior[1] == pytest.approx(expected_rate_1)
    assert posterior[2] == pytest.approx(1 - expected_rate_1)


def test_update_poisson_rate_grid_rejects_negative_rate_hypothesis():
    with pytest.raises(ValueError):
        update_poisson_rate_grid({-1: 0.5, 1: 0.5}, counts=[0])


def test_posterior_predictive_poisson_for_zero_rate():
    result = posterior_predictive_poisson({0: 1.0}, max_count=3)

    assert_pmf_close(result, {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})


def test_posterior_predictive_poisson_truncates_and_normalizes_grid():
    result = posterior_predictive_poisson({1: 1.0}, max_count=1)

    assert_pmf_close(result, {0: 0.5, 1: 0.5})


def test_update_normal_mean_grid():
    posterior = update_normal_mean_grid({-1: 0.5, 1: 0.5}, observations=[1], sigma=1)
    expected_mean_1 = 1 / (1 + exp(-2))

    assert posterior[1] == pytest.approx(expected_mean_1)
    assert posterior[-1] == pytest.approx(1 - expected_mean_1)


def test_update_normal_mean_grid_rejects_bad_sigma():
    with pytest.raises(ValueError):
        update_normal_mean_grid({0: 1.0}, observations=[1], sigma=0)


def test_posterior_predictive_normal():
    result = posterior_predictive_normal({0: 1.0}, x_values=[-1, 0, 1], sigma=1)
    middle_probability = 1 / (1 + 2 * exp(-0.5))

    assert result[0] == pytest.approx(middle_probability)
    assert result[-1] == pytest.approx((1 - middle_probability) / 2)
    assert result[1] == pytest.approx((1 - middle_probability) / 2)


def test_posterior_predictive_normal_rejects_bad_sigma():
    with pytest.raises(ValueError):
        posterior_predictive_normal({0: 1.0}, x_values=[0], sigma=0)


def test_joint_prior_independent():
    result = joint_prior_independent(
        {"a": 0.25, "b": 0.75},
        {1: 0.4, 2: 0.6},
        first_name="letter",
        second_name="number",
    )
    probability = result[(result["letter"] == "b") & (result["number"] == 2)].iloc[0][
        "probability"
    ]

    assert len(result) == 4
    assert result["probability"].sum() == pytest.approx(1)
    assert probability == pytest.approx(0.45)


def test_marginalize_joint():
    joint = joint_prior_independent({"a": 0.25, "b": 0.75}, {1: 0.4, 2: 0.6})

    assert_pmf_close(marginalize_joint(joint, "first"), {"a": 0.25, "b": 0.75})


def test_marginalize_joint_rejects_missing_column():
    with pytest.raises(ValueError):
        marginalize_joint(pd.DataFrame({"probability": [1]}), "missing")


def test_condition_joint():
    joint = joint_prior_independent({"a": 0.25, "b": 0.75}, {1: 0.4, 2: 0.6})

    assert_pmf_close(condition_joint(joint, "first", "a", "second"), {1: 0.4, 2: 0.6})


def test_condition_joint_rejects_zero_probability_value():
    joint = joint_prior_independent({"a": 1.0}, {1: 1.0})

    with pytest.raises(ValueError):
        condition_joint(joint, "first", "b", "second")


def test_mixture_pmf():
    result = mixture_pmf(
        {
            "fair": {"heads": 0.5, "tails": 0.5},
            "loaded": {"heads": 0.9, "tails": 0.1},
        },
        {"fair": 0.25, "loaded": 0.75},
    )

    assert_pmf_close(result, {"heads": 0.8, "tails": 0.2})


def test_mixture_pmf_rejects_missing_component():
    with pytest.raises(ValueError):
        mixture_pmf({"fair": {"heads": 1.0}}, {"loaded": 1.0})


def test_distribution_of_sum():
    result = distribution_of_sum({1: 0.5, 2: 0.5}, {10: 1.0})

    assert_pmf_close(result, {11: 0.5, 12: 0.5})


def test_distribution_of_maximum():
    result = distribution_of_maximum({1: 0.5, 3: 0.5}, {2: 1.0})

    assert_pmf_close(result, {2: 0.5, 3: 0.5})


def test_expected_loss_table():
    result = expected_loss_table(
        {0: 0.25, 10: 0.75},
        actions=[0, 10],
        loss_function=lambda action, state: abs(action - state),
    )

    losses = result.set_index("action")["expected_loss"].to_dict()

    assert losses[0] == pytest.approx(7.5)
    assert losses[10] == pytest.approx(2.5)


def test_expected_loss_table_rejects_negative_loss():
    with pytest.raises(ValueError):
        expected_loss_table(
            {0: 1.0},
            actions=[0],
            loss_function=lambda action, state: -1,
        )


def test_minimum_expected_loss_action():
    result = minimum_expected_loss_action(
        {0: 0.25, 10: 0.75},
        actions=[0, 10],
        loss_function=lambda action, state: abs(action - state),
    )

    assert result == 10


def test_sequential_updates():
    result = sequential_updates(
        {"A": 0.5, "B": 0.5},
        [
            {"A": 2, "B": 1},
            {"A": 1, "B": 3},
        ],
    )

    assert_pmf_close(result, {"A": 0.4, "B": 0.6})


def test_update_mark_recapture_grid():
    prior = {10: 0.5, 20: 0.5}

    result = update_mark_recapture_grid(
        prior,
        marked=5,
        sampled=4,
        recaptured=2,
    )
    likelihood_10 = comb(5, 2) * comb(5, 2) / comb(10, 4)
    likelihood_20 = comb(5, 2) * comb(15, 2) / comb(20, 4)
    expected_10 = likelihood_10 / (likelihood_10 + likelihood_20)

    assert result[10] == pytest.approx(expected_10)
    assert result[20] == pytest.approx(1 - expected_10)


def test_update_mark_recapture_grid_rejects_impossible_recapture_count():
    with pytest.raises(ValueError):
        update_mark_recapture_grid(
            {10: 1.0},
            marked=5,
            sampled=4,
            recaptured=6,
        )
