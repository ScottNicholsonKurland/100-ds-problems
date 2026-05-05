"""Reference solutions for the Think Bayes-inspired extension problems."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import comb, exp, isinf, lgamma, log, pi, sqrt
from typing import Any

import numpy as np
import pandas as pd


def _validate_probability(value: float, name: str) -> float:
    """Validate a scalar probability."""
    probability = float(value)

    if not 0 <= probability <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")

    return probability


def _validate_count(value: int, name: str) -> int:
    """Validate a nonnegative integer count."""
    count = int(value)

    if count != value or count < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")

    return count


def _validate_pmf(pmf: Mapping[Any, float], name: str = "pmf") -> dict[Any, float]:
    """Validate and return a probability mass function."""
    if not pmf:
        raise ValueError(f"{name} must not be empty.")

    parsed = {hypothesis: float(probability) for hypothesis, probability in pmf.items()}

    if any(probability < 0 for probability in parsed.values()):
        raise ValueError(f"{name} probabilities must be nonnegative.")

    total = sum(parsed.values())

    if not np.isclose(total, 1.0):
        raise ValueError(f"{name} probabilities must sum to 1.")

    return parsed


def _as_numeric_grid(values: Iterable[float], name: str) -> np.ndarray:
    """Convert values to a non-empty one-dimensional numeric grid."""
    array = np.asarray(list(values), dtype=float)

    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")

    return array


def _poisson_probability(rate: float, count: int, exposure: float = 1.0) -> float:
    """Return the Poisson probability for count events."""
    mean = rate * exposure

    if mean < 0:
        raise ValueError("rate and exposure must imply a nonnegative mean.")

    if mean == 0:
        return 1.0 if count == 0 else 0.0

    return exp(count * log(mean) - mean - lgamma(count + 1))


def _normal_pdf(x_value: float, mean: float, sigma: float) -> float:
    """Return the normal probability density at x_value."""
    z_score = (x_value - mean) / sigma
    return exp(-0.5 * z_score**2) / (sigma * sqrt(2 * pi))


def normalize_weights(weights: Mapping[Any, float]) -> dict[Any, float]:
    """Return a normalized PMF from nonnegative weights."""
    if not weights:
        raise ValueError("weights must not be empty.")

    parsed = {hypothesis: float(weight) for hypothesis, weight in weights.items()}

    if any(weight < 0 for weight in parsed.values()):
        raise ValueError("weights must be nonnegative.")

    total = sum(parsed.values())

    if total <= 0:
        raise ValueError("at least one weight must be positive.")

    return {hypothesis: weight / total for hypothesis, weight in parsed.items()}


def make_uniform_prior(hypotheses: Sequence[Any]) -> dict[Any, float]:
    """Return a uniform prior over the unique hypotheses in order of appearance."""
    unique_hypotheses = list(dict.fromkeys(hypotheses))

    if not unique_hypotheses:
        raise ValueError("hypotheses must contain at least one value.")

    probability = 1 / len(unique_hypotheses)
    return {hypothesis: probability for hypothesis in unique_hypotheses}


def marginal_likelihood(
    prior: Mapping[Any, float],
    likelihoods: Mapping[Any, float],
) -> float:
    """Compute the total probability of data under a prior and likelihood table."""
    parsed_prior = _validate_pmf(prior, "prior")
    missing = set(parsed_prior) - set(likelihoods)

    if missing:
        missing_text = ", ".join(
            str(hypothesis) for hypothesis in sorted(missing, key=str)
        )
        raise ValueError(f"likelihoods missing hypotheses: {missing_text}")

    total = 0.0

    for hypothesis, prior_probability in parsed_prior.items():
        likelihood = float(likelihoods[hypothesis])

        if likelihood < 0:
            raise ValueError("likelihoods must be nonnegative.")

        total += prior_probability * likelihood

    return total


def update_pmf(
    prior: Mapping[Any, float],
    likelihoods: Mapping[Any, float],
) -> dict[Any, float]:
    """Update a prior PMF with likelihoods and return a posterior PMF."""
    parsed_prior = _validate_pmf(prior, "prior")
    evidence = marginal_likelihood(parsed_prior, likelihoods)

    if evidence <= 0:
        raise ValueError("the data have zero probability under the prior.")

    return {
        hypothesis: parsed_prior[hypothesis] * float(likelihoods[hypothesis]) / evidence
        for hypothesis in parsed_prior
    }


def probability_to_odds(probability: float) -> float:
    """Convert probability to odds."""
    probability = _validate_probability(probability, "probability")

    if probability == 1:
        return float("inf")

    return probability / (1 - probability)


def odds_to_probability(odds: float) -> float:
    """Convert odds to probability."""
    odds = float(odds)

    if odds < 0:
        raise ValueError("odds must be nonnegative.")

    if isinf(odds):
        return 1.0

    return odds / (1 + odds)


def diagnostic_test_posterior(
    prevalence: float,
    sensitivity: float,
    false_positive_rate: float,
) -> float:
    """Return P(condition | positive test)."""
    prevalence = _validate_probability(prevalence, "prevalence")
    sensitivity = _validate_probability(sensitivity, "sensitivity")
    false_positive_rate = _validate_probability(
        false_positive_rate,
        "false_positive_rate",
    )

    numerator = prevalence * sensitivity
    denominator = numerator + (1 - prevalence) * false_positive_rate

    if denominator == 0:
        raise ValueError("positive test has zero probability.")

    return numerator / denominator


def bayes_factor(first_likelihood: float, second_likelihood: float) -> float:
    """Return the Bayes factor comparing two hypotheses."""
    first = float(first_likelihood)
    second = float(second_likelihood)

    if first < 0 or second < 0:
        raise ValueError("likelihoods must be nonnegative.")

    if second == 0:
        raise ValueError("second_likelihood must be positive.")

    return first / second


def binomial_likelihood_grid(
    probabilities: Iterable[float],
    successes: int,
    failures: int,
) -> dict[float, float]:
    """Return exact binomial likelihoods across a probability grid."""
    probability_grid = _as_numeric_grid(probabilities, "probabilities")
    successes = _validate_count(successes, "successes")
    failures = _validate_count(failures, "failures")

    if np.any((probability_grid < 0) | (probability_grid > 1)):
        raise ValueError("probabilities must be between 0 and 1.")

    n_trials = successes + failures
    coefficient = comb(n_trials, successes)

    return {
        float(probability): coefficient
        * probability**successes
        * (1 - probability) ** failures
        for probability in probability_grid
    }


def update_binomial_grid(
    prior: Mapping[float, float],
    successes: int,
    failures: int,
) -> dict[float, float]:
    """Update a grid prior for a binomial success probability."""
    likelihoods = binomial_likelihood_grid(prior.keys(), successes, failures)
    return update_pmf(prior, likelihoods)


def posterior_summary(
    posterior: Mapping[float, float],
    threshold: float,
) -> dict[str, float]:
    """Return mean, MAP estimate, and probability at or below threshold."""
    parsed_posterior = _validate_pmf(posterior, "posterior")
    numeric_items = [
        (float(value), probability) for value, probability in parsed_posterior.items()
    ]
    mean = sum(value * probability for value, probability in numeric_items)
    map_value = min(
        numeric_items,
        key=lambda item: (-item[1], item[0]),
    )[0]
    probability_at_or_below = sum(
        probability for value, probability in numeric_items if value <= threshold
    )

    return {
        "mean": mean,
        "map": map_value,
        "probability_at_or_below": probability_at_or_below,
    }


def credible_interval(
    posterior: Mapping[float, float],
    mass: float = 0.9,
) -> tuple[float, float]:
    """Return a central credible interval from a discrete posterior PMF."""
    parsed_posterior = _validate_pmf(posterior, "posterior")
    mass = float(mass)

    if not 0 < mass <= 1:
        raise ValueError("mass must be between 0 and 1.")

    items = sorted(
        (float(value), probability) for value, probability in parsed_posterior.items()
    )
    lower_probability = (1 - mass) / 2
    upper_probability = 1 - lower_probability

    def quantile(probability: float) -> float:
        cumulative = 0.0

        for value, weight in items:
            cumulative += weight

            if cumulative >= probability:
                return value

        return items[-1][0]

    return quantile(lower_probability), quantile(upper_probability)


def posterior_predictive_binary(posterior: Mapping[float, float]) -> float:
    """Return the posterior predictive probability of Bernoulli success."""
    parsed_posterior = _validate_pmf(posterior, "posterior")

    if any(not 0 <= float(value) <= 1 for value in parsed_posterior):
        raise ValueError("posterior hypotheses must be probabilities.")

    probability = sum(
        float(value) * weight for value, weight in parsed_posterior.items()
    )

    return probability


def beta_binomial_posterior(
    alpha: float,
    beta: float,
    successes: int,
    failures: int,
) -> dict[str, float]:
    """Update Beta prior parameters after binomial observations."""
    alpha = float(alpha)
    beta = float(beta)

    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive.")

    successes = _validate_count(successes, "successes")
    failures = _validate_count(failures, "failures")

    return {"alpha": alpha + successes, "beta": beta + failures}


def beta_binomial_predictive(
    alpha: float,
    beta: float,
    successes: int = 0,
    failures: int = 0,
) -> float:
    """Return posterior predictive success probability for a Beta-binomial model."""
    posterior = beta_binomial_posterior(alpha, beta, successes, failures)
    return posterior["alpha"] / (posterior["alpha"] + posterior["beta"])


def gamma_poisson_posterior(
    shape: float,
    rate: float,
    counts: Iterable[int],
    exposures: float | Iterable[float] = 1.0,
) -> dict[str, float]:
    """Update Gamma prior parameters for a Poisson rate."""
    shape = float(shape)
    rate = float(rate)

    if shape <= 0 or rate <= 0:
        raise ValueError("shape and rate must be positive.")

    count_array = np.asarray(list(counts), dtype=float)

    if count_array.ndim != 1 or count_array.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional sequence.")

    if np.any(count_array < 0) or np.any(count_array != np.floor(count_array)):
        raise ValueError("counts must contain nonnegative integers.")

    if isinstance(exposures, int | float):
        exposure_array = np.full(count_array.shape, float(exposures))
    else:
        exposure_array = np.asarray(list(exposures), dtype=float)

    if exposure_array.shape != count_array.shape:
        raise ValueError("exposures must be scalar or have the same length as counts.")

    if np.any(exposure_array <= 0):
        raise ValueError("exposures must be positive.")

    return {
        "shape": shape + float(np.sum(count_array)),
        "rate": rate + float(np.sum(exposure_array)),
    }


def gamma_poisson_predictive_mean(
    shape: float,
    rate: float,
    exposure: float = 1.0,
) -> float:
    """Return the posterior predictive mean count for a new exposure period."""
    shape = float(shape)
    rate = float(rate)
    exposure = float(exposure)

    if shape <= 0 or rate <= 0:
        raise ValueError("shape and rate must be positive.")

    if exposure <= 0:
        raise ValueError("exposure must be positive.")

    return exposure * shape / rate


def update_poisson_rate_grid(
    prior: Mapping[float, float],
    counts: Iterable[int],
    exposures: float | Iterable[float] = 1.0,
) -> dict[float, float]:
    """Update a discrete prior over Poisson rates after observed counts."""
    parsed_prior = _validate_pmf(prior, "prior")
    count_array = np.asarray(list(counts), dtype=float)

    if count_array.ndim != 1 or count_array.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional sequence.")

    if np.any(count_array < 0) or np.any(count_array != np.floor(count_array)):
        raise ValueError("counts must contain nonnegative integers.")

    if isinstance(exposures, int | float):
        exposure_array = np.full(count_array.shape, float(exposures))
    else:
        exposure_array = np.asarray(list(exposures), dtype=float)

    if exposure_array.shape != count_array.shape:
        raise ValueError("exposures must be scalar or have the same length as counts.")

    if np.any(exposure_array <= 0):
        raise ValueError("exposures must be positive.")

    likelihoods = {}

    for rate in parsed_prior:
        rate = float(rate)

        if rate < 0:
            raise ValueError("rate hypotheses must be nonnegative.")

        likelihood = 1.0

        for count, exposure in zip(
            count_array.astype(int), exposure_array, strict=True
        ):
            likelihood *= _poisson_probability(rate, int(count), float(exposure))

        likelihoods[rate] = likelihood

    return update_pmf(parsed_prior, likelihoods)


def posterior_predictive_poisson(
    posterior: Mapping[float, float],
    max_count: int,
    exposure: float = 1.0,
) -> dict[int, float]:
    """Approximate a posterior predictive Poisson PMF on counts 0 through max_count."""
    parsed_posterior = _validate_pmf(posterior, "posterior")
    max_count = _validate_count(max_count, "max_count")
    exposure = float(exposure)

    if exposure <= 0:
        raise ValueError("exposure must be positive.")

    probabilities = {}

    for count in range(max_count + 1):
        probabilities[count] = sum(
            weight * _poisson_probability(float(rate), count, exposure)
            for rate, weight in parsed_posterior.items()
        )

    return normalize_weights(probabilities)


def update_normal_mean_grid(
    prior: Mapping[float, float],
    observations: Iterable[float],
    sigma: float,
) -> dict[float, float]:
    """Update a discrete prior over normal means with known sigma."""
    parsed_prior = _validate_pmf(prior, "prior")
    observation_array = _as_numeric_grid(observations, "observations")
    sigma = float(sigma)

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    likelihoods = {}

    for mean in parsed_prior:
        likelihood = 1.0

        for observation in observation_array:
            likelihood *= _normal_pdf(float(observation), float(mean), sigma)

        likelihoods[float(mean)] = likelihood

    return update_pmf(parsed_prior, likelihoods)


def posterior_predictive_normal(
    mean_posterior: Mapping[float, float],
    x_values: Iterable[float],
    sigma: float,
) -> dict[float, float]:
    """Approximate a posterior predictive normal distribution on an x grid."""
    parsed_posterior = _validate_pmf(mean_posterior, "mean_posterior")
    x_grid = _as_numeric_grid(x_values, "x_values")
    sigma = float(sigma)

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    weights = {}

    for x_value in x_grid:
        weights[float(x_value)] = sum(
            weight * _normal_pdf(float(x_value), float(mean), sigma)
            for mean, weight in parsed_posterior.items()
        )

    return normalize_weights(weights)


def joint_prior_independent(
    first_pmf: Mapping[Any, float],
    second_pmf: Mapping[Any, float],
    first_name: str = "first",
    second_name: str = "second",
) -> pd.DataFrame:
    """Build a joint distribution from two independent PMFs."""
    first = _validate_pmf(first_pmf, "first_pmf")
    second = _validate_pmf(second_pmf, "second_pmf")
    rows = []

    for first_value, first_probability in first.items():
        for second_value, second_probability in second.items():
            rows.append(
                {
                    first_name: first_value,
                    second_name: second_value,
                    "probability": first_probability * second_probability,
                }
            )

    return pd.DataFrame(rows)


def marginalize_joint(joint: pd.DataFrame, variable: str) -> dict[Any, float]:
    """Marginalize a joint distribution over one variable."""
    if variable not in joint.columns or "probability" not in joint.columns:
        raise ValueError("joint must contain the selected variable and probability.")

    weights = joint.groupby(variable, sort=True)["probability"].sum().to_dict()
    return normalize_weights(weights)


def condition_joint(
    joint: pd.DataFrame,
    given_variable: str,
    given_value: Any,
    target_variable: str,
) -> dict[Any, float]:
    """Condition a joint distribution and return a PMF for another variable."""
    required_columns = {given_variable, target_variable, "probability"}

    if not required_columns <= set(joint.columns):
        raise ValueError("joint is missing required columns.")

    selected = joint[joint[given_variable] == given_value]

    if selected.empty:
        raise ValueError("given_value has zero probability.")

    weights = (
        selected.groupby(target_variable, sort=True)["probability"].sum().to_dict()
    )
    return normalize_weights(weights)


def mixture_pmf(
    component_pmfs: Mapping[Any, Mapping[Any, float]],
    weights: Mapping[Any, float],
) -> dict[Any, float]:
    """Combine component PMFs using a PMF of mixture weights."""
    parsed_weights = _validate_pmf(weights, "weights")
    missing_components = set(parsed_weights) - set(component_pmfs)

    if missing_components:
        missing = ", ".join(
            str(component) for component in sorted(missing_components, key=str)
        )
        raise ValueError(f"component_pmfs missing components: {missing}")

    mixed_weights: defaultdict[Any, float] = defaultdict(float)

    for component, component_weight in parsed_weights.items():
        component_pmf = _validate_pmf(component_pmfs[component], "component_pmf")

        for outcome, probability in component_pmf.items():
            mixed_weights[outcome] += component_weight * probability

    return normalize_weights(mixed_weights)


def distribution_of_sum(
    first_pmf: Mapping[float, float],
    second_pmf: Mapping[float, float],
) -> dict[float, float]:
    """Compute the distribution of a sum of independent discrete variables."""
    first = _validate_pmf(first_pmf, "first_pmf")
    second = _validate_pmf(second_pmf, "second_pmf")
    weights: defaultdict[float, float] = defaultdict(float)

    for first_value, first_probability in first.items():
        for second_value, second_probability in second.items():
            weights[first_value + second_value] += (
                first_probability * second_probability
            )

    return dict(sorted(normalize_weights(weights).items()))


def distribution_of_maximum(
    first_pmf: Mapping[float, float],
    second_pmf: Mapping[float, float],
) -> dict[float, float]:
    """Compute the distribution of the maximum of independent discrete variables."""
    first = _validate_pmf(first_pmf, "first_pmf")
    second = _validate_pmf(second_pmf, "second_pmf")
    weights: defaultdict[float, float] = defaultdict(float)

    for first_value, first_probability in first.items():
        for second_value, second_probability in second.items():
            weights[max(first_value, second_value)] += (
                first_probability * second_probability
            )

    return dict(sorted(normalize_weights(weights).items()))


def expected_loss_table(
    posterior: Mapping[Any, float],
    actions: Sequence[Any],
    loss_function: Callable[[Any, Any], float],
) -> pd.DataFrame:
    """Compute expected loss for each action under a posterior PMF."""
    parsed_posterior = _validate_pmf(posterior, "posterior")

    if not actions:
        raise ValueError("actions must contain at least one value.")

    rows = []

    for action in actions:
        expected_loss = 0.0

        for state, probability in parsed_posterior.items():
            loss = float(loss_function(action, state))

            if loss < 0:
                raise ValueError("loss_function must return nonnegative losses.")

            expected_loss += probability * loss

        rows.append({"action": action, "expected_loss": expected_loss})

    return pd.DataFrame(rows)


def minimum_expected_loss_action(
    posterior: Mapping[Any, float],
    actions: Sequence[Any],
    loss_function: Callable[[Any, Any], float],
) -> Any:
    """Return the action with the smallest expected loss."""
    table = expected_loss_table(posterior, actions, loss_function)
    best_index = table["expected_loss"].idxmin()
    return table.loc[best_index, "action"]


def sequential_updates(
    prior: Mapping[Any, float],
    likelihood_tables: Iterable[Mapping[Any, float]],
) -> dict[Any, float]:
    """Apply a sequence of Bayesian PMF updates."""
    posterior = _validate_pmf(prior, "prior")

    for likelihoods in likelihood_tables:
        posterior = update_pmf(posterior, likelihoods)

    return posterior


def update_mark_recapture_grid(
    population_prior: Mapping[int, float],
    marked: int,
    sampled: int,
    recaptured: int,
) -> dict[int, float]:
    """Update a population-size grid using a mark-and-recapture likelihood."""
    parsed_prior = _validate_pmf(population_prior, "population_prior")
    marked = _validate_count(marked, "marked")
    sampled = _validate_count(sampled, "sampled")
    recaptured = _validate_count(recaptured, "recaptured")

    if recaptured > min(marked, sampled):
        raise ValueError("recaptured cannot exceed marked or sampled.")

    likelihoods = {}

    for population_size in parsed_prior:
        population_size = _validate_count(population_size, "population_size")

        if population_size < max(marked, sampled):
            raise ValueError(
                "population hypotheses must be at least marked and sampled."
            )

        unmarked = population_size - marked

        if sampled - recaptured > unmarked:
            likelihood = 0.0
        else:
            likelihood = (
                comb(marked, recaptured)
                * comb(unmarked, sampled - recaptured)
                / comb(population_size, sampled)
            )

        likelihoods[population_size] = likelihood

    return update_pmf(parsed_prior, likelihoods)
