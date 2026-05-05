"""Reference solutions for the Think Stats-inspired extension problems."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from itertools import combinations
from math import comb
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def _as_numeric_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert values to a non-empty one-dimensional float array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")

    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN values.")

    return array


def _validate_equal_length(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    first_name: str = "first",
    second_name: str = "second",
) -> None:
    """Validate that two arrays have equal length."""
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have equal length.")


def _format_bin_label(lower: float, upper: float, is_last: bool) -> str:
    """Return a compact half-open interval label."""
    left = f"{lower:g}"
    right = f"{upper:g}"
    closing = "]" if is_last else ")"
    return f"[{left}, {right}{closing}"


def _validate_bin_edges(bin_edges: Sequence[float]) -> NDArray[np.float64]:
    """Validate and return strictly increasing bin edges."""
    edges = np.asarray(bin_edges, dtype=float)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must contain at least two values.")

    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing.")

    return edges


def _rank_average_ties(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return one-based ranks with average ranks for ties."""
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def column_summary_report(
    data: pd.DataFrame,
    example_count: int = 3,
) -> pd.DataFrame:
    """Return dtype, missingness, cardinality, and examples for each column."""
    if example_count < 0:
        raise ValueError("example_count must be nonnegative.")

    rows = []

    for column in data.columns:
        series = data[column]
        non_missing = series.dropna()
        examples = non_missing.drop_duplicates().head(example_count).tolist()

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "count": int(series.count()),
                "missing_count": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "examples": examples,
            }
        )

    return pd.DataFrame(rows)


def group_comparison_summary(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Compare a numeric variable across groups."""
    missing_columns = {group_col, value_col} - set(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"data is missing required columns: {missing}")

    if not pd.api.types.is_numeric_dtype(data[value_col]):
        raise ValueError("value_col must be numeric.")

    rows = []

    for group_value, values in data.groupby(group_col, sort=True)[value_col]:
        clean_values = values.dropna()
        quantiles = clean_values.quantile([0.25, 0.75])

        rows.append(
            {
                "group": group_value,
                "count": int(clean_values.count()),
                "mean": float(clean_values.mean()),
                "median": float(clean_values.median()),
                "std": float(clean_values.std(ddof=1)),
                "min": float(clean_values.min()),
                "max": float(clean_values.max()),
                "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
            }
        )

    return pd.DataFrame(rows)


def frequency_table(values: pd.Series) -> pd.DataFrame:
    """Return value counts and proportions sorted by descending count."""
    counts = values.value_counts(dropna=False)
    total = len(values)

    if total == 0:
        raise ValueError("values must contain at least one item.")

    result = counts.rename_axis("value").reset_index(name="count")
    result["proportion"] = result["count"] / total
    result["_sort_value"] = result["value"].astype(str)
    result = result.sort_values(
        ["count", "_sort_value"],
        ascending=[False, True],
        kind="mergesort",
    )

    return result.drop(columns="_sort_value").reset_index(drop=True)


def histogram_bin_counts(
    values: ArrayLike,
    bin_edges: Sequence[float],
) -> pd.DataFrame:
    """Count numeric values in bins defined by bin edges."""
    array = _as_numeric_array(values, "values")
    edges = _validate_bin_edges(bin_edges)
    counts, _ = np.histogram(array, bins=edges)
    total = counts.sum()

    labels = [
        _format_bin_label(edges[index], edges[index + 1], index == len(edges) - 2)
        for index in range(len(edges) - 1)
    ]

    return pd.DataFrame(
        {
            "bin": labels,
            "count": counts.astype(int),
            "proportion": counts / total if total else counts.astype(float),
        }
    )


def build_pmf(values: Sequence[Any]) -> dict[Any, float]:
    """Build a probability mass function from discrete values."""
    if len(values) == 0:
        raise ValueError("values must contain at least one item.")

    counts = Counter(values)
    total = sum(counts.values())

    return {value: count / total for value, count in counts.items()}


def _validate_pmf(pmf: dict[Any, float]) -> None:
    """Validate probability mass values."""
    if not pmf:
        raise ValueError("pmf must not be empty.")

    probabilities = np.asarray(list(pmf.values()), dtype=float)

    if np.any(probabilities < 0):
        raise ValueError("pmf probabilities must be nonnegative.")

    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("pmf probabilities must sum to 1.")


def pmf_mean(pmf: dict[float, float]) -> float:
    """Compute the mean of a discrete distribution represented as a PMF."""
    _validate_pmf(pmf)
    return float(sum(value * probability for value, probability in pmf.items()))


def pmf_variance(pmf: dict[float, float]) -> float:
    """Compute the variance of a discrete distribution represented as a PMF."""
    mean = pmf_mean(pmf)
    return float(
        sum(probability * (value - mean) ** 2 for value, probability in pmf.items())
    )


def compare_pmfs(
    first_pmf: dict[Any, float],
    second_pmf: dict[Any, float],
) -> pd.DataFrame:
    """Compare two PMFs over the same outcome space."""
    _validate_pmf(first_pmf)
    _validate_pmf(second_pmf)

    if set(first_pmf) != set(second_pmf):
        raise ValueError("PMFs must have the same outcome space.")

    try:
        outcomes = sorted(first_pmf)
    except TypeError:
        outcomes = list(first_pmf)

    return pd.DataFrame(
        {
            "outcome": outcomes,
            "first_probability": [first_pmf[outcome] for outcome in outcomes],
            "second_probability": [second_pmf[outcome] for outcome in outcomes],
            "probability_difference": [
                first_pmf[outcome] - second_pmf[outcome] for outcome in outcomes
            ],
        }
    )


def build_cdf(values: ArrayLike) -> pd.DataFrame:
    """Build an empirical CDF from numeric data."""
    array = _as_numeric_array(values, "values")
    unique_values, counts = np.unique(array, return_counts=True)
    cumulative_probabilities = np.cumsum(counts) / array.size

    return pd.DataFrame(
        {
            "value": unique_values,
            "cumulative_probability": cumulative_probabilities,
        }
    )


def percentile_rank(values: ArrayLike, value: float) -> float:
    """Return the percentage of sample values less than or equal to value."""
    array = _as_numeric_array(values, "values")
    return float(np.mean(array <= value) * 100)


def value_at_percentile(values: ArrayLike, percentile: float) -> float:
    """Return the sample value at a requested percentile."""
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")

    array = _as_numeric_array(values, "values")
    return float(np.percentile(array, percentile))


def normal_model_quantiles(
    values: ArrayLike,
    probabilities: Sequence[float],
) -> pd.DataFrame:
    """Fit a normal model and return model quantiles."""
    array = _as_numeric_array(values, "values")

    if array.size < 2:
        raise ValueError("values must contain at least two items.")

    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1))

    if std == 0:
        raise ValueError("values must have nonzero standard deviation.")

    distribution = NormalDist(mu=mean, sigma=std)
    probabilities_array = np.asarray(probabilities, dtype=float)

    if probabilities_array.ndim != 1 or probabilities_array.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional sequence.")

    if np.any((probabilities_array <= 0) | (probabilities_array >= 1)):
        raise ValueError("probabilities must be strictly between 0 and 1.")

    return pd.DataFrame(
        {
            "probability": probabilities_array,
            "quantile": [
                distribution.inv_cdf(probability) for probability in probabilities_array
            ],
            "mean": mean,
            "std": std,
        }
    )


def exponential_rate(values: ArrayLike) -> float:
    """Estimate the rate parameter of an exponential distribution."""
    array = _as_numeric_array(values, "values")

    if np.any(array < 0):
        raise ValueError("values must be nonnegative.")

    mean = float(np.mean(array))

    if mean <= 0:
        raise ValueError("values must have a positive mean.")

    return 1 / mean


def max_cdf_difference(
    values: ArrayLike,
    model_cdf: Callable[[float], float],
) -> float:
    """Return the maximum absolute difference between empirical and model CDFs."""
    empirical = build_cdf(values)
    model_probabilities = empirical["value"].map(model_cdf).to_numpy(dtype=float)

    if np.any((model_probabilities < 0) | (model_probabilities > 1)):
        raise ValueError("model_cdf must return probabilities between 0 and 1.")

    differences = np.abs(
        empirical["cumulative_probability"].to_numpy() - model_probabilities
    )

    return float(np.max(differences))


def kernel_density_grid(
    values: ArrayLike,
    x_values: ArrayLike | None = None,
    grid_size: int = 100,
    bandwidth: float | None = None,
) -> pd.DataFrame:
    """Estimate a Gaussian kernel density on a grid."""
    array = _as_numeric_array(values, "values")

    if bandwidth is None:
        sample_std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        bandwidth = 1.06 * sample_std * array.size ** (-1 / 5)

        if bandwidth <= 0:
            bandwidth = 1.0

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    if x_values is None:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        data_min = float(np.min(array))
        data_max = float(np.max(array))
        span = data_max - data_min
        padding = 0.5 * bandwidth if span == 0 else 0.1 * span
        grid = np.linspace(data_min - padding, data_max + padding, grid_size)
    else:
        grid = _as_numeric_array(x_values, "x_values")

    scaled = (grid[:, np.newaxis] - array[np.newaxis, :]) / bandwidth
    density = np.mean(np.exp(-0.5 * scaled**2), axis=1)
    density = density / (bandwidth * np.sqrt(2 * np.pi))

    return pd.DataFrame({"x": grid, "density": density})


def density_area_is_close(
    x_values: ArrayLike,
    density_values: ArrayLike,
    tolerance: float = 0.05,
) -> bool:
    """Return whether a density curve integrates to approximately 1."""
    x_array = _as_numeric_array(x_values, "x_values")
    density_array = _as_numeric_array(density_values, "density_values")
    _validate_equal_length(x_array, density_array, "x_values", "density_values")

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")

    widths = np.diff(x_array)

    if np.any(widths <= 0):
        raise ValueError("x_values must be strictly increasing.")

    area = float(np.sum(widths * (density_array[:-1] + density_array[1:]) / 2))
    return abs(area - 1) <= tolerance


def covariance(
    first_values: ArrayLike,
    second_values: ArrayLike,
    sample: bool = True,
) -> float:
    """Compute covariance from two equal-length numeric arrays."""
    first = _as_numeric_array(first_values, "first_values")
    second = _as_numeric_array(second_values, "second_values")
    _validate_equal_length(first, second, "first_values", "second_values")

    degrees_of_freedom = 1 if sample else 0

    if first.size <= degrees_of_freedom:
        raise ValueError("not enough values for the requested covariance.")

    centered_first = first - np.mean(first)
    centered_second = second - np.mean(second)

    return float(
        np.sum(centered_first * centered_second) / (first.size - degrees_of_freedom)
    )


def pearson_correlation(
    first_values: ArrayLike,
    second_values: ArrayLike,
) -> float:
    """Compute Pearson correlation from two equal-length numeric arrays."""
    first = _as_numeric_array(first_values, "first_values")
    second = _as_numeric_array(second_values, "second_values")
    _validate_equal_length(first, second, "first_values", "second_values")

    centered_first = first - np.mean(first)
    centered_second = second - np.mean(second)
    denominator = np.sqrt(np.sum(centered_first**2) * np.sum(centered_second**2))

    if denominator == 0:
        raise ValueError("both arrays must have nonzero variance.")

    return float(np.sum(centered_first * centered_second) / denominator)


def spearman_correlation(
    first_values: ArrayLike,
    second_values: ArrayLike,
) -> float:
    """Compute Spearman correlation by ranking values then correlating ranks."""
    first = _as_numeric_array(first_values, "first_values")
    second = _as_numeric_array(second_values, "second_values")
    _validate_equal_length(first, second, "first_values", "second_values")

    first_ranks = _rank_average_ties(first)
    second_ranks = _rank_average_ties(second)

    return pearson_correlation(first_ranks, second_ranks)


def conditional_mean_table(
    bin_values: ArrayLike,
    target_values: ArrayLike,
    bin_edges: Sequence[float],
) -> pd.DataFrame:
    """Bin one numeric variable and compute target means within each bin."""
    bins = _as_numeric_array(bin_values, "bin_values")
    targets = _as_numeric_array(target_values, "target_values")
    _validate_equal_length(bins, targets, "bin_values", "target_values")
    edges = _validate_bin_edges(bin_edges)

    rows = []

    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        is_last = index == len(edges) - 2

        if is_last:
            mask = (bins >= lower) & (bins <= upper)
        else:
            mask = (bins >= lower) & (bins < upper)

        selected_targets = targets[mask]

        rows.append(
            {
                "bin": _format_bin_label(lower, upper, is_last),
                "count": int(selected_targets.size),
                "mean": (
                    float(np.mean(selected_targets))
                    if selected_targets.size
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_confidence_interval(
    values: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float],
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Estimate a confidence interval by bootstrap resampling."""
    array = _as_numeric_array(values, "values")

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    statistics = np.empty(n_resamples)

    for index in range(n_resamples):
        resample = rng.choice(array, size=array.size, replace=True)
        statistics[index] = statistic(resample)

    tail_probability = (1 - confidence_level) / 2

    return {
        "statistic": float(statistic(array)),
        "lower": float(np.quantile(statistics, tail_probability)),
        "upper": float(np.quantile(statistics, 1 - tail_probability)),
        "confidence_level": confidence_level,
    }


def simulated_standard_error(
    population: ArrayLike,
    statistic: Callable[[NDArray[np.float64]], float],
    sample_size: int,
    n_simulations: int = 1_000,
    seed: int | None = None,
) -> float:
    """Estimate standard error by repeated sampling from a population."""
    population_array = _as_numeric_array(population, "population")

    if sample_size < 1:
        raise ValueError("sample_size must be positive.")

    if n_simulations < 2:
        raise ValueError("n_simulations must be at least 2.")

    rng = np.random.default_rng(seed)
    simulated_statistics = np.empty(n_simulations)

    for index in range(n_simulations):
        sample = rng.choice(population_array, size=sample_size, replace=True)
        simulated_statistics[index] = statistic(sample)

    return float(np.std(simulated_statistics, ddof=1))


def permutation_test_difference_in_means(
    first_group: ArrayLike,
    second_group: ArrayLike,
    n_permutations: int | None = None,
    seed: int | None = None,
    max_exact_partitions: int = 20_000,
) -> dict[str, float | int]:
    """Run a two-sided permutation test for a difference in means."""
    first = _as_numeric_array(first_group, "first_group")
    second = _as_numeric_array(second_group, "second_group")
    combined = np.concatenate([first, second])
    first_size = first.size
    observed_difference = float(np.mean(first) - np.mean(second))

    total_partitions = int(comb(combined.size, first_size))

    if n_permutations is None and total_partitions <= max_exact_partitions:
        differences = []

        for selected_indices in combinations(range(combined.size), first_size):
            mask = np.zeros(combined.size, dtype=bool)
            mask[list(selected_indices)] = True
            differences.append(
                float(np.mean(combined[mask]) - np.mean(combined[~mask]))
            )

        differences_array = np.asarray(differences)
        p_value = float(np.mean(np.abs(differences_array) >= abs(observed_difference)))

        return {
            "observed_difference": observed_difference,
            "p_value": p_value,
            "n_permutations": total_partitions,
        }

    if n_permutations is None:
        n_permutations = max_exact_partitions

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")

    rng = np.random.default_rng(seed)
    extreme_count = 0

    for _ in range(n_permutations):
        shuffled = rng.permutation(combined)
        permuted_first = shuffled[:first_size]
        permuted_second = shuffled[first_size:]
        difference = float(np.mean(permuted_first) - np.mean(permuted_second))

        if abs(difference) >= abs(observed_difference):
            extreme_count += 1

    p_value = (extreme_count + 1) / (n_permutations + 1)

    return {
        "observed_difference": observed_difference,
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }


def chi_square_statistic(
    observed_counts: ArrayLike,
    expected_counts: ArrayLike,
) -> float:
    """Compute a chi-square goodness-of-fit statistic."""
    observed = _as_numeric_array(observed_counts, "observed_counts")
    expected = _as_numeric_array(expected_counts, "expected_counts")
    _validate_equal_length(observed, expected, "observed_counts", "expected_counts")

    if np.any(observed < 0):
        raise ValueError("observed_counts must be nonnegative.")

    if np.any(expected <= 0):
        raise ValueError("expected_counts must be positive.")

    return float(np.sum((observed - expected) ** 2 / expected))


def simulate_two_group_power(
    effect_size: float,
    sample_size: int,
    noise_sd: float,
    n_simulations: int = 1_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> float:
    """Simulate power for a normal two-group mean comparison."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive.")

    if noise_sd <= 0:
        raise ValueError("noise_sd must be positive.")

    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    normal = NormalDist()
    standard_error = np.sqrt(2 * noise_sd**2 / sample_size)
    significant_count = 0

    for _ in range(n_simulations):
        control = rng.normal(loc=0, scale=noise_sd, size=sample_size)
        treatment = rng.normal(loc=effect_size, scale=noise_sd, size=sample_size)
        observed_difference = float(np.mean(treatment) - np.mean(control))
        z_score = observed_difference / standard_error
        p_value = 2 * (1 - normal.cdf(abs(z_score)))

        if p_value < alpha:
            significant_count += 1

    return significant_count / n_simulations


def simple_least_squares_fit(
    x_values: ArrayLike,
    y_values: ArrayLike,
) -> dict[str, float]:
    """Fit simple linear regression using closed-form formulas."""
    x_array = _as_numeric_array(x_values, "x_values")
    y_array = _as_numeric_array(y_values, "y_values")
    _validate_equal_length(x_array, y_array, "x_values", "y_values")

    x_centered = x_array - np.mean(x_array)
    denominator = np.sum(x_centered**2)

    if denominator == 0:
        raise ValueError("x_values must have nonzero variance.")

    slope = float(np.sum(x_centered * (y_array - np.mean(y_array))) / denominator)
    intercept = float(np.mean(y_array) - slope * np.mean(x_array))

    return {"intercept": intercept, "slope": slope}


def residual_diagnostics_table(
    x_values: ArrayLike,
    y_values: ArrayLike,
    intercept: float,
    slope: float,
) -> pd.DataFrame:
    """Return fitted values and residual diagnostics for simple regression."""
    x_array = _as_numeric_array(x_values, "x_values")
    y_array = _as_numeric_array(y_values, "y_values")
    _validate_equal_length(x_array, y_array, "x_values", "y_values")

    fitted = intercept + slope * x_array
    residuals = y_array - fitted

    return pd.DataFrame(
        {
            "x": x_array,
            "y": y_array,
            "fitted": fitted,
            "residual": residuals,
            "absolute_residual": np.abs(residuals),
            "squared_residual": residuals**2,
        }
    )


def build_design_matrix(
    data: pd.DataFrame,
    numeric_columns: Sequence[str],
    add_intercept: bool = True,
) -> NDArray[np.float64]:
    """Build a numeric design matrix from selected DataFrame columns."""
    missing_columns = set(numeric_columns) - set(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"data is missing required columns: {missing}")

    matrices = []

    if add_intercept:
        matrices.append(np.ones((len(data), 1)))

    if numeric_columns:
        matrices.append(data.loc[:, numeric_columns].to_numpy(dtype=float))

    if not matrices:
        raise ValueError("at least one column or intercept is required.")

    return np.column_stack(matrices)


def fit_regression_with_categoricals(
    data: pd.DataFrame,
    target_col: str,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    add_intercept: bool = True,
    drop_first: bool = True,
) -> dict[str, Any]:
    """One-hot encode categoricals and fit a least-squares regression model."""
    required_columns = {target_col, *numeric_columns, *categorical_columns}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"data is missing required columns: {missing}")

    numeric_data = data.loc[:, numeric_columns].astype(float)
    categorical_data = pd.get_dummies(
        data.loc[:, categorical_columns],
        columns=list(categorical_columns),
        drop_first=drop_first,
        dtype=float,
    )
    design_frame = pd.concat([numeric_data, categorical_data], axis=1)

    if add_intercept:
        design_frame.insert(0, "intercept", 1.0)

    design_matrix = design_frame.to_numpy(dtype=float)
    target = data[target_col].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design_matrix, target, rcond=None)
    predictions = design_matrix @ coefficients

    return {
        "coefficients": pd.Series(coefficients, index=design_frame.columns),
        "design_matrix": design_matrix,
        "predictions": predictions,
        "feature_names": list(design_frame.columns),
    }


def survival_curve(
    durations: ArrayLike,
    events: Iterable[bool | int],
) -> pd.DataFrame:
    """Build a Kaplan-Meier-style survival table from event data."""
    duration_array = _as_numeric_array(durations, "durations")
    event_array = np.asarray(list(events), dtype=bool)

    if event_array.ndim != 1:
        raise ValueError("events must be one-dimensional.")

    if duration_array.shape[0] != event_array.shape[0]:
        raise ValueError("durations and events must have equal length.")

    if np.any(duration_array < 0):
        raise ValueError("durations must be nonnegative.")

    event_times = np.sort(np.unique(duration_array[event_array]))
    survival_probability = 1.0
    rows = []

    for event_time in event_times:
        at_risk = int(np.sum(duration_array >= event_time))
        n_events = int(np.sum((duration_array == event_time) & event_array))
        survival_probability *= 1 - n_events / at_risk

        rows.append(
            {
                "time": float(event_time),
                "number_at_risk": at_risk,
                "number_of_events": n_events,
                "survival_probability": survival_probability,
            }
        )

    return pd.DataFrame(rows)
