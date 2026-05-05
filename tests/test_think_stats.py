import numpy as np
import pandas as pd
import pytest

from solutions.think_stats import (
    bootstrap_confidence_interval,
    build_cdf,
    build_design_matrix,
    build_pmf,
    chi_square_statistic,
    column_summary_report,
    compare_pmfs,
    conditional_mean_table,
    covariance,
    density_area_is_close,
    exponential_rate,
    fit_regression_with_categoricals,
    frequency_table,
    group_comparison_summary,
    histogram_bin_counts,
    kernel_density_grid,
    max_cdf_difference,
    normal_model_quantiles,
    pearson_correlation,
    percentile_rank,
    permutation_test_difference_in_means,
    pmf_mean,
    pmf_variance,
    residual_diagnostics_table,
    simple_least_squares_fit,
    simulate_two_group_power,
    simulated_standard_error,
    spearman_correlation,
    survival_curve,
    value_at_percentile,
)


def test_column_summary_report():
    data = pd.DataFrame(
        {
            "age": [10, 20, np.nan],
            "city": ["A", "A", "B"],
        }
    )

    result = column_summary_report(data)
    age_row = result.set_index("column").loc["age"]

    assert age_row["dtype"] == "float64"
    assert age_row["count"] == 2
    assert age_row["missing_count"] == 1
    assert age_row["missing_fraction"] == pytest.approx(1 / 3)
    assert age_row["n_unique"] == 2
    assert age_row["examples"] == [10.0, 20.0]


def test_column_summary_report_rejects_bad_example_count():
    with pytest.raises(ValueError):
        column_summary_report(pd.DataFrame({"x": [1]}), example_count=-1)


def test_group_comparison_summary():
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1, 3, 10, 14],
        }
    )

    result = group_comparison_summary(data, "group", "value")
    group_a = result.set_index("group").loc["A"]

    assert group_a["count"] == 2
    assert group_a["mean"] == pytest.approx(2)
    assert group_a["median"] == pytest.approx(2)
    assert group_a["std"] == pytest.approx(np.sqrt(2))
    assert group_a["min"] == pytest.approx(1)
    assert group_a["max"] == pytest.approx(3)
    assert group_a["iqr"] == pytest.approx(1)


def test_group_comparison_summary_rejects_non_numeric_value():
    data = pd.DataFrame({"group": ["A"], "value": ["high"]})

    with pytest.raises(ValueError):
        group_comparison_summary(data, "group", "value")


def test_frequency_table():
    result = frequency_table(pd.Series(["b", "a", "b", "c", "a", "b"]))

    assert result["value"].tolist() == ["b", "a", "c"]
    assert result["count"].tolist() == [3, 2, 1]
    assert result["proportion"].tolist() == pytest.approx([0.5, 1 / 3, 1 / 6])


def test_frequency_table_rejects_empty_series():
    with pytest.raises(ValueError):
        frequency_table(pd.Series([], dtype=object))


def test_histogram_bin_counts():
    result = histogram_bin_counts([0.2, 0.7, 1.5, 2.0, 2.8], [0, 1, 2, 3])

    assert result["bin"].tolist() == ["[0, 1)", "[1, 2)", "[2, 3]"]
    assert result["count"].tolist() == [2, 1, 2]
    assert result["proportion"].tolist() == pytest.approx([0.4, 0.2, 0.4])


def test_histogram_bin_counts_rejects_unsorted_edges():
    with pytest.raises(ValueError):
        histogram_bin_counts([1, 2, 3], [0, 2, 1])


def test_build_pmf():
    result = build_pmf(["a", "b", "a"])

    assert result == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


def test_build_pmf_rejects_empty_values():
    with pytest.raises(ValueError):
        build_pmf([])


def test_pmf_mean_and_variance():
    pmf = {0: 0.25, 2: 0.75}

    assert pmf_mean(pmf) == pytest.approx(1.5)
    assert pmf_variance(pmf) == pytest.approx(0.75)


def test_pmf_mean_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        pmf_mean({0: 0.5, 1: 0.6})


def test_compare_pmfs():
    result = compare_pmfs(
        {"heads": 0.5, "tails": 0.5},
        {"heads": 0.75, "tails": 0.25},
    )

    heads = result.set_index("outcome").loc["heads"]

    assert heads["first_probability"] == pytest.approx(0.5)
    assert heads["second_probability"] == pytest.approx(0.75)
    assert heads["probability_difference"] == pytest.approx(-0.25)


def test_compare_pmfs_rejects_different_outcomes():
    with pytest.raises(ValueError):
        compare_pmfs({"a": 1.0}, {"b": 1.0})


def test_build_cdf():
    result = build_cdf([3, 1, 1, 2])

    assert result["value"].tolist() == [1, 2, 3]
    assert result["cumulative_probability"].tolist() == pytest.approx([0.5, 0.75, 1])


def test_percentile_rank():
    assert percentile_rank([10, 20, 30, 40], 25) == pytest.approx(50)


def test_value_at_percentile():
    assert value_at_percentile([0, 10, 20, 30], 50) == pytest.approx(15)


def test_value_at_percentile_rejects_bad_percentile():
    with pytest.raises(ValueError):
        value_at_percentile([1, 2, 3], 101)


def test_normal_model_quantiles():
    result = normal_model_quantiles([1, 2, 3], [0.5])

    assert result.loc[0, "probability"] == pytest.approx(0.5)
    assert result.loc[0, "quantile"] == pytest.approx(2)
    assert result.loc[0, "mean"] == pytest.approx(2)
    assert result.loc[0, "std"] == pytest.approx(1)


def test_normal_model_quantiles_rejects_constant_values():
    with pytest.raises(ValueError):
        normal_model_quantiles([1, 1, 1], [0.5])


def test_exponential_rate():
    assert exponential_rate([1, 2, 3]) == pytest.approx(0.5)


def test_exponential_rate_rejects_negative_values():
    with pytest.raises(ValueError):
        exponential_rate([1, -1, 3])


def test_max_cdf_difference():
    result = max_cdf_difference([1, 2], lambda value: value / 2)

    assert result == pytest.approx(0)


def test_max_cdf_difference_rejects_invalid_model_probability():
    with pytest.raises(ValueError):
        max_cdf_difference([1, 2], lambda value: value)


def test_kernel_density_grid():
    result = kernel_density_grid([0, 0], x_values=[0], bandwidth=1)

    assert result["x"].tolist() == [0]
    assert result.loc[0, "density"] == pytest.approx(0.39894228)


def test_kernel_density_grid_rejects_bad_bandwidth():
    with pytest.raises(ValueError):
        kernel_density_grid([0, 1], bandwidth=0)


def test_density_area_is_close():
    assert density_area_is_close([0, 1, 2], [0.5, 0.5, 0.5])
    assert not density_area_is_close([0, 1, 2], [0.25, 0.25, 0.25])


def test_density_area_is_close_rejects_length_mismatch():
    with pytest.raises(ValueError):
        density_area_is_close([0, 1], [0.5])


def test_covariance_and_pearson_correlation():
    first = [1, 2, 3]
    second = [2, 4, 6]

    assert covariance(first, second) == pytest.approx(2)
    assert pearson_correlation(first, second) == pytest.approx(1)


def test_pearson_correlation_rejects_zero_variance():
    with pytest.raises(ValueError):
        pearson_correlation([1, 1, 1], [1, 2, 3])


def test_spearman_correlation_with_ties():
    assert spearman_correlation([10, 20, 20, 30], [1, 2, 2, 5]) == pytest.approx(1)


def test_conditional_mean_table():
    result = conditional_mean_table(
        [0.2, 0.8, 1.2, 2.5],
        [1, 3, 5, 7],
        [0, 1, 2, 3],
    )

    assert result["bin"].tolist() == ["[0, 1)", "[1, 2)", "[2, 3]"]
    assert result["count"].tolist() == [2, 1, 1]
    assert result["mean"].tolist() == pytest.approx([2, 5, 7])


def test_bootstrap_confidence_interval():
    result = bootstrap_confidence_interval(
        [1, 2, 3, 4, 5],
        np.mean,
        n_resamples=200,
        confidence_level=0.80,
        seed=2026,
    )

    assert result["statistic"] == pytest.approx(3)
    assert result["lower"] <= result["statistic"] <= result["upper"]
    assert result["confidence_level"] == pytest.approx(0.80)


def test_bootstrap_confidence_interval_rejects_bad_confidence_level():
    with pytest.raises(ValueError):
        bootstrap_confidence_interval([1, 2, 3], np.mean, confidence_level=1.0)


def test_simulated_standard_error():
    result = simulated_standard_error(
        [1, 2, 3, 4],
        np.mean,
        sample_size=2,
        n_simulations=200,
        seed=2026,
    )

    assert result > 0


def test_simulated_standard_error_rejects_bad_simulation_count():
    with pytest.raises(ValueError):
        simulated_standard_error([1, 2, 3], np.mean, sample_size=2, n_simulations=1)


def test_permutation_test_difference_in_means_exact():
    result = permutation_test_difference_in_means([1, 2], [10, 11])

    assert result["observed_difference"] == pytest.approx(-9)
    assert result["p_value"] == pytest.approx(1 / 3)
    assert result["n_permutations"] == 6


def test_permutation_test_difference_in_means_rejects_bad_permutation_count():
    with pytest.raises(ValueError):
        permutation_test_difference_in_means([1, 2], [3, 4], n_permutations=0)


def test_chi_square_statistic():
    assert chi_square_statistic([8, 12], [10, 10]) == pytest.approx(0.8)


def test_chi_square_statistic_rejects_nonpositive_expected_counts():
    with pytest.raises(ValueError):
        chi_square_statistic([8, 12], [10, 0])


def test_simulate_two_group_power():
    result = simulate_two_group_power(
        effect_size=2,
        sample_size=30,
        noise_sd=1,
        n_simulations=200,
        seed=2026,
    )

    assert result > 0.95


def test_simulate_two_group_power_rejects_bad_noise():
    with pytest.raises(ValueError):
        simulate_two_group_power(effect_size=1, sample_size=10, noise_sd=0)


def test_simple_least_squares_fit():
    result = simple_least_squares_fit([0, 1, 2], [1, 3, 5])

    assert result["intercept"] == pytest.approx(1)
    assert result["slope"] == pytest.approx(2)


def test_simple_least_squares_fit_rejects_constant_x():
    with pytest.raises(ValueError):
        simple_least_squares_fit([1, 1, 1], [1, 2, 3])


def test_residual_diagnostics_table():
    result = residual_diagnostics_table([0, 1, 2], [1, 3, 5], intercept=1, slope=2)

    assert result["fitted"].tolist() == pytest.approx([1, 3, 5])
    assert result["residual"].tolist() == pytest.approx([0, 0, 0])
    assert result["squared_residual"].tolist() == pytest.approx([0, 0, 0])


def test_build_design_matrix():
    data = pd.DataFrame({"x": [1, 2], "z": [3, 4]})

    result = build_design_matrix(data, ["x", "z"])

    np.testing.assert_allclose(result, np.array([[1, 1, 3], [1, 2, 4]]))


def test_build_design_matrix_rejects_missing_column():
    with pytest.raises(ValueError):
        build_design_matrix(pd.DataFrame({"x": [1]}), ["z"])


def test_fit_regression_with_categoricals():
    data = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "group": ["A", "A", "B", "B"],
            "y": [1, 3, 4, 6],
        }
    )

    result = fit_regression_with_categoricals(
        data,
        target_col="y",
        numeric_columns=["x"],
        categorical_columns=["group"],
    )
    coefficients = result["coefficients"]

    assert coefficients["intercept"] == pytest.approx(1)
    assert coefficients["x"] == pytest.approx(2)
    assert coefficients["group_B"] == pytest.approx(3)
    np.testing.assert_allclose(result["predictions"], data["y"].to_numpy())


def test_fit_regression_with_categoricals_rejects_missing_target():
    with pytest.raises(ValueError):
        fit_regression_with_categoricals(
            pd.DataFrame({"x": [1]}),
            target_col="y",
            numeric_columns=["x"],
            categorical_columns=[],
        )


def test_survival_curve():
    result = survival_curve([1, 2, 2, 3], [1, 1, 0, 1])

    assert result["time"].tolist() == [1, 2, 3]
    assert result["number_at_risk"].tolist() == [4, 3, 1]
    assert result["number_of_events"].tolist() == [1, 1, 1]
    assert result["survival_probability"].tolist() == pytest.approx([0.75, 0.5, 0])


def test_survival_curve_rejects_length_mismatch():
    with pytest.raises(ValueError):
        survival_curve([1, 2], [1])
