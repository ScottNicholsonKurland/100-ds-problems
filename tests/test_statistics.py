import pytest

from solutions.statistics import binomial_exact_test, test_for_cheating


def test_binomial_exact_test():
    result = binomial_exact_test(
        n_samples=10,
        n_positive_samples=8,
        proposed_value=0.5,
    )

    assert result == pytest.approx(56 / 1024)


def test_binomial_exact_test_with_zero_positive_samples():
    result = binomial_exact_test(
        n_samples=10,
        n_positive_samples=0,
        proposed_value=0.5,
    )

    assert result == pytest.approx(1.0)


def test_binomial_exact_test_rejects_bad_sample_count():
    with pytest.raises(ValueError):
        binomial_exact_test(
            n_samples=0,
            n_positive_samples=0,
            proposed_value=0.5,
        )


def test_binomial_exact_test_rejects_bad_positive_count():
    with pytest.raises(ValueError):
        binomial_exact_test(
            n_samples=10,
            n_positive_samples=11,
            proposed_value=0.5,
        )


def test_binomial_exact_test_rejects_bad_proposed_value():
    with pytest.raises(ValueError):
        binomial_exact_test(
            n_samples=10,
            n_positive_samples=8,
            proposed_value=1.5,
        )


def test_for_cheating_example_scores_not_suspicious():
    result = test_for_cheating(
        {
            "STR": 11,
            "DEX": "14",
            "CON": 10,
            "INT": 14,
            "WIS": 9,
            "CHR": 16,
        },
    )

    assert result["total_score"] == 74
    assert result["mean_score"] == pytest.approx(74 / 6)
    assert result["p_value"] == pytest.approx(0.07404161442743634)
    assert result["suspicious"] is False


def test_for_cheating_flags_extremely_high_scores():
    result = test_for_cheating(
        {
            "STR": 18,
            "DEX": 18,
            "CON": 18,
            "INT": 18,
            "WIS": 18,
            "CHR": 18,
        },
    )

    assert result["total_score"] == 108
    assert result["p_value"] < 0.001
    assert result["suspicious"] is True


def test_for_cheating_rejects_wrong_number_of_attributes():
    with pytest.raises(ValueError):
        test_for_cheating({"STR": 10, "DEX": 10})


def test_for_cheating_rejects_invalid_attribute_score():
    with pytest.raises(ValueError):
        test_for_cheating(
            {
                "STR": 19,
                "DEX": 10,
                "CON": 10,
                "INT": 10,
                "WIS": 10,
                "CHR": 10,
            },
        )
