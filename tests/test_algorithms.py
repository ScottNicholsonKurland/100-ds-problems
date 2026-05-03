import numpy as np
import pytest

from solutions.algorithms import (
    HurdleModel,
    LinearRegression,
    PWLinearRegression,
    gradient_descent,
    logistic_accuracy,
    logistic_profit,
    newton_method,
)


def test_gradient_descent_converges_on_quadratic():
    def f(x):
        return float(np.sum(x**2))

    def df(x):
        return 2 * x

    points = list(
        gradient_descent(
            f=f,
            df=df,
            x_0=np.array([10.0]),
            learning_rate=0.1,
            tolerance=1e-6,
        ),
    )

    assert points[0] == pytest.approx(np.array([10.0]))
    assert points[-1] == pytest.approx(np.array([0.0]), abs=1e-4)


def test_gradient_descent_rejects_bad_learning_rate():
    def f(x):
        return float(np.sum(x**2))

    def df(x):
        return 2 * x

    with pytest.raises(ValueError):
        list(
            gradient_descent(
                f=f,
                df=df,
                x_0=np.array([1.0]),
                learning_rate=0,
            ),
        )


def test_newton_method_converges_on_quadratic_in_one_step():
    def f(x):
        return float(np.sum(x**2))

    def df(x):
        return 2 * x

    def ddf(x):
        return np.array([[2.0]])

    points = list(
        newton_method(
            f=f,
            df=df,
            ddf=ddf,
            x_0=np.array([10.0]),
        ),
    )

    assert points[0] == pytest.approx(np.array([10.0]))
    assert points[-1] == pytest.approx(np.array([0.0]))


def test_linear_regression_fits_line_with_intercept():
    x = np.array([0, 1, 2, 3])
    y = np.array([2, 5, 8, 11])

    model = LinearRegression().fit(x, y)

    np.testing.assert_allclose(model.coeffs_, np.array([2, 3]))
    np.testing.assert_allclose(model.predict(np.array([4, 5])), np.array([14, 17]))


def test_linear_regression_requires_fit_before_predict():
    model = LinearRegression()

    with pytest.raises(ValueError):
        model.predict(np.array([1, 2, 3]))


def test_logistic_accuracy_on_separable_data():
    x_train = np.array([[-2], [-1], [1], [2]])
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[-3], [3], [0.5], [-0.5]])
    y_test = np.array([0, 1, 1, 0])

    accuracy = logistic_accuracy(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        thres=0.5,
    )

    assert accuracy == pytest.approx(1.0)


def test_logistic_accuracy_rejects_bad_threshold():
    with pytest.raises(ValueError):
        logistic_accuracy(
            x_train=np.array([[-1], [1]]),
            y_train=np.array([0, 1]),
            x_test=np.array([[-1], [1]]),
            y_test=np.array([0, 1]),
            thres=1.5,
        )


def test_logistic_profit_on_separable_data():
    x_train = np.array([[-2], [-1], [1], [2]])
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[-3], [3], [0.5], [-0.5]])
    y_test = np.array([0, 1, 1, 0])
    profit_matrix = np.array(
        [
            [0, -5],
            [-1, 10],
        ],
    )

    profit = logistic_profit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        profit_matrix=profit_matrix,
    )

    assert profit == pytest.approx(20)


def test_logistic_profit_rejects_bad_profit_matrix_shape():
    with pytest.raises(ValueError):
        logistic_profit(
            x_train=np.array([[-1], [1]]),
            y_train=np.array([0, 1]),
            x_test=np.array([[-1], [1]]),
            y_test=np.array([0, 1]),
            profit_matrix=np.array([1, 2]),
        )


def test_piecewise_linear_regression_fits_continuous_hinge_model():
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = 1 + 2 * x + 3 * np.maximum(0, x - 2)

    model = PWLinearRegression(knots=np.array([2.0])).fit(x, y)

    np.testing.assert_allclose(model.predict(x), y, atol=1e-10)
    np.testing.assert_allclose(
        model.predict(np.array([5.0])),
        np.array([1 + 2 * 5 + 3 * (5 - 2)]),
    )


def test_piecewise_linear_regression_requires_fit_before_predict():
    model = PWLinearRegression(knots=np.array([2.0]))

    with pytest.raises(ValueError):
        model.predict(np.array([1, 2, 3]))


def test_hurdle_model_predictions_increase_for_nonzero_region():
    x = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([0, 0, 0, 10, 12, 14], dtype=float)

    model = HurdleModel().fit(x, y)
    predictions = model.predict(np.array([[0], [5]], dtype=float))

    assert predictions.shape == (2,)
    assert predictions[1] > predictions[0]
    assert predictions[1] > 5


def test_hurdle_model_rejects_all_zero_target():
    x = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 0, 0], dtype=float)

    with pytest.raises(ValueError):
        HurdleModel().fit(x, y)


def test_hurdle_model_rejects_all_nonzero_target():
    x = np.array([[0], [1], [2]], dtype=float)
    y = np.array([1, 2, 3], dtype=float)

    with pytest.raises(ValueError):
        HurdleModel().fit(x, y)
