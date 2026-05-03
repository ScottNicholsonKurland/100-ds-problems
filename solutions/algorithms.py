"""Reference solutions for the algorithms problem set."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gradient_descent(
    f: Callable[[NDArray[np.float64]], float],
    df: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x_0: ArrayLike,
    learning_rate: float = 0.01,
    tolerance: float = 0.01,
    max_iter: int = 10_000,
):
    """Generate points from gradient descent.

    The generator yields the initial point first, then each updated point.

    Convergence is defined by the step size:

        ||x_next - x_current|| < tolerance
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    if max_iter < 1:
        raise ValueError("max_iter must be positive.")

    current = np.asarray(x_0, dtype=float)
    yield current.copy()

    for _ in range(max_iter):
        gradient = np.asarray(df(current), dtype=float)
        next_point = current - learning_rate * gradient

        yield next_point.copy()

        if np.linalg.norm(next_point - current) < tolerance:
            return

        current = next_point

    raise RuntimeError("gradient_descent did not converge within max_iter.")


def newton_method(
    f: Callable[[NDArray[np.float64]], float],
    df: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ddf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x_0: ArrayLike,
    tolerance: float = 1e-6,
    max_iter: int = 100,
):
    """Generate points from Newton's method.

    The function ``f`` is accepted for consistency with the problem statement,
    but Newton's update only requires the gradient ``df`` and Hessian ``ddf``.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    if max_iter < 1:
        raise ValueError("max_iter must be positive.")

    current = np.asarray(x_0, dtype=float)
    yield current.copy()

    for _ in range(max_iter):
        gradient = np.asarray(df(current), dtype=float)
        hessian = np.asarray(ddf(current), dtype=float)
        step = np.linalg.solve(hessian, gradient)
        next_point = current - step

        yield next_point.copy()

        if np.linalg.norm(next_point - current) < tolerance:
            return

        current = next_point

    raise RuntimeError("newton_method did not converge within max_iter.")


class LinearRegression:
    """Ordinary least-squares linear regression using NumPy."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coeffs_: NDArray[np.float64] | None = None

    def _prepare_features(self, x: ArrayLike) -> NDArray[np.float64]:
        x_array = np.asarray(x, dtype=float)

        if x_array.ndim == 1:
            x_array = x_array.reshape(-1, 1)

        if x_array.ndim != 2:
            raise ValueError("X must be one-dimensional or two-dimensional.")

        if not self.fit_intercept:
            return x_array

        intercept_column = np.ones((x_array.shape[0], 1))
        return np.column_stack((intercept_column, x_array))

    def fit(self, x: ArrayLike, y: ArrayLike) -> LinearRegression:
        x_design = self._prepare_features(x)
        y_array = np.asarray(y, dtype=float)

        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        if x_design.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.coeffs_ = np.linalg.lstsq(x_design, y_array, rcond=None)[0]

        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.coeffs_ is None:
            raise ValueError("Model must be fit before prediction.")

        x_design = self._prepare_features(x)

        return x_design @ self.coeffs_


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the logistic sigmoid stably enough for small practice problems."""
    clipped_values = np.clip(values, -500, 500)

    return 1 / (1 + np.exp(-clipped_values))


def _prepare_logistic_features(x: ArrayLike) -> NDArray[np.float64]:
    """Convert features to a two-dimensional array with intercept column."""
    x_array = np.asarray(x, dtype=float)

    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)

    if x_array.ndim != 2:
        raise ValueError("X must be one-dimensional or two-dimensional.")

    intercept_column = np.ones((x_array.shape[0], 1))

    return np.column_stack((intercept_column, x_array))


def _fit_logistic_regression(
    x: ArrayLike,
    y: ArrayLike,
    learning_rate: float = 0.1,
    tolerance: float = 1e-8,
    max_iter: int = 10_000,
) -> NDArray[np.float64]:
    """Fit binary logistic regression with batch gradient descent."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    if max_iter < 1:
        raise ValueError("max_iter must be positive.")

    x_design = _prepare_logistic_features(x)
    y_array = np.asarray(y, dtype=float)

    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    if x_design.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    unique_labels = set(np.unique(y_array))
    if not unique_labels <= {0.0, 1.0}:
        raise ValueError("y must contain only binary labels 0 and 1.")

    weights = np.zeros(x_design.shape[1], dtype=float)

    for _ in range(max_iter):
        probabilities = _sigmoid(x_design @ weights)
        gradient = x_design.T @ (probabilities - y_array) / len(y_array)
        next_weights = weights - learning_rate * gradient

        if np.linalg.norm(next_weights - weights) < tolerance:
            return next_weights

        weights = next_weights

    return weights


def _predict_logistic_probabilities(
    x: ArrayLike,
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict positive-class probabilities from fitted logistic weights."""
    x_design = _prepare_logistic_features(x)

    if x_design.shape[1] != weights.shape[0]:
        raise ValueError("X has the wrong number of columns for these weights.")

    return _sigmoid(x_design @ weights)


def logistic_accuracy(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    y_test: ArrayLike,
    thres: float = 0.5,
) -> float:
    """Fit logistic regression and return thresholded classification accuracy."""
    if not 0 <= thres <= 1:
        raise ValueError("thres must be between 0 and 1.")

    weights = _fit_logistic_regression(x_train, y_train)
    probabilities = _predict_logistic_probabilities(x_test, weights)
    predictions = (probabilities >= thres).astype(int)
    y_test_array = np.asarray(y_test, dtype=int)

    if predictions.shape != y_test_array.shape:
        raise ValueError("Predictions and y_test must have the same shape.")

    return float(np.mean(predictions == y_test_array))


def logistic_profit(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    y_test: ArrayLike,
    profit_matrix: ArrayLike,
) -> float:
    """Fit logistic regression and return the best thresholded test-set profit.

    ``profit_matrix[actual_class, predicted_class]`` gives the profit/cost for
    each classification outcome.
    """
    profit_array = np.asarray(profit_matrix, dtype=float)

    if profit_array.shape != (2, 2):
        raise ValueError("profit_matrix must have shape (2, 2).")

    weights = _fit_logistic_regression(x_train, y_train)
    probabilities = _predict_logistic_probabilities(x_test, weights)
    y_test_array = np.asarray(y_test, dtype=int)

    if not set(np.unique(y_test_array)) <= {0, 1}:
        raise ValueError("y_test must contain only binary labels 0 and 1.")

    thresholds = np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                probabilities,
                np.array([np.nextafter(1.0, 2.0)]),
            ),
        ),
    )

    best_profit = float("-inf")

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        profit = sum(
            profit_array[actual, predicted]
            for actual, predicted in zip(y_test_array, predictions, strict=True)
        )
        best_profit = max(best_profit, float(profit))

    return best_profit


class PWLinearRegression:
    """Piecewise linear regression with continuous knots."""

    def __init__(self, knots: ArrayLike):
        self.knots = np.asarray(knots, dtype=float)
        self.coeffs_: NDArray[np.float64] | None = None

    def _design_matrix(self, x: ArrayLike) -> NDArray[np.float64]:
        x_array = np.asarray(x, dtype=float).reshape(-1)
        columns = [np.ones_like(x_array), x_array]

        for knot in self.knots:
            columns.append(np.maximum(0, x_array - knot))

        return np.column_stack(columns)

    def fit(self, x: ArrayLike, y: ArrayLike) -> PWLinearRegression:
        x_design = self._design_matrix(x)
        y_array = np.asarray(y, dtype=float)

        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        if x_design.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must have the same number of rows.")

        self.coeffs_ = np.linalg.lstsq(x_design, y_array, rcond=None)[0]

        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.coeffs_ is None:
            raise ValueError("Model must be fit before prediction.")

        return self._design_matrix(x) @ self.coeffs_


class HurdleModel:
    """Two-stage model for outcomes with many exact zeros.

    Stage 1 predicts whether the target is nonzero with logistic regression.
    Stage 2 predicts the target value among nonzero observations with linear
    regression. The final prediction is:

        P(nonzero | X) * E(y | X, y != 0)
    """

    def __init__(self):
        self.logistic_weights_: NDArray[np.float64] | None = None
        self.positive_model_: LinearRegression | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> HurdleModel:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)

        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        if len(x_array) != len(y_array):
            raise ValueError("X and y must have the same number of rows.")

        nonzero_mask = y_array != 0

        if not nonzero_mask.any():
            raise ValueError("At least one y value must be nonzero.")

        if nonzero_mask.all():
            raise ValueError("At least one y value must be exactly zero.")

        self.logistic_weights_ = _fit_logistic_regression(
            x_array,
            nonzero_mask.astype(int),
        )
        self.positive_model_ = LinearRegression().fit(
            x_array[nonzero_mask],
            y_array[nonzero_mask],
        )

        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.logistic_weights_ is None or self.positive_model_ is None:
            raise ValueError("Model must be fit before prediction.")

        nonzero_probabilities = _predict_logistic_probabilities(
            x,
            self.logistic_weights_,
        )
        positive_predictions = self.positive_model_.predict(x)

        return nonzero_probabilities * positive_predictions
