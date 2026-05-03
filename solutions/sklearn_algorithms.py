"""Scikit-learn reference solutions for selected algorithm problems."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LinearRegression, LogisticRegression


def _as_2d_features(x: ArrayLike) -> NDArray[np.float64]:
    """Convert feature input to a two-dimensional float array."""
    x_array = np.asarray(x, dtype=float)

    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)

    if x_array.ndim != 2:
        raise ValueError("X must be one-dimensional or two-dimensional.")

    return x_array


def _as_binary_target(y: ArrayLike, name: str = "y") -> NDArray[np.int_]:
    """Validate and return a one-dimensional binary target array."""
    y_array = np.asarray(y, dtype=int)

    if y_array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    unique_values = set(np.unique(y_array))
    if not unique_values <= {0, 1}:
        raise ValueError(f"{name} must contain only binary labels 0 and 1.")

    return y_array


def _positive_class_probabilities(
    model: LogisticRegression,
    x: ArrayLike,
) -> NDArray[np.float64]:
    """Return predicted probabilities for class 1."""
    class_positions = np.where(model.classes_ == 1)[0]

    if len(class_positions) != 1:
        raise ValueError("Model does not contain positive class label 1.")

    positive_class_index = int(class_positions[0])

    return model.predict_proba(_as_2d_features(x))[:, positive_class_index]


def fit_logistic_classifier(
    x_train: ArrayLike,
    y_train: ArrayLike,
    max_iter: int = 1_000,
) -> LogisticRegression:
    """Fit a binary logistic-regression classifier with scikit-learn."""
    x_train_array = _as_2d_features(x_train)
    y_train_array = _as_binary_target(y_train, name="y_train")

    if x_train_array.shape[0] != y_train_array.shape[0]:
        raise ValueError("x_train and y_train must have the same number of rows.")

    if len(np.unique(y_train_array)) != 2:
        raise ValueError("y_train must contain both classes 0 and 1.")

    model = LogisticRegression(max_iter=max_iter)
    model.fit(x_train_array, y_train_array)

    return model


def logistic_accuracy_sklearn(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    y_test: ArrayLike,
    thres: float = 0.5,
) -> float:
    """Fit logistic regression and return thresholded classification accuracy."""
    if not 0 <= thres <= 1:
        raise ValueError("thres must be between 0 and 1.")

    model = fit_logistic_classifier(x_train, y_train)
    probabilities = _positive_class_probabilities(model, x_test)
    predictions = (probabilities >= thres).astype(int)
    y_test_array = _as_binary_target(y_test, name="y_test")

    if predictions.shape != y_test_array.shape:
        raise ValueError("Predictions and y_test must have the same shape.")

    return float(np.mean(predictions == y_test_array))


def logistic_profit_sklearn(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    y_test: ArrayLike,
    profit_matrix: ArrayLike,
) -> float:
    """Fit logistic regression and return the best thresholded test-set profit.

    ``profit_matrix[actual_class, predicted_class]`` gives the profit or cost
    for each classification outcome.
    """
    profit_array = np.asarray(profit_matrix, dtype=float)

    if profit_array.shape != (2, 2):
        raise ValueError("profit_matrix must have shape (2, 2).")

    model = fit_logistic_classifier(x_train, y_train)
    probabilities = _positive_class_probabilities(model, x_test)
    y_test_array = _as_binary_target(y_test, name="y_test")

    if probabilities.shape != y_test_array.shape:
        raise ValueError("Predictions and y_test must have the same shape.")

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


class SklearnHurdleModel:
    """Two-stage hurdle model using scikit-learn estimators.

    Stage 1:
        Logistic regression predicts whether y is nonzero.

    Stage 2:
        Linear regression predicts y among nonzero observations.

    Final prediction:
        P(y != 0 | X) * E(y | X, y != 0)
    """

    def __init__(self, max_iter: int = 1_000):
        self.max_iter = max_iter
        self.classifier_: LogisticRegression | None = None
        self.regressor_: LinearRegression | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> SklearnHurdleModel:
        """Fit the classifier and positive-outcome regressor."""
        x_array = _as_2d_features(x)
        y_array = np.asarray(y, dtype=float)

        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        nonzero_mask = y_array != 0

        if not nonzero_mask.any():
            raise ValueError("At least one y value must be nonzero.")

        if nonzero_mask.all():
            raise ValueError("At least one y value must be exactly zero.")

        self.classifier_ = LogisticRegression(max_iter=self.max_iter)
        self.classifier_.fit(x_array, nonzero_mask.astype(int))

        self.regressor_ = LinearRegression()
        self.regressor_.fit(x_array[nonzero_mask], y_array[nonzero_mask])

        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        """Predict the expected target value."""
        if self.classifier_ is None or self.regressor_ is None:
            raise ValueError("Model must be fit before prediction.")

        x_array = _as_2d_features(x)
        nonzero_probabilities = _positive_class_probabilities(self.classifier_, x_array)
        positive_predictions = self.regressor_.predict(x_array)

        return nonzero_probabilities * positive_predictions
