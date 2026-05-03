import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from solutions.sklearn_algorithms import (
    SklearnHurdleModel,
    fit_logistic_classifier,
    logistic_accuracy_sklearn,
    logistic_profit_sklearn,
)


def test_fit_logistic_classifier_returns_fitted_model():
    x_train = np.array([[-3], [-2], [-1], [1], [2], [3]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = fit_logistic_classifier(x_train, y_train)

    assert isinstance(model, LogisticRegression)
    assert set(model.classes_) == {0, 1}


def test_fit_logistic_classifier_rejects_single_class_target():
    x_train = np.array([[-3], [-2], [-1]], dtype=float)
    y_train = np.array([0, 0, 0])

    with pytest.raises(ValueError):
        fit_logistic_classifier(x_train, y_train)


def test_logistic_accuracy_sklearn_on_separable_data():
    x_train = np.array([[-3], [-2], [-1], [1], [2], [3]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_test = np.array([[-4], [4], [0.8], [-0.8]], dtype=float)
    y_test = np.array([0, 1, 1, 0])

    accuracy = logistic_accuracy_sklearn(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        thres=0.5,
    )

    assert accuracy == pytest.approx(1.0)


def test_logistic_accuracy_sklearn_rejects_bad_threshold():
    with pytest.raises(ValueError):
        logistic_accuracy_sklearn(
            x_train=np.array([[-1], [1]], dtype=float),
            y_train=np.array([0, 1]),
            x_test=np.array([[-1], [1]], dtype=float),
            y_test=np.array([0, 1]),
            thres=1.5,
        )


def test_logistic_profit_sklearn_on_separable_data():
    x_train = np.array([[-3], [-2], [-1], [1], [2], [3]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_test = np.array([[-4], [4], [0.8], [-0.8]], dtype=float)
    y_test = np.array([0, 1, 1, 0])
    profit_matrix = np.array(
        [
            [0, -5],
            [-1, 10],
        ],
    )

    profit = logistic_profit_sklearn(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        profit_matrix=profit_matrix,
    )

    assert profit == pytest.approx(20)


def test_logistic_profit_sklearn_rejects_bad_profit_matrix_shape():
    with pytest.raises(ValueError):
        logistic_profit_sklearn(
            x_train=np.array([[-1], [1]], dtype=float),
            y_train=np.array([0, 1]),
            x_test=np.array([[-1], [1]], dtype=float),
            y_test=np.array([0, 1]),
            profit_matrix=np.array([1, 2]),
        )


def test_sklearn_hurdle_model_predictions_increase_for_nonzero_region():
    x = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 10, 12, 14, 16, 18], dtype=float)

    model = SklearnHurdleModel().fit(x, y)
    predictions = model.predict(np.array([[0], [9]], dtype=float))

    assert predictions.shape == (2,)
    assert predictions[1] > predictions[0]
    assert predictions[1] > 10


def test_sklearn_hurdle_model_requires_fit_before_predict():
    model = SklearnHurdleModel()

    with pytest.raises(ValueError):
        model.predict(np.array([[1], [2]], dtype=float))


def test_sklearn_hurdle_model_rejects_all_zero_target():
    x = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 0, 0], dtype=float)

    with pytest.raises(ValueError):
        SklearnHurdleModel().fit(x, y)


def test_sklearn_hurdle_model_rejects_all_nonzero_target():
    x = np.array([[0], [1], [2]], dtype=float)
    y = np.array([1, 2, 3], dtype=float)

    with pytest.raises(ValueError):
        SklearnHurdleModel().fit(x, y)
