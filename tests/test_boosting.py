"""Tests for boosting engines."""

import numpy as np
import pytest

from grnbt.boosting import GradientRegularizedNewtonBoosting, VanillaNewtonBoosting
from grnbt.losses import CharbonnierLoss, MSELoss


def test_vanilla_newton_runs(synthetic_regression):
    """Vanilla Newton must fit and predict with correct shapes."""
    x, y = synthetic_regression
    model = VanillaNewtonBoosting(
        loss=MSELoss(),
        n_estimators=5,
        learning_rate=1.0,
        max_depth=2,
        lam_base=0.0,
    )
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == y.shape


def test_grn_adaptive_lambda_increases(synthetic_regression):
    """GRN must record adaptive λ_k > 0 for Charbonnier loss."""
    x, y = synthetic_regression
    model = GradientRegularizedNewtonBoosting(
        loss=CharbonnierLoss(),
        n_estimators=5,
        learning_rate=1.0,
        max_depth=2,
        lam_base=0.0,
    )
    model.fit(x, y)
    lam_history = model.history.get("lambda_k")
    assert len(lam_history) == 5
    assert any(lam > 0 for lam in lam_history)


def test_loss_decreases_for_strongly_convex(synthetic_regression):
    """MSE is strongly convex; vanilla Newton with small η should decrease loss."""
    x, y = synthetic_regression
    model = VanillaNewtonBoosting(
        loss=MSELoss(),
        n_estimators=10,
        learning_rate=0.5,
        max_depth=3,
        lam_base=0.1,
    )
    model.fit(x, y)
    losses = model.history.get("loss")
    assert losses[0] > losses[-1]


def test_predict_before_fit_raises():
    """Predicting before fit must raise a clear error."""
    model = VanillaNewtonBoosting(loss=MSELoss(), n_estimators=3)
    with pytest.raises(ValueError):
        model.predict(np.zeros((4, 2)))


def test_boosting_invalid_loss_type():
    """Boosting must reject non-Loss instances."""
    with pytest.raises(TypeError):
        VanillaNewtonBoosting(loss="mse")


def test_boosting_invalid_n_estimators():
    """Boosting must validate n_estimators."""
    with pytest.raises(ValueError):
        VanillaNewtonBoosting(loss=MSELoss(), n_estimators=0)
    with pytest.raises(ValueError):
        VanillaNewtonBoosting(loss=MSELoss(), n_estimators=-1)


def test_boosting_invalid_learning_rate():
    """Boosting must validate learning_rate."""
    with pytest.raises(ValueError):
        VanillaNewtonBoosting(loss=MSELoss(), learning_rate=0.0)
    with pytest.raises(ValueError):
        VanillaNewtonBoosting(loss=MSELoss(), learning_rate=-0.1)


def test_boosting_invalid_max_depth():
    """Boosting must validate max_depth."""
    with pytest.raises(ValueError):
        VanillaNewtonBoosting(loss=MSELoss(), max_depth=-1)


def test_boosting_invalid_lam_base():
    """Boosting must validate lam_base."""
    with pytest.raises(ValueError):
        VanillaNewtonBoosting(loss=MSELoss(), lam_base=-1.0)


def test_boosting_empty_data_raises():
    """Boosting must raise on empty data."""
    model = VanillaNewtonBoosting(loss=MSELoss())
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([]))


def test_boosting_zero_features_raises():
    """Boosting must raise on data with zero features."""
    model = VanillaNewtonBoosting(loss=MSELoss())
    with pytest.raises(ValueError):
        model.fit(np.empty((5, 0)), np.ones(5))


def test_boosting_nan_inputs_raises():
    """Boosting must raise on NaN inputs."""
    model = VanillaNewtonBoosting(loss=MSELoss())
    with pytest.raises(ValueError):
        model.fit(np.array([[np.nan, 1.0]]), np.array([1.0]))


def test_boosting_predict_2d_required():
    """Predict must require 2-D input."""
    model = VanillaNewtonBoosting(loss=MSELoss(), n_estimators=1)
    model.fit(np.ones((3, 2)), np.ones(3))
    with pytest.raises(ValueError):
        model.predict(np.ones(3))


def test_boosting_grn_lam_base_zero_mse():
    """GRN with MSE (M_0=0) should have zero adaptive component."""
    x = np.random.RandomState(0).randn(20, 3)
    y = x[:, 0] + np.random.RandomState(0).randn(20) * 0.1
    model = GradientRegularizedNewtonBoosting(
        loss=MSELoss(), n_estimators=3, max_depth=1, lam_base=0.5
    )
    model.fit(x, y)
    # MSE has M_0=0, so adaptive lambda should be 0; only lam_base remains
    for lam in model.history.get("lambda_k"):
        assert np.isclose(lam, 0.5)


def test_boosting_history_keys():
    """History must record expected keys."""
    x, y = np.random.RandomState(0).randn(10, 2), np.ones(10)
    model = VanillaNewtonBoosting(loss=MSELoss(), n_estimators=3)
    model.fit(x, y)
    assert "loss" in model.history.keys()
    assert "lambda_k" in model.history.keys()
    assert "grad_norm" in model.history.keys()


def test_boosting_single_estimator():
    """Boosting with n_estimators=1 must run successfully."""
    x = np.random.RandomState(1).randn(8, 2)
    y = x[:, 0]
    model = VanillaNewtonBoosting(loss=MSELoss(), n_estimators=1, max_depth=1)
    model.fit(x, y)
    assert len(model.trees) == 1


def test_boosting_predict_new_samples():
    """Predict must work on a different number of samples than training."""
    rng = np.random.RandomState(2)
    x_train = rng.randn(50, 3)
    y_train = x_train[:, 0]
    x_test = rng.randn(10, 3)
    model = VanillaNewtonBoosting(loss=MSELoss(), n_estimators=5, max_depth=2)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    assert preds.shape == (10,)
