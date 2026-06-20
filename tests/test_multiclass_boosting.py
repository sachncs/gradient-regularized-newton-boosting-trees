"""Tests for multi-class boosting engines."""

import numpy as np
import pytest

from grnbt.boosting import MultiClassNewtonBoosting
from grnbt.losses import CategoricalCrossEntropyLoss, MSELoss


def test_multiclass_boosting_runs(synthetic_multiclass):
    """Multi-class boosting must fit and predict with correct shapes."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(
        loss=loss,
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        lam_base=0.01,
        n_classes=n_classes,
    )
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (x.shape[0], n_classes)


def test_multiclass_boosting_probabilities_sum_to_one(synthetic_multiclass):
    """Predicted probabilities must sum to 1 for each sample."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(
        loss=loss,
        n_estimators=10,
        learning_rate=0.1,
        max_depth=2,
        lam_base=0.01,
        n_classes=n_classes,
    )
    model.fit(x, y)
    probs = model.predict_proba(x)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_multiclass_boosting_logits_output(synthetic_multiclass):
    """With softmax_output=False, predict must return logits."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(
        loss=loss,
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        n_classes=n_classes,
        softmax_output=False,
    )
    model.fit(x, y)
    logits = model.predict(x)
    assert logits.shape == (x.shape[0], n_classes)
    # Logits should not sum to 1
    assert not np.allclose(logits.sum(axis=1), 1.0)


def test_multiclass_boosting_loss_decreases(synthetic_multiclass):
    """Loss should decrease over boosting rounds."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(
        loss=loss,
        n_estimators=20,
        learning_rate=0.1,
        max_depth=3,
        lam_base=0.01,
        n_classes=n_classes,
    )
    model.fit(x, y)
    losses = model.history.get("loss")
    assert len(losses) == 20
    assert losses[-1] < losses[0]


def test_multiclass_boosting_adaptive_lambda(synthetic_multiclass):
    """GRN variant must record adaptive lambda > 0."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(
        loss=loss,
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        lam_base=0.0,
        n_classes=n_classes,
    )
    model.fit(x, y)
    lam_history = model.history.get("lambda_k")
    assert len(lam_history) == 5
    assert any(lam > 0 for lam in lam_history)


def test_multiclass_boosting_predict_new_samples(synthetic_multiclass):
    """Predict must work on different number of samples than training."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(
        loss=loss,
        n_estimators=5,
        learning_rate=0.1,
        max_depth=2,
        n_classes=n_classes,
    )
    model.fit(x, y)
    x_test = np.random.RandomState(99).randn(10, x.shape[1])
    preds = model.predict(x_test)
    assert preds.shape == (10, n_classes)


def test_multiclass_boosting_predict_before_fit_raises():
    """Predicting before fit must raise a clear error."""
    loss = CategoricalCrossEntropyLoss(n_classes=3)
    model = MultiClassNewtonBoosting(loss=loss, n_estimators=3, n_classes=3)
    with pytest.raises(ValueError):
        model.predict(np.zeros((4, 2)))


def test_multiclass_boosting_invalid_n_classes():
    """Boosting must validate n_classes."""
    loss = CategoricalCrossEntropyLoss(n_classes=3)
    with pytest.raises(ValueError):
        MultiClassNewtonBoosting(loss=loss, n_classes=1)


def test_multiclass_boosting_invalid_labels_raises(synthetic_multiclass):
    """Boosting must raise on invalid labels."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(loss=loss, n_classes=n_classes)
    bad_y = np.array([0, 1, 99])  # 99 is out of range
    with pytest.raises(ValueError):
        model.fit(x[:3], bad_y)


def test_multiclass_boosting_non_integer_labels_raises(synthetic_multiclass):
    """Boosting must raise on non-integer labels."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(loss=loss, n_classes=n_classes)
    with pytest.raises(TypeError):
        model.fit(x, y.astype(float))


def test_multiclass_boosting_history_keys(synthetic_multiclass):
    """History must record expected keys."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(loss=loss, n_estimators=3, n_classes=n_classes)
    model.fit(x, y)
    assert "loss" in model.history.keys()
    assert "lambda_k" in model.history.keys()
    assert "grad_norm" in model.history.keys()


def test_multiclass_boosting_single_estimator(synthetic_multiclass):
    """Boosting with n_estimators=1 must run successfully."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes=n_classes)
    model = MultiClassNewtonBoosting(
        loss=loss, n_estimators=1, max_depth=1, n_classes=n_classes
    )
    model.fit(x, y)
    assert len(model.trees) == 1


def test_multiclass_boosting_empty_data_raises():
    """Boosting must raise on empty data."""
    loss = CategoricalCrossEntropyLoss(n_classes=3)
    model = MultiClassNewtonBoosting(loss=loss, n_classes=3)
    with pytest.raises(ValueError):
        model.fit(np.empty((0, 2)), np.array([], dtype=int))
