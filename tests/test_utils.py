"""Tests for the small helpers in :mod:`grnbt.utils`.

Covers three concerns:

* :func:`empirical_norm` — unweighted RMS, weighted ``L^2``, and
  strict validation (empty arrays, ``NaN``/``Inf``, negative or
  mis-shaped weights).
* :func:`unique_thresholds` — midpoint computation, empty result
  for fewer than two unique values, and input validation.
* :class:`History` — append/get/keys semantics, deep-copy contract
  of :meth:`History.as_dict`, and rejection of non-string keys
  and non-finite values.
"""

import numpy as np
import pytest

from grnbt.utils import History, empirical_norm, unique_thresholds


def test_empirical_norm_no_weights():
    """Empirical norm without weights is RMS."""
    v = np.array([3.0, 4.0])
    norm = empirical_norm(v)
    expected = np.sqrt((9 + 16) / 2)
    assert np.isclose(norm, expected)


def test_empirical_norm_with_weights():
    """Empirical norm with weights uses sqrt(sum(w * v^2))."""
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 0.0, 1.0])
    norm = empirical_norm(v, w)
    assert np.isclose(norm, np.sqrt(10.0))


def test_empirical_norm_empty_raises():
    """Empirical norm must reject empty vectors."""
    with pytest.raises(ValueError):
        empirical_norm(np.array([]))


def test_empirical_norm_nan_raises():
    """Empirical norm must reject NaN."""
    with pytest.raises(ValueError):
        empirical_norm(np.array([1.0, np.nan]))


def test_empirical_norm_inf_raises():
    """Empirical norm must reject infinite values."""
    with pytest.raises(ValueError):
        empirical_norm(np.array([1.0, np.inf]))


def test_empirical_norm_negative_weights_raises():
    """Empirical norm must reject negative weights."""
    with pytest.raises(ValueError):
        empirical_norm(np.ones(3), np.array([-1.0, 1.0, 1.0]))


def test_empirical_norm_weight_shape_mismatch_raises():
    """Empirical norm must reject weight shape mismatches."""
    with pytest.raises(ValueError):
        empirical_norm(np.ones(3), np.ones(2))


def test_unique_thresholds_basic():
    """Midpoints between sorted unique values."""
    x = np.array([1.0, 3.0, 2.0, 1.0])
    thresh = unique_thresholds(x)
    assert np.allclose(thresh, np.array([1.5, 2.5]))


def test_unique_thresholds_single_value():
    """Single unique value returns empty array."""
    x = np.array([5.0, 5.0, 5.0])
    thresh = unique_thresholds(x)
    assert thresh.size == 0


def test_unique_thresholds_empty_raises():
    """Empty input raises ValueError."""
    with pytest.raises(ValueError):
        unique_thresholds(np.array([]))


def test_unique_thresholds_1d_required():
    """Input must be 1-D."""
    with pytest.raises(ValueError):
        unique_thresholds(np.zeros((2, 2)))


def test_history_basic():
    """History logs and retrieves values."""
    hist = History()
    hist.log("loss", 1.0)
    hist.log("loss", 0.5)
    assert hist.get("loss") == [1.0, 0.5]


def test_history_unknown_key():
    """Unknown key returns empty list."""
    hist = History()
    assert hist.get("unknown") == []


def test_history_non_finite_raises():
    """History rejects non-finite values."""
    hist = History()
    with pytest.raises(ValueError):
        hist.log("loss", np.nan)
    with pytest.raises(ValueError):
        hist.log("loss", np.inf)


def test_history_invalid_key_type_raises():
    """History key must be a string."""
    hist = History()
    with pytest.raises(TypeError):
        hist.log(123, 1.0)


def test_history_empty_key_raises():
    """History key must not be empty."""
    hist = History()
    with pytest.raises(ValueError):
        hist.log("", 1.0)


def test_history_keys_sorted():
    """History.keys() returns sorted metric names."""
    hist = History()
    hist.log("z", 1.0)
    hist.log("a", 2.0)
    assert hist.keys() == ["a", "z"]


def test_history_as_dict():
    """History.as_dict() returns a complete copy."""
    hist = History()
    hist.log("x", 1.0)
    hist.log("x", 2.0)
    d = hist.as_dict()
    assert d == {"x": [1.0, 2.0]}
    # Mutating the dict should not affect history
    d["x"].append(3.0)
    assert hist.get("x") == [1.0, 2.0]
