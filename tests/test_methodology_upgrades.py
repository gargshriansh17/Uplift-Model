"""Tests for new reproducibility and statistical helper functions."""
import numpy as np
import pandas as pd

from src.multiple_comparison import holm_adjust, corrected_resampled_test


def test_holm_adjust_is_monotone_in_sorted_p_values():
    raw = np.array([0.04, 0.001, 0.02])
    adjusted = holm_adjust(raw)
    assert np.all(adjusted >= raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_holm_adjust_single_test_is_identity():
    assert np.allclose(holm_adjust([0.03]), [0.03])


def test_holm_adjust_all_zero_stays_zero():
    assert np.allclose(holm_adjust([0.0, 0.0]), [0.0, 0.0])


def test_corrected_resampled_test_returns_valid_two_sided_p_value():
    statistic, p_value, standard_error = corrected_resampled_test([0.01, 0.02, -0.01, 0.03, 0.00])
    assert np.isfinite(statistic)
    assert 0.0 <= p_value <= 1.0
    assert standard_error > 0
