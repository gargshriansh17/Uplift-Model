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


def test_hc3_robust_score_test_returns_full_rank_omnibus_result():
    from src.heterogeneity_test import _hc3_robust_score_test

    rng = np.random.default_rng(42)
    n = 3000
    treatment = rng.binomial(1, 0.5, size=n)
    features = rng.normal(size=(n, 3))
    restricted = np.column_stack([np.ones(n), features, treatment])
    interactions = features * treatment[:, None]

    probabilities = 1.0 / (
        1.0 + np.exp(-(-3.2 + 0.25 * features[:, 0] + 0.35 * treatment))
    )
    outcome = rng.binomial(1, probabilities)

    result = _hc3_robust_score_test(outcome, restricted, interactions)

    assert result["interaction_design_rank"] == 3
    assert result["covariance_rank"] == 3
    assert result["degrees_freedom"] == 3
    assert np.isfinite(result["statistic"])
    assert 0.0 <= result["p_value"] <= 1.0


def test_hc3_robust_score_test_reduces_duplicate_interactions_to_estimable_rank():
    from src.heterogeneity_test import _hc3_robust_score_test

    rng = np.random.default_rng(7)
    n = 1500
    treatment = rng.binomial(1, 0.5, size=n)
    feature = rng.normal(size=n)
    restricted = np.column_stack([np.ones(n), feature, treatment])
    interaction = feature * treatment
    duplicated_interactions = np.column_stack([interaction, interaction])

    result = _hc3_robust_score_test(
        rng.binomial(1, 0.3, size=n),
        restricted,
        duplicated_interactions,
    )

    assert result["interaction_design_rank"] == 1
    assert result["covariance_rank"] == 1
    assert result["degrees_freedom"] == 1
    assert len(result["selected_interaction_indices"]) == 1
    assert np.isfinite(result["statistic"])
    assert 0.0 <= result["p_value"] <= 1.0



def test_covariate_selection_preserves_intercept_and_treatment_controls():
    from src.heterogeneity_test import _select_covariates_after_controls

    rng = np.random.default_rng(19)
    n = 1200
    treatment = rng.binomial(1, 0.5, size=n).astype(float)
    controls = np.column_stack([np.ones(n), treatment])
    x = rng.normal(size=n)
    candidates = np.column_stack([
        x,
        x,  # duplicate covariate
        treatment,  # aliased with mandatory treatment control
        np.ones(n),  # aliased with intercept
        rng.normal(size=n),
    ])

    selected, selected_indices = _select_covariates_after_controls(
        controls, candidates
    )
    restricted = np.column_stack([controls, selected])

    assert len(selected_indices) == 2
    assert np.linalg.matrix_rank(restricted) == restricted.shape[1]
    assert np.allclose(selected, candidates[:, selected_indices])


def test_covariate_selection_rejects_rank_deficient_mandatory_controls():
    import pytest
    from src.heterogeneity_test import _select_covariates_after_controls

    controls = np.column_stack([np.ones(10), np.ones(10)])
    candidates = np.arange(20, dtype=float).reshape(10, 2)

    with pytest.raises(ValueError, match="Mandatory intercept/treatment controls"):
        _select_covariates_after_controls(controls, candidates)
