"""Global treatment-effect heterogeneity test for the two-arm Hillstrom experiment.

The two-arm development data contains Men's Email (1) and No Email (0).
The null model is a constant-effect linear probability model. The alternative
adds treatment-by-feature interactions, testing whether the conditional
conversion-probability treatment effect varies with observed features.

The joint test is an HC3-robust score test evaluated under the restricted
(constant-effect) model. The intercept and treatment indicator are retained
as mandatory controls; covariate and interaction directions that are aliased
with the controls are removed using pivoted QR. Effective degrees of freedom
are reported rather than overstated. This is a global test of the estimable
linear interaction basis, not proof that individual CATE estimates are accurate.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr
from scipy.stats import chi2

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, TARGET_COL
from src.data_preprocessing import FEATURE_COLUMNS, build_preprocessor

INPUTS = [PROCESSED_DATA_DIR / name for name in
          ("train_data.csv", "validation_data.csv")]
OUTPUT = REPORTS_DIR / "heterogeneity" / "global_interaction_tests.csv"


def _independent_column_indices(matrix):
    """Return pivoted-QR indices for a numerically independent column basis."""
    _, r, pivots = qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.array([], dtype=int)
    tolerance = np.finfo(float).eps * max(matrix.shape) * float(diagonal.max())
    rank = int(np.sum(diagonal > tolerance))
    return np.sort(pivots[:rank]).astype(int)


def _select_covariates_after_controls(controls, candidates):
    """Select candidate columns that add independent rank beyond mandatory controls.

    The intercept and treatment assignment must remain in the restricted model.
    Candidate covariates are residualized against those controls before QR
    selection, preventing an intercept/one-hot or treatment/covariate alias from
    making the restricted design singular.
    """
    controls = np.asarray(controls, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if controls.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Controls and candidates must be two-dimensional.")
    if controls.shape[0] != candidates.shape[0]:
        raise ValueError("Controls and candidates must have the same row count.")
    if np.linalg.matrix_rank(controls) != controls.shape[1]:
        raise ValueError("Mandatory intercept/treatment controls are rank deficient.")

    projection = np.linalg.lstsq(controls, candidates, rcond=None)[0]
    residualized = candidates - controls @ projection
    selected = _independent_column_indices(residualized)
    return candidates[:, selected], selected


def _estimable_interaction_columns(restricted_design, interaction_design):
    """Select interactions that add independent directions beyond the null model."""
    projection = np.linalg.lstsq(
        restricted_design, interaction_design, rcond=None
    )[0]
    residualized = interaction_design - restricted_design @ projection
    return _independent_column_indices(residualized)


def _hc3_robust_score_test(y, restricted_design, interaction_design):
    """HC3-robust score test for omitted, estimable interaction terms.

    Interaction columns are residualized against the restricted design
    (Frisch-Waugh-Lovell). Pivoted QR selects the independent, estimable
    directions. The score covariance uses HC3-adjusted restricted-model
    residuals. A p-value is not returned if the robust score covariance is
    rank deficient.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    restricted_design = np.asarray(restricted_design, dtype=float)
    interaction_design = np.asarray(interaction_design, dtype=float)

    if restricted_design.ndim != 2 or interaction_design.ndim != 2:
        raise ValueError("Design matrices must be two-dimensional.")
    if len(y) != restricted_design.shape[0] or len(y) != interaction_design.shape[0]:
        raise ValueError("Outcome and design matrices must have the same row count.")
    if interaction_design.shape[1] == 0:
        raise ValueError("At least one interaction term is required.")

    restricted_columns = restricted_design.shape[1]
    restricted_rank = int(np.linalg.matrix_rank(restricted_design))
    if restricted_rank != restricted_columns:
        raise ValueError(
            "The restricted constant-effect design is rank deficient: "
            f"rank={restricted_rank}, columns={restricted_columns}."
        )

    selected = _estimable_interaction_columns(
        restricted_design, interaction_design
    )
    if selected.size == 0:
        raise ValueError(
            "No treatment-by-feature interaction directions are estimable "
            "beyond the constant-effect model. No omnibus p-value was produced."
        )

    selected_interactions = interaction_design[:, selected]
    projection_coefficients = np.linalg.lstsq(
        restricted_design, selected_interactions, rcond=None
    )[0]
    residualized_interactions = (
        selected_interactions - restricted_design @ projection_coefficients
    )
    interaction_rank = int(np.linalg.matrix_rank(residualized_interactions))
    if interaction_rank != selected.size:
        raise ValueError(
            "The selected treatment-by-feature interaction basis is still "
            "rank deficient after residualization. No omnibus p-value was produced."
        )

    restricted_beta = np.linalg.lstsq(restricted_design, y, rcond=None)[0]
    residuals = y - restricted_design @ restricted_beta

    # QR gives restricted-model leverage without forming an n-by-n hat matrix.
    q_matrix, _ = np.linalg.qr(restricted_design, mode="reduced")
    leverage = np.sum(q_matrix ** 2, axis=1)
    one_minus_leverage = 1.0 - leverage
    if np.any(one_minus_leverage <= 1e-10):
        raise ValueError(
            "HC3 adjustment is undefined or unstable because restricted-model "
            "leverage is effectively one."
        )

    hc3_residuals = residuals / one_minus_leverage
    score = residualized_interactions.T @ residuals
    score_contributions = residualized_interactions * hc3_residuals[:, None]
    score_covariance = score_contributions.T @ score_contributions

    diagonal = np.diag(score_covariance)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise ValueError(
            "The HC3 score covariance has a zero or non-finite diagonal; "
            "no omnibus p-value was produced."
        )

    scales = np.sqrt(diagonal)
    normalized_covariance = score_covariance / np.outer(scales, scales)
    normalized_covariance = (
        normalized_covariance + normalized_covariance.T
    ) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(normalized_covariance)
    largest_eigenvalue = float(np.max(eigenvalues))
    tolerance = max(largest_eigenvalue, 1.0) * 1e-10
    covariance_rank = int(np.sum(eigenvalues > tolerance))

    if covariance_rank != selected.size:
        raise ValueError(
            "The HC3 score covariance remains rank deficient: "
            f"rank={covariance_rank}, estimable interaction terms={selected.size}. "
            "No omnibus p-value was produced."
        )
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("The HC3 score covariance is not positive semidefinite.")

    standardized_score = score / scales
    rotated_score = eigenvectors.T @ standardized_score
    statistic = float(np.sum((rotated_score ** 2) / eigenvalues))
    degrees_freedom = covariance_rank
    p_value = float(chi2.sf(statistic, degrees_freedom))

    return {
        "statistic": statistic,
        "degrees_freedom": degrees_freedom,
        "p_value": p_value,
        "covariance_rank": covariance_rank,
        "interaction_design_rank": interaction_rank,
        "selected_interaction_indices": selected,
    }


def run():
    frames = [pd.read_csv(path) for path in INPUTS]
    df = pd.concat(frames, ignore_index=True)
    if df["source_row_index"].duplicated().any():
        raise ValueError("Duplicate customer IDs in development data.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(float).to_numpy()
    t = df["treatment"].astype(int).to_numpy()

    treatment_values = set(np.unique(t))
    if treatment_values != {0, 1}:
        raise ValueError(
            "Expected treatment values 0/1 for the two-arm experiment; "
            f"found {sorted(treatment_values)}."
        )

    transformed = build_preprocessor().fit_transform(X)
    transformed = (
        transformed.toarray()
        if hasattr(transformed, "toarray")
        else np.asarray(transformed)
    )

    # Mandatory null-model controls are established first and retained.
    controls = np.column_stack([np.ones(len(t)), t.astype(float)])

    # Remove constant features and select covariates only after accounting for
    # both the intercept and treatment indicator.
    variable = np.std(transformed, axis=0) > 1e-12
    variable_features = transformed[:, variable]
    selected_features, selected_feature_indices = (
        _select_covariates_after_controls(controls, variable_features)
    )
    base = np.column_stack([controls, selected_features])

    # The alternative allows the treatment effect to vary with each retained
    # covariate; any further aliases are handled by the interaction QR step.
    interactions = selected_features * t[:, None]
    test = _hc3_robust_score_test(y, base, interactions)

    selected_interactions = interactions[
        :, test["selected_interaction_indices"]
    ]
    full = np.column_stack([base, selected_interactions])

    base_fit = sm.OLS(y, base).fit()
    full_fit = sm.OLS(y, full).fit()
    if int(base_fit.model.rank) != base.shape[1]:
        raise ValueError(
            "The restricted model unexpectedly became rank deficient after "
            "control-first feature selection."
        )
    if int(full_fit.model.rank) != full.shape[1]:
        raise ValueError(
            "The selected unrestricted interaction design is unexpectedly "
            "rank deficient: "
            f"rank={full_fit.model.rank}, columns={full.shape[1]}."
        )

    n_nominal = int(variable_features.shape[1])
    n_selected_features = int(selected_features.shape[1])
    n_estimable_interactions = int(selected_interactions.shape[1])

    result = pd.DataFrame([{
        "contrast": "Mens E-Mail vs No Email",
        "n": len(y),
        "n_transformed_variable_features": n_nominal,
        "n_covariates_retained_after_controls": n_selected_features,
        "n_covariates_aliased_with_controls": n_nominal - n_selected_features,
        "n_estimable_interaction_terms": n_estimable_interactions,
        "n_aliased_interaction_terms": (
            n_selected_features - n_estimable_interactions
        ),
        "interaction_design_rank": test["interaction_design_rank"],
        "robust_covariance_rank": test["covariance_rank"],
        "constant_effect_r2": float(base_fit.rsquared),
        "heterogeneous_effect_r2": float(full_fit.rsquared),
        "test_type": "HC3-robust score test (chi-square)",
        "robust_score_statistic": test["statistic"],
        "degrees_freedom": test["degrees_freedom"],
        "p_value": test["p_value"],
        "alpha": 0.05,
        "reject_constant_effect": bool(test["p_value"] < 0.05),
    }])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
