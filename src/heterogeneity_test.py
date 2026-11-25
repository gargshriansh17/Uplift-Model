"""Global treatment-effect heterogeneity test for the two-arm Hillstrom experiment.

The two-arm development data contains Men's Email (1) and No Email (0).
The null model is a constant-effect linear probability model. The alternative
adds treatment-by-feature interactions, testing whether the conditional
conversion-probability treatment effect varies with observed features.

The joint test is an HC3-robust score test evaluated under the restricted
(constant-effect) model. This avoids relying on the potentially unstable
interaction-block covariance from the unrestricted model. It remains a global
test of the specified linear interaction basis, not proof that individual
CATE estimates are accurate.
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


def _independent_columns(matrix):
    """Return a full-column-rank basis using pivoted QR."""
    _, r, pivots = qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        raise ValueError("No usable covariates for heterogeneity test.")
    tolerance = np.finfo(float).eps * max(matrix.shape) * diagonal.max()
    rank = int(np.sum(diagonal > tolerance))
    selected = np.sort(pivots[:rank])
    return matrix[:, selected]


def _hc3_robust_score_test(y, restricted_design, interaction_design):
    """HC3-robust score test for omitted interaction terms.

    The interaction columns are first residualized against the restricted
    design (Frisch-Waugh-Lovell). The score covariance uses HC3-adjusted
    restricted-model residuals. A scale-normalized covariance is used for
    numerical rank detection and inversion.
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

    n, restricted_columns = restricted_design.shape
    n_interactions = interaction_design.shape[1]
    restricted_rank = int(np.linalg.matrix_rank(restricted_design))
    if restricted_rank != restricted_columns:
        raise ValueError(
            "The restricted constant-effect design is rank deficient: "
            f"rank={restricted_rank}, columns={restricted_columns}."
        )

    # Fit the null model and residualize the candidate interaction columns
    # against it, so the score tests only their incremental contribution.
    restricted_beta = np.linalg.lstsq(restricted_design, y, rcond=None)[0]
    residuals = y - restricted_design @ restricted_beta
    projection_coefficients = np.linalg.lstsq(
        restricted_design, interaction_design, rcond=None
    )[0]
    residualized_interactions = (
        interaction_design - restricted_design @ projection_coefficients
    )

    interaction_rank = int(np.linalg.matrix_rank(residualized_interactions))
    if interaction_rank != n_interactions:
        raise ValueError(
            "The treatment-by-feature interaction design is rank deficient "
            f"after residualizing against the null model: rank={interaction_rank}, "
            f"terms={n_interactions}. No omnibus p-value was produced."
        )

    # QR gives the restricted-model leverage without forming an n-by-n hat matrix.
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

    # Normalize to a correlation-like matrix before rank detection. This avoids
    # mistaking differences in interaction-column scales for singularity.
    scales = np.sqrt(diagonal)
    normalized_covariance = score_covariance / np.outer(scales, scales)
    normalized_covariance = (
        normalized_covariance + normalized_covariance.T
    ) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(normalized_covariance)
    largest_eigenvalue = float(np.max(eigenvalues))
    tolerance = max(largest_eigenvalue, 1.0) * 1e-10
    covariance_rank = int(np.sum(eigenvalues > tolerance))

    if covariance_rank != n_interactions:
        raise ValueError(
            "The HC3 score covariance remains rank deficient: "
            f"rank={covariance_rank}, interaction terms={n_interactions}. "
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

    # Drop constant transformed columns and remove exact linear dependencies
    # (notably one-hot blocks) before constructing treatment interactions.
    variable = np.std(transformed, axis=0) > 1e-12
    z_features = _independent_columns(transformed[:, variable])
    z = sm.add_constant(z_features, has_constant="add")

    # Restricted model: customer features plus one constant treatment effect.
    base = np.column_stack([z, t.astype(float)])

    # Alternative: treatment effect may vary linearly with each retained feature.
    interactions = z[:, 1:] * t[:, None]
    full = np.column_stack([base, interactions])

    base_fit = sm.OLS(y, base).fit()
    full_fit = sm.OLS(y, full).fit()

    if int(full_fit.model.rank) != full.shape[1]:
        raise ValueError(
            "The unrestricted interaction design is rank deficient: "
            f"rank={full_fit.model.rank}, columns={full.shape[1]}."
        )

    test = _hc3_robust_score_test(y, base, interactions)

    result = pd.DataFrame([{
        "contrast": "Mens E-Mail vs No Email",
        "n": len(y),
        "n_interaction_terms": interactions.shape[1],
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
