"""Global treatment-effect heterogeneity test for the two-arm Hillstrom experiment.

The two-arm development data contains Men's Email (1) and No Email (0).
Compare a constant-effect linear probability model with a model containing
treatment-by-feature interactions. HC3-robust covariance and a joint Wald test
are used. Rank diagnostics are saved because a singular robust covariance must
not be mistaken for a full-rank omnibus test. This is exploratory, not proof of
individual-level CATE accuracy.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr

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


def run():
    frames = [pd.read_csv(path) for path in INPUTS]
    df = pd.concat(frames, ignore_index=True)
    if df["source_row_index"].duplicated().any():
        raise ValueError("Duplicate customer IDs in development data.")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(float).to_numpy()
    t = df["treatment"].astype(int).to_numpy()
    transformed = build_preprocessor().fit_transform(X)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    # Drop constant transformed columns, then remove exact linear dependencies
    # (notably full one-hot blocks) before forming treatment interactions.
    variable = np.std(transformed, axis=0) > 1e-12
    z = _independent_columns(transformed[:, variable])
    z = sm.add_constant(z, has_constant="add")

    treatment_values = set(np.unique(t))
    if treatment_values != {0, 1}:
        raise ValueError(
            f"Expected treatment values 0/1 for the two-arm experiment; found {sorted(treatment_values)}."
        )

    # This input is the existing two-arm experiment (Men's Email vs No Email).
    # Women's Email belongs to the separate three-arm pipeline and must not be
    # tested using these files.
    y_arm = y
    t_arm = t.astype(float)
    x_arm = z
    base = np.column_stack([x_arm, t_arm])
    interactions = x_arm[:, 1:] * t_arm[:, None]
    full = np.column_stack([base, interactions])

    base_fit = sm.OLS(y_arm, base).fit(cov_type="HC3")
    full_fit = sm.OLS(y_arm, full).fit(cov_type="HC3")
    interaction_idx = np.arange(base.shape[1], full.shape[1])
    restriction = np.zeros((len(interaction_idx), full.shape[1]))
    restriction[np.arange(len(interaction_idx)), interaction_idx] = 1.0

    constrained_covariance = (
        restriction @ np.asarray(full_fit.cov_params()) @ restriction.T
    )
    covariance_rank = int(np.linalg.matrix_rank(constrained_covariance))
    test = full_fit.wald_test(restriction, use_f=True, scalar=True)

    result = pd.DataFrame([{
        "contrast": "Mens E-Mail vs No Email",
        "n": len(y_arm),
        "n_interaction_terms": len(interaction_idx),
        "robust_covariance_rank": covariance_rank,
        "constant_effect_r2": float(base_fit.rsquared),
        "heterogeneous_effect_r2": float(full_fit.rsquared),
        "robust_wald_statistic": float(np.asarray(test.statistic).squeeze()),
        "degrees_freedom_num": int(test.df_num),
        "degrees_freedom_denom": int(test.df_denom),
        "p_value": float(test.pvalue),
        "alpha": 0.05,
        "reject_constant_effect": bool(test.pvalue < 0.05),
    }])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    if covariance_rank < len(interaction_idx):
        print(
            "WARNING: The HC3 covariance for the interaction block is rank "
            f"{covariance_rank}, below {len(interaction_idx)} interaction terms. "
            "Interpret the reported effective Wald degrees of freedom cautiously."
        )
    return result


if __name__ == "__main__":
    run()
