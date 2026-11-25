"""Global test for treatment-effect heterogeneity in the Hillstrom experiment.

For each email arm versus No Email, compare a constant-effect linear probability
model with a model containing treatment-by-feature interactions. HC3 robust
covariance and a joint Wald F-test are used. This is a global exploratory test,
not proof of individual-level CATE accuracy.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, TARGET_COL
from src.data_preprocessing import FEATURE_COLUMNS, build_preprocessor

INPUTS = [PROCESSED_DATA_DIR / name for name in
          ("train_data.csv", "validation_data.csv")]
OUTPUT = REPORTS_DIR / "heterogeneity" / "global_interaction_tests.csv"


def run():
    frames = [pd.read_csv(path) for path in INPUTS]
    df = pd.concat(frames, ignore_index=True)
    if df["source_row_index"].duplicated().any():
        raise ValueError("Duplicate customer IDs in development data.")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(float).to_numpy()
    t = df["treatment"].astype(int).to_numpy()
    z = build_preprocessor().fit_transform(X)
    z = z.toarray() if hasattr(z, "toarray") else np.asarray(z)
    z = sm.add_constant(z, has_constant="add")
    names = ["const"] + list(build_preprocessor().fit(X).get_feature_names_out())

    rows = []
    for arm in (1,):
        mask = np.isin(t, [0, arm])
        y_arm, t_arm, x_arm = y[mask], (t[mask] == arm).astype(float), z[mask]
        base = np.column_stack([x_arm, t_arm])
        full = np.column_stack([x_arm, t_arm, x_arm[:, 1:] * t_arm[:, None]])
        base_fit = sm.OLS(y_arm, base).fit(cov_type="HC3")
        full_fit = sm.OLS(y_arm, full).fit(cov_type="HC3")
        # Joint robust Wald test: every treatment-feature interaction = 0.
        interaction_idx = np.arange(base.shape[1], full.shape[1])
        restriction = np.zeros((len(interaction_idx), full.shape[1]))
        restriction[np.arange(len(interaction_idx)), interaction_idx] = 1.0
        test = full_fit.wald_test(restriction, use_f=True, scalar=True)
        rows.append({
            "contrast": f"arm_{arm}_vs_control",
            "n": int(mask.sum()),
            "n_interactions": len(interaction_idx),
            "constant_effect_r2": float(base_fit.rsquared),
            "heterogeneous_effect_r2": float(full_fit.rsquared),
            "robust_wald_statistic": float(np.asarray(test.statistic).squeeze()),
            "degrees_freedom_num": int(test.df_num),
            "degrees_freedom_denom": int(test.df_denom),
            "p_value": float(test.pvalue),
            "alpha": 0.05,
            "reject_constant_effect": bool(test.pvalue < 0.05),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
