"""Multiplicity-adjusted pairwise comparisons on outer-fold uplift scores.

Uses the Nadeau-Bengio corrected resampled t-test rather than an ordinary
paired t-test: outer-fold training sets overlap, so fold scores are dependent.
Holm's step-down procedure controls family-wise error across pairwise tests
for the pre-declared primary metric (normalized Qini AUC).
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.config import REPORTS_DIR

INPUT = REPORTS_DIR / "nested_cv" / "outer_fold_metrics.csv"
OUTPUT = REPORTS_DIR / "nested_cv" / "holm_pairwise_qini.csv"
OUTER_FOLDS = 5


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m-rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def corrected_resampled_test(differences, test_train_ratio=1/(OUTER_FOLDS-1)):
    """Two-sided corrected resampled t-test for overlapping CV training sets."""
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    if n < 2:
        raise ValueError("At least two outer-fold paired differences are required.")
    mean_difference = float(differences.mean())
    variance = float(differences.var(ddof=1))
    correction = (1.0 / n) + test_train_ratio
    standard_error = float(np.sqrt(correction * variance))
    if standard_error == 0:
        statistic = 0.0 if mean_difference == 0 else np.sign(mean_difference) * np.inf
        p_value = 1.0 if mean_difference == 0 else 0.0
    else:
        statistic = mean_difference / standard_error
        p_value = float(2 * student_t.sf(abs(statistic), df=n-1))
    return statistic, p_value, standard_error


def run():
    if not INPUT.exists():
        raise FileNotFoundError(f"Run nested CV first: {INPUT}")
    df = pd.read_csv(INPUT)
    wide = df.pivot(index="outer_fold", columns="model", values="sklift_qini_auc")
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        differences = (wide[a] - wide[b]).to_numpy()
        stat, p, se = corrected_resampled_test(differences)
        rows.append({
            "model_a": a, "model_b": b, "metric": "sklift_qini_auc",
            "mean_difference_a_minus_b": float(differences.mean()),
            "corrected_standard_error": se,
            "corrected_t_statistic": stat, "degrees_freedom": len(differences)-1,
            "raw_p_value": p,
        })
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(result["raw_p_value"])
    result["significant_holm_0_05"] = result["holm_adjusted_p_value"] < 0.05
    result.to_csv(OUTPUT, index=False)
    print(result.round(6).to_string(index=False))
    return result


if __name__ == "__main__":
    run()
