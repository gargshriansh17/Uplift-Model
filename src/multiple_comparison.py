"""Multiple-comparison correction for paired outer-fold uplift scores.

Holm's step-down procedure controls family-wise error across all pairwise
comparisons for the pre-declared primary metric (normalized Qini AUC).
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.config import REPORTS_DIR

INPUT = REPORTS_DIR / "nested_cv" / "outer_fold_metrics.csv"
OUTPUT = REPORTS_DIR / "nested_cv" / "holm_pairwise_qini.csv"


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


def run():
    if not INPUT.exists():
        raise FileNotFoundError(f"Run nested CV first: {INPUT}")
    df = pd.read_csv(INPUT)
    wide = df.pivot(index="outer_fold", columns="model", values="sklift_qini_auc")
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        stat, p = ttest_rel(wide[a], wide[b], nan_policy="raise")
        rows.append({"model_a": a, "model_b": b, "metric": "sklift_qini_auc",
                     "mean_difference_a_minus_b": float((wide[a]-wide[b]).mean()),
                     "paired_t_statistic": float(stat), "raw_p_value": float(p)})
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(result["raw_p_value"])
    result["significant_holm_0_05"] = result["holm_adjusted_p_value"] < 0.05
    result.to_csv(OUTPUT, index=False)
    print(result.round(6).to_string(index=False))
    return result


if __name__ == "__main__":
    run()
