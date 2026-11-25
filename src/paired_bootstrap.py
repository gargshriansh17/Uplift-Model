"""Paired bootstrap: LightGBM T-Learner vs Logistic T-Learner on the final test split.

Both models are scored on the SAME resampled test rows in every replication, so
the metric differences (LightGBM minus Logistic) respect the correlation between
the two models. No model is refit and nothing is tuned here.

Needs the outputs of `python -m src.final_test_evaluation`:
    reports/final_test/lightgbm_test_predictions.csv
    reports/final_test/logistic_test_predictions.csv

Run from the project root:
    python -m src.paired_bootstrap
"""

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, RANDOM_STATE
from src.final_test_evaluation import (
    BOOTSTRAP_REPEATS,
    CONFIDENCE_LEVEL,
    TEST_PATH,
    metrics_from_predictions,
)

OUTPUT_DIR = REPORTS_DIR / "final_test"
LIGHTGBM_PRED_PATH = OUTPUT_DIR / "lightgbm_test_predictions.csv"
LOGISTIC_PRED_PATH = OUTPUT_DIR / "logistic_test_predictions.csv"

METRICS = [
    "qini_coefficient",
    "model_auuc",
    "uplift_at_10_pp",
    "uplift_at_20_pp",
    "uplift_at_30_pp",
]


def load_aligned_predictions():
    """Load both prediction files and restore the original test-set row order,
    so tie-breaking in the ranking matches final_test_evaluation.py."""
    test_order = pd.read_csv(TEST_PATH)["source_row_index"]

    frames = []
    for path in (LIGHTGBM_PRED_PATH, LOGISTIC_PRED_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions file: {path}")
        df = pd.read_csv(path).set_index("source_row_index").loc[test_order]
        frames.append(df.reset_index())

    lgb, log = frames
    if not (
        (lgb["actual_treatment"].to_numpy() == log["actual_treatment"].to_numpy()).all()
        and (lgb["actual_conversion"].to_numpy() == log["actual_conversion"].to_numpy()).all()
    ):
        raise ValueError("The two prediction files do not describe the same test rows")

    return lgb, log


def paired_bootstrap(lgb, log, repeats=BOOTSTRAP_REPEATS):
    rng = np.random.default_rng(RANDOM_STATE)
    treated_idx = np.flatnonzero(lgb["actual_treatment"].to_numpy() == 1)
    control_idx = np.flatnonzero(lgb["actual_treatment"].to_numpy() == 0)

    rows = []
    for _ in range(repeats):
        sample_idx = np.concatenate([
            rng.choice(treated_idx, size=len(treated_idx), replace=True),
            rng.choice(control_idx, size=len(control_idx), replace=True),
        ])
        try:
            m_lgb = metrics_from_predictions(lgb.iloc[sample_idx])
            m_log = metrics_from_predictions(log.iloc[sample_idx])
        except ValueError:
            continue
        rows.append({m: m_lgb[m] - m_log[m] for m in METRICS})

    return pd.DataFrame(rows)


def summarise(diff_df, point_lgb, point_log):
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    out = []
    for metric in METRICS:
        d = diff_df[metric].dropna().to_numpy()
        share_pos = float(np.mean(d > 0))
        share_neg = float(np.mean(d < 0))
        out.append({
            "metric": metric,
            "point_difference_lgbm_minus_logistic": point_lgb[metric] - point_log[metric],
            "bootstrap_std": float(np.std(d, ddof=1)),
            "ci_lower": float(np.quantile(d, alpha)),
            "ci_upper": float(np.quantile(d, 1 - alpha)),
            "share_of_replications_lgbm_better": share_pos,
            "approx_two_sided_p": min(1.0, 2 * min(share_pos, share_neg)),
            "valid_repetitions": int(len(d)),
        })
    return pd.DataFrame(out)


def run_paired_bootstrap():
    lgb, log = load_aligned_predictions()

    point_lgb = metrics_from_predictions(lgb)
    point_log = metrics_from_predictions(log)
    print("Point estimates (should match final_test_metrics.csv):")
    for m in METRICS:
        print(f"  {m}: LightGBM {point_lgb[m]:.4f} | Logistic {point_log[m]:.4f}")

    diff_df = paired_bootstrap(lgb, log)
    summary = summarise(diff_df, point_lgb, point_log)

    diff_df.to_csv(OUTPUT_DIR / "paired_bootstrap_distribution.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "paired_bootstrap_summary.csv", index=False)

    print("\nPAIRED BOOTSTRAP (LightGBM minus Logistic, 95% CI)")
    print(summary.round(4).to_string(index=False))
    print("\nSaved in:", OUTPUT_DIR)
    return summary


if __name__ == "__main__":
    run_paired_bootstrap()