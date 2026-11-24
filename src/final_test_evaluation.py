"""Final, one-time evaluation of the selected model (LightGBM T-Learner).

Protocol
--------
1. Model choice was made BEFORE this script, using validation and repeated
   out-of-fold results only (LightGBM T-Learner).
2. The tuned hyper-parameters from validation are re-used. The model is refit
   on train + validation so that it uses all non-test data.
3. The untouched test split is scored ONCE. Nothing is tuned on it.
4. Reported: Qini / AUUC, Uplift@K, probability metrics, and a stratified
   bootstrap confidence interval. The Logistic T-Learner (baseline) is scored
   on the same test rows so the comparison is paired.

Run from the project root:
    python -m src.final_test_evaluation
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    RANDOM_STATE,
)
from src.data_preprocessing import FEATURE_COLUMNS, build_preprocessor
from src.model_comparison import (
    calculate_group_uplift,
    calculate_qini_curve,
    calculate_qini_auuc,
    calculate_probability_metrics,
)

TRAIN_PATH = PROCESSED_DATA_DIR / "train_data.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_data.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test_data.csv"

LIGHTGBM_BUNDLE_PATH = ARTIFACTS_DIR / "lightgbm_t_learner.joblib"
LOGISTIC_PARAMS_PATH = ARTIFACTS_DIR / "t_learner_tuned_params.joblib"

OUTPUT_DIR = REPORTS_DIR / "final_test"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_lightgbm_t_learner.joblib"

TARGET_PERCENTAGES = [0.10, 0.20, 0.30]
BOOTSTRAP_REPEATS = 2000
CONFIDENCE_LEVEL = 0.95


# ----------------------------------------------------------------------
# Metrics (pure functions, no model needed)
# ----------------------------------------------------------------------

def metrics_from_predictions(pred_df):
    """Qini, AUUC and Uplift@K from a frame with actual_treatment,
    actual_conversion and predicted_uplift."""
    qini_df, observed_ate_pp = calculate_qini_curve(pred_df)
    auuc = calculate_qini_auuc(qini_df)

    ranked = pred_df.sort_values(
        "predicted_uplift", ascending=False, kind="mergesort"
    ).reset_index(drop=True)

    result = {
        "observed_ate_pp": observed_ate_pp,
        "model_auuc": auuc["model_auuc"],
        "random_auuc": auuc["random_auuc"],
        "qini_coefficient": auuc["qini_coefficient"],
    }

    for pct in TARGET_PERCENTAGES:
        k = int(np.ceil(len(ranked) * pct))
        try:
            group = calculate_group_uplift(ranked.iloc[:k])
            result[f"uplift_at_{int(pct * 100)}_pp"] = group["observed_uplift_pp"]
        except ValueError:
            result[f"uplift_at_{int(pct * 100)}_pp"] = np.nan

    return result


def stratified_bootstrap(pred_df, repeats=BOOTSTRAP_REPEATS):
    """Resample treated and control rows separately (arm sizes preserved)."""
    rng = np.random.default_rng(RANDOM_STATE)
    pred_df = pred_df.reset_index(drop=True)

    treated_idx = np.flatnonzero(pred_df["actual_treatment"].to_numpy() == 1)
    control_idx = np.flatnonzero(pred_df["actual_treatment"].to_numpy() == 0)

    rows = []
    for _ in range(repeats):
        sample_idx = np.concatenate([
            rng.choice(treated_idx, size=len(treated_idx), replace=True),
            rng.choice(control_idx, size=len(control_idx), replace=True),
        ])
        try:
            rows.append(metrics_from_predictions(pred_df.iloc[sample_idx]))
        except ValueError:
            continue

    return pd.DataFrame(rows)


def summarise_bootstrap(boot_df, point_estimates, model_name):
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    out = []
    for metric in ["qini_coefficient", "model_auuc",
                   "uplift_at_10_pp", "uplift_at_20_pp", "uplift_at_30_pp"]:
        values = boot_df[metric].dropna().to_numpy()
        out.append({
            "model": model_name,
            "metric": metric,
            "point_estimate": point_estimates[metric],
            "bootstrap_std": float(np.std(values, ddof=1)),
            "ci_lower": float(np.quantile(values, alpha)),
            "ci_upper": float(np.quantile(values, 1 - alpha)),
            "valid_repetitions": int(len(values)),
        })
    return out


# ----------------------------------------------------------------------
# Model fitting / scoring
# ----------------------------------------------------------------------

def load_splits():
    for path in (TRAIN_PATH, VALIDATION_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required = FEATURE_COLUMNS + [TARGET_COL, "treatment", "source_row_index"]
    for name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} data is missing columns: {missing}")

    # Guard against leakage between splits.
    ids = {
        "train": set(train_df["source_row_index"]),
        "validation": set(val_df["source_row_index"]),
        "test": set(test_df["source_row_index"]),
    }
    if (ids["train"] & ids["test"]) or (ids["validation"] & ids["test"]):
        raise ValueError("Test rows overlap with train/validation rows")

    return train_df, val_df, test_df


def refit_t_learner(bundle, fit_df):
    """Clone the tuned treatment/control pipelines and refit on fit_df."""
    treated = fit_df[fit_df["treatment"] == 1]
    control = fit_df[fit_df["treatment"] == 0]

    treatment_model = clone(bundle["treatment_model"])
    control_model = clone(bundle["control_model"])

    treatment_model.fit(treated[FEATURE_COLUMNS], treated[TARGET_COL].astype(int))
    control_model.fit(control[FEATURE_COLUMNS], control[TARGET_COL].astype(int))

    return {
        "treatment_model": treatment_model,
        "control_model": control_model,
        "feature_columns": FEATURE_COLUMNS,
    }


def build_logistic_bundle(fit_df):
    """Baseline: Logistic T-Learner rebuilt from its tuned settings.

    The saved t_learner_tuned.joblib stores a custom class that only unpickles
    inside the original script, so the baseline is rebuilt here instead.
    """
    params = joblib.load(LOGISTIC_PARAMS_PATH)

    def make_pipeline(c_value):
        return Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(
                C=c_value,
                max_iter=params["max_iter"],
                solver="lbfgs",
                class_weight=None,
            )),
        ])

    template = {
        "treatment_model": make_pipeline(params["treatment_C"]),
        "control_model": make_pipeline(params["control_C"]),
    }
    return refit_t_learner(template, fit_df)


def score_test(bundle, test_df):
    X_test = test_df[FEATURE_COLUMNS]
    p_treated = bundle["treatment_model"].predict_proba(X_test)[:, 1]
    p_control = bundle["control_model"].predict_proba(X_test)[:, 1]

    return pd.DataFrame({
        "source_row_index": test_df["source_row_index"].to_numpy(),
        "actual_treatment": test_df["treatment"].astype(int).to_numpy(),
        "actual_conversion": test_df[TARGET_COL].astype(int).to_numpy(),
        "predicted_purchase_if_treated": p_treated,
        "predicted_purchase_if_control": p_control,
        "predicted_uplift": p_treated - p_control,
    })


def evaluate_model(name, bundle, test_df):
    pred_df = score_test(bundle, test_df)
    point = metrics_from_predictions(pred_df)
    prob = calculate_probability_metrics(pred_df, name)
    boot = stratified_bootstrap(pred_df)
    return pred_df, point, prob, boot


def run_final_test_evaluation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits()
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    print("Train + validation customers:", len(fit_df))
    print("Test customers (scored once):", len(test_df))

    lightgbm_bundle = refit_t_learner(joblib.load(LIGHTGBM_BUNDLE_PATH), fit_df)
    logistic_bundle = build_logistic_bundle(fit_df)

    joblib.dump(lightgbm_bundle, FINAL_MODEL_PATH)

    models = {
        "LightGBM T-Learner (final)": lightgbm_bundle,
        "Logistic T-Learner (baseline)": logistic_bundle,
    }

    point_rows, prob_rows, boot_rows = [], [], []
    for name, bundle in models.items():
        pred_df, point, prob, boot = evaluate_model(name, bundle, test_df)

        slug = "lightgbm" if "LightGBM" in name else "logistic"
        pred_df.sort_values("predicted_uplift", ascending=False).to_csv(
            OUTPUT_DIR / f"{slug}_test_predictions.csv", index=False
        )

        point_rows.append({"model": name, **point})
        prob_rows.append(prob)
        boot_rows.extend(summarise_bootstrap(boot, point, name))

        qini_df, _ = calculate_qini_curve(pred_df)
        qini_df.to_csv(OUTPUT_DIR / f"{slug}_test_qini_curve.csv", index=False)

        if slug == "lightgbm":
            boot.to_csv(OUTPUT_DIR / "lightgbm_test_bootstrap_distribution.csv",
                        index=False)

    point_df = pd.DataFrame(point_rows)
    prob_df = pd.DataFrame(prob_rows)
    boot_df = pd.DataFrame(boot_rows)

    point_df.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)
    prob_df.to_csv(OUTPUT_DIR / "final_test_probability_metrics.csv", index=False)
    boot_df.to_csv(OUTPUT_DIR / "final_test_bootstrap_summary.csv", index=False)

    print("\nFINAL TEST METRICS")
    print(point_df.round(4).to_string(index=False))
    print("\nPROBABILITY METRICS")
    print(prob_df.round(5).to_string(index=False))
    print("\nBOOTSTRAP (95% CI)")
    print(boot_df.round(4).to_string(index=False))
    print("\nResults saved in:", OUTPUT_DIR)
    print(
        "\nReminder: the test split has now been used. Do not re-tune the model "
        "based on these numbers."
    )

    return point_df, prob_df, boot_df


if __name__ == "__main__":
    run_final_test_evaluation()
