"""Dedicated X-Learner fit-on-train / evaluate-on-validation stage.

Hyperparameters for the outcome and effect models are tuned only inside the
training split. The validation split is used once for model assessment.
"""
import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score, uplift_auc_score
from sklearn.metrics import log_loss

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, TARGET_COL
from src.data_preprocessing import FEATURE_COLUMNS
from src.nested_cv import _fit_x_learner, _predict_x

TRAIN = PROCESSED_DATA_DIR / "train_data.csv"
VALIDATION = PROCESSED_DATA_DIR / "validation_data.csv"
OUT = REPORTS_DIR / "x_learner_validation_predictions.csv"
SUMMARY = REPORTS_DIR / "x_learner_validation_metrics.csv"


def run():
    train, val = pd.read_csv(TRAIN), pd.read_csv(VALIDATION)
    required = FEATURE_COLUMNS + [TARGET_COL, "treatment", "source_row_index"]
    for name, frame in (("train", train), ("validation", val)):
        missing = [col for col in required if col not in frame]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    if set(train.source_row_index) & set(val.source_row_index):
        raise ValueError("Train and validation IDs overlap.")
    X_train, X_val = train[FEATURE_COLUMNS].copy(), val[FEATURE_COLUMNS].copy()
    y_train = train[TARGET_COL].astype(int).to_numpy()
    t_train = train["treatment"].astype(int).to_numpy()
    y_val = val[TARGET_COL].astype(int).to_numpy()
    t_val = val["treatment"].astype(int).to_numpy()
    model = _fit_x_learner(X_train, y_train, t_train, seed=42)
    p1, p0, uplift = _predict_x(model, X_val)
    summary = pd.DataFrame([{
        "model": "X-Learner",
        "n_validation": len(val),
        "conversions": int(y_val.sum()),
        "treated_log_loss": log_loss(y_val[t_val == 1], p1[t_val == 1], labels=[0, 1]),
        "control_log_loss": log_loss(y_val[t_val == 0], p0[t_val == 0], labels=[0, 1]),
        "sklift_qini_auc": qini_auc_score(y_val, uplift, t_val),
        "sklift_uplift_auc": uplift_auc_score(y_val, uplift, t_val),
    }])
    predictions = pd.DataFrame({
        "source_row_index": val.source_row_index.to_numpy(),
        "actual_treatment": t_val,
        "actual_conversion": y_val,
        "predicted_purchase_if_treated": p1,
        "predicted_purchase_if_control": p0,
        "predicted_uplift": uplift,
    }).sort_values("predicted_uplift", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT, index=False)
    summary.to_csv(SUMMARY, index=False)
    print(summary.round(6).to_string(index=False))
    print("Predictions:", OUT)
    return summary, predictions


if __name__ == "__main__":
    run()
