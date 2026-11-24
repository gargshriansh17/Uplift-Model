import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import log_loss,brier_score_loss

from src.config import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    RANDOM_STATE
)

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor
)

TRAIN_PATH=PROCESSED_DATA_DIR/"train_data.csv"
VALIDATION_PATH=PROCESSED_DATA_DIR/"validation_data.csv"

MODEL_PATH=ARTIFACTS_DIR/"lightgbm_t_learner.joblib"

TREATMENT_CV_PATH=REPORTS_DIR/"lightgbm_treatment_cv_results.csv"
CONTROL_CV_PATH=REPORTS_DIR/"lightgbm_control_cv_results.csv"

PARAMS_PATH=ARTIFACTS_DIR/"lightgbm_t_learner_params.joblib"

DIAGNOSTICS_PATH=REPORTS_DIR/"lightgbm_t_learner_diagnostics.csv"

PREDICTIONS_PATH=REPORTS_DIR/"lightgbm_t_learner_validation_predictions.csv"


CV_FOLDS=5
MAX_ITERATIONS=2000

PARAM_GRID=[
    {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.03, 0.05],
        "model__num_leaves": [3],
        "model__max_depth": [2, 3]
    },
    {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.03, 0.05],
        "model__num_leaves": [7],
        "model__max_depth": [3]
    }
]

def load_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found:{TRAIN_PATH}")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Validation data not found:{VALIDATION_PATH}")

    train_df=pd.read_csv(TRAIN_PATH)
    val_df=pd.read_csv(VALIDATION_PATH)

    required_cols=FEATURE_COLUMNS+[
        TARGET_COL,
        "treatment",
        "source_row_index"
    ]

    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing training column:{col}")

        if col not in val_df.columns:
            raise ValueError(f"Missing validation column:{col}")

    X_train=train_df[FEATURE_COLUMNS].copy()
    y_train=train_df[TARGET_COL].astype(int).to_numpy()
    treatment_train=train_df["treatment"].astype(int).to_numpy()

    X_val=val_df[FEATURE_COLUMNS].copy()
    y_val=val_df[TARGET_COL].astype(int).copy()
    treatment_val=val_df["treatment"].astype(int).copy()

    return(
        train_df,
        val_df,
        X_train,
        y_train,
        treatment_train,
        X_val,
        y_val,
        treatment_val
    )


def build_lightgbm_pipeline():
    model=LGBMClassifier(
        objective="binary",
        min_child_samples=50,
        reg_lambda=5.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1
    )

    pipeline=Pipeline([
        ("preprocessor",build_preprocessor()),
        ("model",model)
    ])

    return pipeline

def train_arm(X,y,arm_name):
    print(f"\n{'=' * 50}")
    print(f"TRAINING {arm_name.upper()} MODEL")
    print(f"{'=' * 50}")

    conversions=int(np.sum(y))

    print("Customers:",len(X))
    print("Conversions:",conversions)

    if len(np.unique(y))!=2:
        raise ValueError(
            f"{arm_name} training data must contain both outcome classes"
        )

    if min(np.bincount(y))<CV_FOLDS:
        raise ValueError(
            f"Not enough observations in each outcome class for {CV_FOLDS}-fold CV in {arm_name}"
        )

    cv=StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    pipeline=build_lightgbm_pipeline()

    grid_search=GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="neg_log_loss",
        return_train_score=True,
        n_jobs=1,
        refit=True,
        error_score="raise"
    )

    grid_search.fit(X,y)

    results=grid_search.cv_results_

    cv_df=pd.DataFrame({
        "mean_cv_log_loss":-results["mean_test_score"],
        "std_cv_log_loss":results["std_test_score"],
        "mean_train_log_loss":-results["mean_train_score"],
        "rank":results["rank_test_score"],
        "params":[str(p) for p in results["params"]]
    })

    cv_df=cv_df.sort_values("rank")

    best_model=grid_search.best_estimator_

    print("\nCROSS-VALIDATION RESULTS")
    print(cv_df.head(12).round(6).to_string(index=False))

    print("\nBEST PARAMETERS:")

    print(grid_search.best_params_)

    print(
        "Best CV Log Loss:",
        round(float(-grid_search.best_score_), 6)
    )

    print(
        "CV Standard Deviation:",
        round(float(results["std_test_score"][grid_search.best_index_]), 6)
    )

    return best_model, cv_df, grid_search.best_params_


def run_lightgbm_t_learner():
    (
        train_df,
        val_df,
        X_train,
        y_train,
        treatment_train,
        X_val,
        y_val,
        treatment_val
    )=load_data()

    print("\nLIGHTGBM T-LEARNER")

    print("Training Customers:", len(X_train))
    print("Validation Customers:", len(X_val))
    print("Training Conversions:", int(y_train.sum()))

    treatment_mask_train = treatment_train == 1
    control_mask_train = treatment_train == 0

    X_treatment = X_train.loc[treatment_mask_train].copy()
    y_treatment = y_train[treatment_mask_train]

    X_control = X_train.loc[control_mask_train].copy()
    y_control = y_train[control_mask_train]

    print("\nTreatment Customers:", len(X_treatment))
    print("Treatment Conversions:", int(y_treatment.sum()))

    print("Control Customers:", len(X_control))
    print("Control Conversions:", int(y_control.sum()))

    # ----------------------------------------------
    # TRAIN TREATMENT MODEL
    # ----------------------------------------------

    treatment_model, treatment_cv_df, treatment_params = train_arm(
        X_treatment,
        y_treatment,
        "Treatment"
    )

    treatment_cv_df.to_csv(
        TREATMENT_CV_PATH,
        index=False
    )

    # ----------------------------------------------
    # TRAIN CONTROL MODEL
    # ----------------------------------------------

    control_model, control_cv_df, control_params = train_arm(
        X_control,
        y_control,
        "Control"
    )

    control_cv_df.to_csv(
        CONTROL_CV_PATH,
        index=False
    )

    # ----------------------------------------------
    # VALIDATION PREDICTIONS
    # ----------------------------------------------

    probability_treated = treatment_model.predict_proba(X_val)[:, 1]

    probability_control = control_model.predict_proba(X_val)[:, 1]

    predicted_uplift = probability_treated - probability_control

    treatment_mask_val = treatment_val == 1
    control_mask_val = treatment_val == 0

    # ----------------------------------------------
    # VALIDATION DIAGNOSTICS
    # ----------------------------------------------

    treatment_log_loss = log_loss(
        y_val[treatment_mask_val],
        probability_treated[treatment_mask_val],
        labels=[0, 1]
    )

    control_log_loss = log_loss(
        y_val[control_mask_val],
        probability_control[control_mask_val],
        labels=[0, 1]
    )

    treatment_brier = brier_score_loss(
        y_val[treatment_mask_val],
        probability_treated[treatment_mask_val]
    )

    control_brier = brier_score_loss(
        y_val[control_mask_val],
        probability_control[control_mask_val]
    )

    diagnostics_df = pd.DataFrame([
        {
            "model": "LightGBM T-Learner",
            "arm": "Treatment",
            "sample_size": int(treatment_mask_val.sum()),
            "conversions": int(y_val[treatment_mask_val].sum()),
            "log_loss": treatment_log_loss,
            "brier_score": treatment_brier
        },
        {
            "model": "LightGBM T-Learner",
            "arm": "Control",
            "sample_size": int(control_mask_val.sum()),
            "conversions": int(y_val[control_mask_val].sum()),
            "log_loss": control_log_loss,
            "brier_score": control_brier
        }
    ])

    diagnostics_df.to_csv(
        DIAGNOSTICS_PATH,
        index=False
    )

    print("\nVALIDATION DIAGNOSTICS")

    print(
        diagnostics_df.round(6).to_string(index=False)
    )

    # ----------------------------------------------
    # VALIDATION PREDICTIONS FILE
    # ----------------------------------------------

    validation_predictions = pd.DataFrame({

        "source_row_index": val_df["source_row_index"],

        "actual_treatment": treatment_val,

        "actual_conversion": y_val,

        "predicted_purchase_if_treated": probability_treated,

        "predicted_purchase_if_control": probability_control,

        "predicted_uplift": predicted_uplift,

        "uplift_percentage_points": predicted_uplift * 100

    })

    validation_predictions = validation_predictions.sort_values(
        by="predicted_uplift",
        ascending=False
    )

    validation_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # ----------------------------------------------
    # UPLIFT SUMMARY
    # ----------------------------------------------

    print("\nLIGHTGBM UPLIFT SUMMARY")

    print(
        "Average predicted uplift:",
        round(float(np.mean(predicted_uplift)) * 100, 4),
        "percentage points"
    )

    print(
        "Median predicted uplift:",
        round(float(np.median(predicted_uplift)) * 100, 4),
        "percentage points"
    )

    print(
        "Minimum predicted uplift:",
        round(float(np.min(predicted_uplift)) * 100, 4),
        "percentage points"
    )

    print(
        "Maximum predicted uplift:",
        round(float(np.max(predicted_uplift)) * 100, 4),
        "percentage points"
    )

    print(
        "Positive uplift customers:",
        int(np.sum(predicted_uplift > 0))
    )

    print(
        "Negative uplift customers:",
        int(np.sum(predicted_uplift < 0))
    )

    # ----------------------------------------------
    # SAVE MODEL AND PARAMETERS
    # ----------------------------------------------

    model_bundle = {
        "treatment_model": treatment_model,
        "control_model": control_model,
        "feature_columns": FEATURE_COLUMNS
    }

    joblib.dump(model_bundle, MODEL_PATH)

    joblib.dump(
        {
            "treatment_best_params": treatment_params,
            "control_best_params": control_params
        },
        PARAMS_PATH
    )

    print("\nLIGHTGBM T-LEARNER COMPLETED")

    print("Model saved at:", MODEL_PATH)

    print("Treatment CV results:", TREATMENT_CV_PATH)

    print("Control CV results:", CONTROL_CV_PATH)

    print("Parameters saved at:", PARAMS_PATH)

    print("Diagnostics saved at:", DIAGNOSTICS_PATH)

    print("Predictions saved at:", PREDICTIONS_PATH)

    return (
        model_bundle,
        treatment_cv_df,
        control_cv_df,
        diagnostics_df,
        validation_predictions
    )


if __name__ == "__main__":

    run_lightgbm_t_learner()
