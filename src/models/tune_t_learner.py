import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold

from src.config import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    RANDOM_STATE
)

from src.data_preprocessing import FEATURE_COLUMNS

TRAIN_PATH=PROCESSED_DATA_DIR/"train_data.csv"
PREPROCESSOR_PATH=ARTIFACTS_DIR/"preprocessor.joblib"
TUNING_RESULTS_PATH=REPORTS_DIR/"t_learner_c_tuning.csv"
TUNED_PARAMS_PATH=ARTIFACTS_DIR/"t_learner_tuned_params.joblib"

C_VALUES=[0.001,0.01,0.1,1.0,10.0]
CV_FOLDS=5
MAX_ITER=2000

def load_training_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:{TRAIN_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:{PREPROCESSOR_PATH}"
        )

    train_df=pd.read_csv(TRAIN_PATH)

    required_columns=(
        FEATURE_COLUMNS+[TARGET_COL,"treatment"]
    )

    missing_columns=[
        col for col in required_columns
        if col not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing training outcomes:{missing_columns}"
        )

    X=train_df[FEATURE_COLUMNS].copy()
    y=train_df[TARGET_COL].astype(int).to_numpy()
    treatment=train_df["treatment"].astype(int).to_numpy()

    if not np.isin(y,[0,1]).all():
        raise ValueError("Invalid conversion values")

    if not np.isin(treatment,[0,1]).all():
        raise ValueError("Invalid treatment values")

    preprocessor=joblib.load(PREPROCESSOR_PATH)

    return X,y,treatment,preprocessor

def tune_arm(X_arm,y_arm,arm_name,preprocessor):
    if len(np.unique(y_arm))!=2:
        raise ValueError(
            f"{arm_name} must contain both conversion classes"
        )

    class_counts=np.bincount(y_arm)

    if class_counts.min()<CV_FOLDS:
        raise ValueError(
            f"Insufficient minority-class samples for {CV_FOLDS}-fold CV in {arm_name}"
        )

    print(f"\nTUNING {arm_name.upper()} MODEL")
    print("Customers:",len(y_arm))
    print("Conversion:",int(y_arm.sum()))

    cv=StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    pipeline=Pipeline([(
        "preprocessor",
        clone(preprocessor)
        ),
        (
        "model",
        LogisticRegression(
            max_iter=MAX_ITER,
            solver="lbfgs",
            class_weight=None
        )
        )
    ])

    param_grid={"model__C":C_VALUES}

    grid_search=GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True
    )

    grid_search.fit(X_arm,y_arm)

    best_c=grid_search.best_params_["model__C"]

    best_log_loss=-grid_search.best_score_

    print("\nC VALUE COMPARISON")

    results=[]

    for i,params in enumerate(
        grid_search.cv_results_["params"]
    ):

        mean_test_loss=-grid_search.cv_results_["mean_test_score"][i]

        std_test_loss=grid_search.cv_results_["std_test_score"][i]

        mean_train_loss=-grid_search.cv_results_["mean_train_score"][i]

        results.append({
            "model":arm_name,
            "C":params["model__C"],
            "mean_cv_log_loss":mean_test_loss,
            "std_cv_log_loss":std_test_loss,
            "mean_train_log_loss":mean_train_loss,
            "selected":np.isclose(params["model__C"],best_c)
        })

    results_df=pd.DataFrame(results)

    print(results_df.round(6).to_string(index=False))
    print("\nBest C:",best_c)

    print("Best CV Log Loss:",round(best_log_loss,6))

    return best_c,results_df

def run_tuning():
    X, y, treatment, preprocessor = load_training_data()

    # Treatment group

    treatment_mask = treatment == 1

    X_treatment = X.loc[treatment_mask]

    y_treatment = y[treatment_mask]

    treatment_best_c, treatment_results = tune_arm(
        X_treatment,
        y_treatment,
        "Treatment",
        preprocessor
    )

    # Control group

    control_mask = treatment == 0

    X_control = X.loc[control_mask]

    y_control = y[control_mask]

    control_best_c, control_results = tune_arm(
        X_control,
        y_control,
        "Control",
        preprocessor
    )

    # Combine results

    tuning_results = pd.concat(
        [treatment_results, control_results],
        ignore_index=True
    )

    tuning_results.to_csv(
        TUNING_RESULTS_PATH,
        index=False
    )

    selected_params = {

        "treatment_C": float(treatment_best_c),

        "control_C": float(control_best_c),

        "cv_folds": CV_FOLDS,

        "scoring": "neg_log_loss"

    }

    joblib.dump(
        selected_params,
        TUNED_PARAMS_PATH
    )

    print("\nSELECTED HYPERPARAMETERS")

    print("Treatment C:", treatment_best_c)

    print("Control C:", control_best_c)

    print("\nTuning results saved at:", TUNING_RESULTS_PATH)

    print("Selected parameters saved at:", TUNED_PARAMS_PATH)

    return selected_params, tuning_results


if __name__ == "__main__":

    run_tuning()

