import joblib 
import numpy as np
import pandas as pd


from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss,log_loss
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    RANDOM_STATE
)

from src.data_preprocessing import FEATURE_COLUMNS,build_preprocessor

THREE_ARM_DATA_DIR=PROCESSED_DATA_DIR/"three_arm"
THREE_ARM_ARTIFACTS_DIR=ARTIFACTS_DIR/"three_arm"
THREE_ARM_REPORTS_DIR=REPORTS_DIR/"three_arm"

TRAIN_PATH=THREE_ARM_DATA_DIR/"train_data.csv"
VAL_PATH=THREE_ARM_DATA_DIR/"validation_data.csv"

MODEL_PATH=THREE_ARM_ARTIFACTS_DIR/"lightgbm_t_learner_bundle.joblib"
PARAMS_PATH=THREE_ARM_ARTIFACTS_DIR/"lightgbm_t_learner_params.joblib"
DIAGNOSTICS_PATH=THREE_ARM_REPORTS_DIR/"lightgbm_t_learner_diagnostics.csv"
PREDICTIONS_PATH=THREE_ARM_REPORTS_DIR/"lightgbm_t_learner_validation_predictions.csv"

ARM_LABELS={
    0:"No Email",
    1:"Mens E-Mail",
    2:"Womens E-Mail"
}

ARM_IDS=tuple(ARM_LABELS)

CV_FOLDS=5
PARAM_GRID=[
    {
        "model__n_estimators":[50,100],
        "model__learning_rate":[0.03,0.05],
        "model__num_leaves":[3],
        "model__max_depth":[2,3]
    },
    {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.03, 0.05],
        "model__num_leaves": [7],
        "model__max_depth": [3]
    }
]


def _take_rows(X,indices):
    if hasattr(X,"iloc"):
        return X.iloc[indices]

    return X[indices]

def load_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Three-arm training data not found: {TRAIN_PATH}")

    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Three-arm validation data not found: {VAL_PATH}")

    train_df=pd.read_csv(TRAIN_PATH)
    val_df=pd.read_csv(VAL_PATH)
    required_cols=FEATURE_COLUMNS+[TARGET_COL,"treatment","source_row_index"]

    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing training column:{col}")

        if col not in val_df.columns:
            raise ValueError(f"Missing validation column:{col}")

    if train_df["source_row_index"].duplicated().any():
        raise ValueError("Duplicate source_row_index values in three-arm training data.")
    if val_df["source_row_index"].duplicated().any():
        raise ValueError("Duplicate source_row_index values in three-arm validation data.")

    X_train=train_df[FEATURE_COLUMNS].copy()
    y_train=train_df[TARGET_COL].astype(int).to_numpy()
    treatment_train=train_df["treatment"].astype(int).to_numpy()

    X_val=val_df[FEATURE_COLUMNS].copy()
    y_val=val_df[TARGET_COL].astype(int).to_numpy()
    treatment_val=val_df["treatment"].astype(int).to_numpy()

    for split_name,y,treatment in [
        ("training",y_train,treatment_train),
        ("validation",y_val,treatment_val)
    ]:

        if not set(np.unique(y)).issubset({0,1}):
            raise ValueError("Invalid conversion values in {split_name} data")

        if set(np.unique(treatment))!=set(ARM_IDS):
            raise ValueError(
                f"{split_name.title()} data must contain treatment values 0,1 and 2"
            )

    return train_df,val_df,X_train,y_train,treatment_train,X_val,y_val,treatment_val

def build_lightgbm_pipeline():
    model=LGBMClassifier(
        objective="binary",
        min_child_samples=50,
        reg_lambda=5.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1
    )

    return Pipeline([
        ("preprocessor",build_preprocessor()),
        ("model",model)
    ])

def train_arm(X,y,arm_id):
    arm_name=ARM_LABELS[arm_id]
    print(f"\n{'=' * 55}")
    print(f"TRAINING LIGHTGBM MODEL: {arm_name.upper()}")
    print(f"{'=' * 55}")

    y=np.asarray(y,dtype=int)
    conversions=int(y.sum())
    counts=np.bincount(y,minlength=2)

    print("Customers:", len(y))
    print("Conversions:", conversions)
    print("Conversion rate:", round(float(y.mean()), 6))

    if len(np.unique(y))!=2:
        raise ValueError(f"{arm_name} training data must contain both outcome classes.")

    if counts.min()<CV_FOLDS:
        raise ValueError(
            f"Not enough observations in each outcome class for {CV_FOLDS}-fold CV in {arm_name}"
        )

    cv=StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search=GridSearchCV(
        estimator=build_lightgbm_pipeline(),
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
        "mean_cv_log_loss": -results["mean_test_score"],
        "std_cv_log_loss": results["std_test_score"],
        "mean_train_log_loss": -results["mean_train_score"],
        "rank": results["rank_test_score"],
        "params": [str(params) for params in results["params"]]
    }).sort_values("rank")

    print("\nCROSS-VALIDATION RESULTS")
    print(cv_df.head(12).round(6).to_string(index=False))
    print("\nBEST PARAMETERS:")
    print(grid_search.best_params_)
    print("Best CV Log Loss:", round(float(-grid_search.best_score_), 6))
    print(
        "CV Standard Deviation:",
        round(float(results["std_test_score"][grid_search.best_index_]), 6),
    )

    return grid_search.best_estimator_,cv_df,grid_search.best_params_

def calculate_arm_diagnostics(model,X,y,treatment,arm,split_name):
    indices=np.flatnonzero(treatment==arm)
    y_arm=y[indices]
    if len(indices)==0 or len(np.unique(y_arm))!=2:
        raise ValueError(
            f"Both conversion classes are required for {ARM_LABELS[arm]} "
            f"{split_name} diagnostics."
        )

    probabilities=model.predict_proba(_take_rows(X,indices))[:,1]

    return {
        "model": "LightGBM T-Learner",
        "arm": ARM_LABELS[arm],
        "treatment": arm,
        "split": split_name,
        "sample_size": len(indices),
        "conversions": int(y_arm.sum()),
        "conversion_rate": float(y_arm.mean()),
        "log_loss": log_loss(y_arm, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y_arm, probabilities)
    }

def build_validation_predictions(val_df,treatment,y,probabilities):
    prediction_matrix=np.column_stack([probabilities[arm] for arm in ARM_IDS])
    best_arm=np.argmax(prediction_matrix,axis=1)
    row_indices=np.arange(len(val_df))

    predictions=pd.DataFrame({
        "source_row_index": val_df["source_row_index"].to_numpy(),
        "actual_treatment": treatment,
        "actual_treatment_label": [ARM_LABELS[arm] for arm in treatment],
        "actual_conversion": y,
        "predicted_conversion_no_email": probabilities[0],
        "predicted_conversion_mens_email": probabilities[1],
        "predicted_conversion_womens_email": probabilities[2],
        "predicted_uplift_mens_vs_no_email": probabilities[1] - probabilities[0],
        "predicted_uplift_womens_vs_no_email": probabilities[2] - probabilities[0],
        "predicted_best_treatment": best_arm,
        "predicted_best_campaign": [ARM_LABELS[arm] for arm in best_arm],
        "predicted_best_conversion_probability": prediction_matrix[row_indices, best_arm]
    })

    predictions["means_uplift_percentage_points"]=(
        predictions["predicted_uplift_mens_vs_no_email"]*100
    )
    predictions["womens_uplift_percentage_points"]=(
        predictions["predicted_uplift_womens_vs_no_email"]*100
    )

    return predictions

def run_three_arm_lightgbm_t_learner():
    THREE_ARM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    THREE_ARM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    (
        train_df,
        val_df,
        X_train,
        y_train,
        treatment_train,
        X_val,
        y_val,
        treatment_val,
    ) = load_data()

    print("\nTHREE-ARM LIGHTGBM T-LEARNER")
    print("Training customers:", len(train_df))
    print("Validation customers:", len(val_df))
    print("Training conversions:", int(y_train.sum()))

    arm_models = {}
    cv_results = {}
    best_params = {}

    for arm in ARM_IDS:
        indices = np.flatnonzero(treatment_train == arm)
        X_arm = _take_rows(X_train, indices)
        y_arm = y_train[indices]

        print("\nArm customers:", len(indices))
        print("Arm conversions:", int(y_arm.sum()))

        model, cv_df, params = train_arm(X_arm, y_arm, arm)
        arm_models[arm] = model
        cv_results[arm] = cv_df
        best_params[arm] = params

        cv_path = THREE_ARM_REPORTS_DIR / f"lightgbm_{arm}_cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        print("CV results saved at:", cv_path)

    probabilities = {
        arm: arm_models[arm].predict_proba(X_val)[:, 1]
        for arm in ARM_IDS
    }

    diagnostics = []
    for split_name, X_split, y_split, treatment_split in [
        ("Training", X_train, y_train, treatment_train),
        ("Validation", X_val, y_val, treatment_val),
    ]:
        for arm in ARM_IDS:
            diagnostics.append(
                calculate_arm_diagnostics(
                    arm_models[arm],
                    X_split,
                    y_split,
                    treatment_split,
                    arm,
                    split_name,
                )
            )

    diagnostics_df = pd.DataFrame(diagnostics)
    print("\nMODEL DIAGNOSTICS")
    print(diagnostics_df.round(6).to_string(index=False))

    validation_predictions = build_validation_predictions(
        val_df, treatment_val, y_val, probabilities
    )

    print("\nVALIDATION PREDICTION SUMMARY")
    print(
        "Mean predicted Men's uplift vs No Email (pp):",
        round(float(validation_predictions["predicted_uplift_mens_vs_no_email"].mean() * 100), 4),
    )
    print(
        "Mean predicted Women's uplift vs No Email (pp):",
        round(float(validation_predictions["predicted_uplift_womens_vs_no_email"].mean() * 100), 4),
    )
    print("\nPredicted best-campaign distribution:")
    print(validation_predictions["predicted_best_campaign"].value_counts().to_string())

    model_bundle = {
        "model_type": "Three-arm LightGBM T-Learner",
        "arm_models": arm_models,
        "feature_columns": FEATURE_COLUMNS,
        "arm_labels": ARM_LABELS,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    joblib.dump(best_params, PARAMS_PATH)
    diagnostics_df.to_csv(DIAGNOSTICS_PATH, index=False)
    validation_predictions.to_csv(PREDICTIONS_PATH, index=False)

    print("\nTHREE-ARM LIGHTGBM T-LEARNER COMPLETED")
    print("Model bundle saved at:", MODEL_PATH)
    print("Parameters saved at:", PARAMS_PATH)
    print("Diagnostics saved at:", DIAGNOSTICS_PATH)
    print("Validation predictions saved at:", PREDICTIONS_PATH)

    return model_bundle, cv_results, diagnostics_df, validation_predictions


if __name__ == "__main__":
    run_three_arm_lightgbm_t_learner()