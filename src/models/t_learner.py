import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss

from src.config import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    TARGET_COL,
    RANDOM_STATE
)

from src.data_preprocessing import FEATURE_COLUMNS

TRAIN_PATH = PROCESSED_DATA_DIR / "train_data.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_data.csv"

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

MODEL_PATH = ARTIFACTS_DIR / "t_learner_tuned.joblib"

DIAGNOSTICS_PATH = REPORTS_DIR / "t_learner_tuned_diagnostics.csv"

PREDICTIONS_PATH = REPORTS_DIR / "t_learner_tuned_validation_predictions.csv"

TUNED_PARAMS_PATH=ARTIFACTS_DIR/"t_learner_tuned_params.joblib"

TREATMENT_C=0.01
CONTROL_C=0.1
MAX_ITER=2000

class Tlearner:
    def __init__(self,treatment_C=0.01,control_C=0.1,max_iter=2000):

        self.treatment_C=treatment_C
        self.control_C=control_C
        self.max_iter=max_iter

        self.treatment_model=LogisticRegression(
            C=self.treatment_C,
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight=None
        )

        self.control_model=LogisticRegression(
            C=self.control_C,
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight=None
        )

        self.is_fitted=False

    def fit(self,X,y,treatment):
        X=X
        y=np.asarray(y)
        treatment=np.asarray(treatment)
        if len(X)!=len(y) or len(y)!=len(treatment):
            raise ValueError(
                "X,y and treatment must have equal lengths"
            )

        if not set(np.unique(treatment)).issubset({0,1}):
            raise ValueError(
                "Treatment must contain only 0 and 1"
            )

        treatment_mask=treatment==1
        control_mask=treatment==0

        X_treatment=X[treatment_mask]
        X_control=X[control_mask]

        y_treatment=y[treatment_mask]
        y_control=y[control_mask]

        if len(np.unique(y_treatment)) != 2:
            raise ValueError(
                "Treatment group must contain both."
                "conversion class"
            )

        if len(np.unique(y_control)) != 2:
            raise ValueError(
                "Control group must contain both."
                "conversion classes."
            )

        print("\nTraining T-Learner")
        print("\nTreatment training customers:",len(y_treatment))
        print("Treatment conversion:",int(y_treatment.sum()))
        print("\nControl training customers:",len(y_control))
        print("Control conversion:",int(y_control.sum()))
        print("\n Training treatment model...")

        self.treatment_model.fit(X_treatment,y_treatment)
        self.control_model.fit(X_control,y_control)

        self.is_fitted=True

        return self

    def predict_potential_outcomes(self,X):
        if not self.is_fitted:
            raise ValueError(
                "Train the T-Learner before making predictions."
            )

        treatment_probability=(
            self.treatment_model.predict_proba(X)[:,1]
        )

        control_probability=(
            self.control_model.predict_proba(X)[:,1]
        )

        return treatment_probability,control_probability

    def predict_uplift(self,X):
        treatment_probability,control_probability=(self.predict_potential_outcomes(X))

        uplift=(treatment_probability-control_probability)

        return uplift

def load_split_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:{file_path}"
        )

    df=pd.read_csv(file_path)

    required_cols=(FEATURE_COLUMNS+[TARGET_COL,"treatment"])

    missing_cols=[
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns in {file_path.name}:{missing_cols}"
        )

    X=df[FEATURE_COLUMNS].copy()
    y=df[TARGET_COL].astype(int).copy()

    treatment=df["treatment"].astype(int).copy()

    if not set(y.unique()).issubset({0,1}):
        raise ValueError(
            f"Invalid conversion values in {file_path.name}"
        )

    if not set(treatment.unique()).issubset({0,1}):
        raise ValueError(
            f"Invalid treatment values in {file_path.name}"
        )

    return df,X,y,treatment

def calculate_model_diagnostics(
        model,
        X,
        y,
        treatment,
        model_name,
        split_name
):
    mask=treatment.to_numpy()==(
        1 if model_name=="Treatment" else 0
    )

    X_group=X[mask]
    y_group=y.to_numpy()[mask]

    if len(y_group)==0:
        raise ValueError(
            f"No observations available for {model_name}"
        )

    if len(np.unique(y_group))!=2:
        raise ValueError(
            f"Both conversion classes are needed for"
            f"{model_name} {split_name} diagnostics."
        )

    probabilities=model.predict_proba(X_group)[:,1]

    loss=log_loss(
        y_group,
        probabilities,
        labels=[0,1]
    )

    brier=brier_score_loss(y_group,probabilities)

    result={"model":model_name,
            "split":split_name,
            "sample_size":len(y_group),
            "conversions":int(y_group.sum()),
            "conversion_rate":y_group.mean(),
            "log_loss":loss,
            "brier_score":brier}

    return result

def run_t_learner():
    print("\nT-Learner")

    _,X_train,y_train,treatment_train=(load_split_data(TRAIN_PATH))

    val_df,X_val,y_val,treatment_val=(load_split_data(VALIDATION_PATH))

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Saved preprocessor not found."
            "Run Phase 2 first"
        )

    preprocessor=joblib.load(PREPROCESSOR_PATH)

    X_train_processed=preprocessor.transform(X_train)
    X_val_processed=preprocessor.transform(X_val)

    print("\nFEATURE TRANSFORMATION")
    print("Training shape:",X_train_processed.shape)
    print("Validation shape:",X_val_processed.shape)
    t_learner=Tlearner(
        treatment_C=TREATMENT_C,
        control_C=CONTROL_C,
        max_iter=MAX_ITER
    )
    t_learner.fit(X_train_processed,
                  y_train,
                  treatment_train)

    diagnostics=[]

    models=[
        ("Treatment",t_learner.treatment_model),
        ("Control",t_learner.control_model)
    ]

    datasets=[
        (
            "Training",
            X_train_processed,
            y_train,
            treatment_train
        ),

        (
            "Validation",
            X_val_processed,
            y_val,
            treatment_val
        )
    ]

    for split_name,X_split,y_split,treatment_split in datasets:
        for model_name,model in models:
            result=calculate_model_diagnostics(
                model,
                X_split,
                y_split,
                treatment_split,
                model_name,
                split_name
            )
            diagnostics.append(result)

    diagnostics_df=pd.DataFrame(diagnostics)
    print("\nMODEL DIAGNOSTICS")
    print(diagnostics_df.round(6).to_string(index=False))

    diagnostics_df.to_csv(DIAGNOSTICS_PATH,index=False)

    treatment_probability,control_probability=(
        t_learner.predict_potential_outcomes(X_val_processed)
    )

    uplift=(
        treatment_probability-control_probability
    )

    validation_predictions=val_df[["source_row_index"]+FEATURE_COLUMNS].copy()

    validation_predictions["actual_treatment"]=(treatment_val.to_numpy())

    validation_predictions["actual_conversion"]=(y_val.to_numpy())

    validation_predictions["predicted_purchase_if_treated"] = (treatment_probability)

    validation_predictions["predicted_purchase_if_control"] = (control_probability)

    validation_predictions["predicted_uplift"] = uplift

    validation_predictions["uplift_percentage_points"] = (uplift * 100)

    # Sort by predicted uplift
    validation_predictions = validation_predictions.sort_values(
        by="predicted_uplift",
        ascending=False
    )

    validation_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print("\nVALIDATION UPLIFT SUMMARY")
    print(
    "Average predicted uplift:",
    round(float(np.mean(uplift)) * 100, 4),
    "percentage points"
    )

    print(
        "Median predicted uplift:",
        round(float(np.median(uplift)) * 100, 4),
        "percentage points"
    )

    print(
        "Minimum predicted uplift:",
        round(float(np.min(uplift)) * 100, 4),
        "percentage points"
    )

    print(
        "Maximum predicted uplift:",
        round(float(np.max(uplift)) * 100, 4),
        "percentage points"
    )

    print(
        "Customers with positive predicted uplift:",
        int(np.sum(uplift > 0))
    )

    print(
        "Customers with negative predicted uplift:",
        int(np.sum(uplift < 0))
    )

    print("\nTop 10 customers by predicted uplift:")

    print(
        validation_predictions[
            [
                "source_row_index",
                "predicted_purchase_if_treated",
                "predicted_purchase_if_control",
                "uplift_percentage_points"
            ]
        ].head(10).round(4).to_string(index=False)
    )

    joblib.dump(t_learner,MODEL_PATH)

    joblib.dump(
    {
        "treatment_C": TREATMENT_C,
        "control_C": CONTROL_C,
        "max_iter": MAX_ITER
    },
    TUNED_PARAMS_PATH
)

    print("Tuned parameters saved at:", TUNED_PARAMS_PATH)
    print("Trained model saved at:", MODEL_PATH)

    print("Diagnostics saved at:", DIAGNOSTICS_PATH)

    print("Validation predictions saved at:", PREDICTIONS_PATH)

    return t_learner, diagnostics_df, validation_predictions

if __name__=="__main__":
    run_t_learner()