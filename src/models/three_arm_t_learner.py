import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss

from src.config import(
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

PREPROCESSOR_PATH=THREE_ARM_ARTIFACTS_DIR/"logistic_t_learner_preprocessor.joblib"
MODEL_PATH=THREE_ARM_ARTIFACTS_DIR/"logistic_t_learner_bundle.joblib"
PARAMS_PATH=THREE_ARM_ARTIFACTS_DIR/"logistic_t_learner_params.joblib"
DIAGNOSTIC_PATH=THREE_ARM_REPORTS_DIR/"logistic_t_learner_diagnostics.csv"
PREDICTIONS_PATH=THREE_ARM_REPORTS_DIR/"logistic_t_learner_validation_predictions.csv"

ARM_LABELS={
    0:"No Email",
    1:"Mens E-Mail",
    2:"Womens E-Mail"
}

ARM_IDS=tuple(ARM_LABELS)

CONTROL_C=0.1
EMAIL_TREATMENT_C=0.01
MAX_ITER=2000
C_BY_ARM={
    0:CONTROL_C,
    1:EMAIL_TREATMENT_C,
    2:EMAIL_TREATMENT_C
}

def _take_rows(X,indices):
    if hasattr(X,"iloc"):
        return X.iloc[indices]

    return X[indices]

class ThreeArmTLearner:
    def __init__(self,c_by_arm=None,max_iter=MAX_ITER,random_state=RANDOM_STATE):
        self.c_by_arm=dict(C_BY_ARM if c_by_arm is None else c_by_arm)
        self.max_iter=max_iter
        self.random_state=random_state
        self.models={
            arm:LogisticRegression(
                C=self.c_by_arm[arm],
                max_iter=self.max_iter,
                solver="lbfgs",
                class_weight=None,
                random_state=self.random_state
            )
            for arm in ARM_IDS
        }

        self.is_fitted=False

    def fit(self,X,y,treatment):
        y=np.asarray(y,dtype=int)

        treatment=np.asarray(treatment,dtype=int)

        if len(X)!=len(y) or len(y)!=len(treatment):
            raise ValueError("X, y, and treatment must have equal lengths.")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("Conversion must contain only 0 and 1.")
        if set(np.unique(treatment)) != set(ARM_IDS):
            raise ValueError("Training data must contain treatment arms 0, 1, and 2.")

        for arm in ARM_IDS:
            indices=np.flatnonzero(treatment==arm)
            y_arm=y[indices]

            if len(np.unique(y_arm))!=2:
                raise ValueError(
                    f"{ARM_LABELS[arm]} training data must contain both conversion classes."
                )

            print(f"\n{ARM_LABELS[arm]}")
            print("Customers:", len(indices))
            print("Conversions:", int(y_arm.sum()))
            print("C:", self.c_by_arm[arm])
            self.models[arm].fit(_take_rows(X, indices), y_arm)

        self.is_fitted = True
        return self

    def predict_potential_outcomes(self,X):
        if not self.is_fitted:
            raise ValueError("Fit the three-arm T-Learner before predicting")

        return {
            arm:self.models[arm].predict_proba(X)[:,1]
            for arm in ARM_IDS
        }

    def predict_uplifts(self,X):
        probabilities=self.predict_potential_outcomes(X)
        return{
            "mens_vs_no_email":probabilities[1]-probabilities[0],
            "womens_vs_no_email":probabilities[2]-probabilities[0]
        }

def load_split_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Three-arm dataset not found:{file_path}")

    df=pd.read_csv(file_path)

    required_cols=FEATURE_COLUMNS+[TARGET_COL,"treatment","source_row_index"]
    missing_cols=[col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in {file_path.name}:{missing_cols}")

    if df["source_row_index"].duplicated().any():
        raise ValueError(f"Duplicate source_row_index values in {file_path.name}")

    X=df[FEATURE_COLUMNS].copy()
    y=df[TARGET_COL].astype(int).to_numpy()
    treatment=df["treatment"].astype(int).to_numpy()

    if not set(np.unique(y)).issubset({0,1}):
        raise ValueError(f"Invalid conversion values in {file_path.name}.")

    if set(np.unique(treatment))!=set(ARM_IDS):
        raise ValueError(
            f"{file_path.name} must contain treatment values 0,1 and 2"
        )

    return df,X,y,treatment

def calculate_arm_diagnostic(model,X,y,treatment,arm,split_name):
    indices=np.flatnonzero(treatment==arm)
    y_arm=y[indices]
    if len(indices)==0 or len(np.unique(y_arm))!=2:
        raise ValueError(
            f"Both conversion classes are required for {ARM_LABELS[arm]} "
            f"{split_name} diagnostics."
        )

    probabilities=model.predict_proba(_take_rows(X,indices))[:,1]
    result={
        "model": "Logistic Regression T-Learner",
        "arm": ARM_LABELS[arm],
        "treatment": arm,
        "split": split_name,
        "sample_size": len(indices),
        "conversions": int(y_arm.sum()),
        "conversion_rate": float(y_arm.mean()),
        "log_loss": log_loss(y_arm, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y_arm, probabilities)
    }

    return result


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

    predictions["mens_uplift_percentage_points"]=(
        predictions["predicted_uplift_mens_vs_no_email"]*100
    )

    predictions["womens_uplift_percentage_points"]=(
        predictions["predicted_uplift_womens_vs_no_email"]*100
    )

    return predictions

def run_three_arm_t_learner():
    THREE_ARM_ARTIFACTS_DIR.mkdir(parents=True,exist_ok=True)
    THREE_ARM_REPORTS_DIR.mkdir(parents=True,exist_ok=True)

    print("\nTHREE ARM LOGISTIC REGRESSION T-LEARNER")
    train_df,X_train,y_train,treatment_train=load_split_data(TRAIN_PATH)
    val_df,X_val,y_val,treatment_val=load_split_data(VAL_PATH)

    print("Training customers:", len(train_df))
    print("Validation customers:", len(val_df))
    print("Training conversions:", int(y_train.sum()))

    preprocessor=build_preprocessor()
    X_train_processed=preprocessor.fit_transform(X_train)
    X_val_processed=preprocessor.transform(X_val)

    print("\nFEATURE TRANSFORMATION")
    print("Training shape:", X_train_processed.shape)
    print("Validation shape:", X_val_processed.shape)

    learner=ThreeArmTLearner(c_by_arm=C_BY_ARM)
    learner.fit(X_train_processed,y_train,treatment_train)

    diagnostic=[]
    for split_name,X_split,y_split,treatment_split in [
        ("Training",X_train_processed,y_train,treatment_train),
        ("Validation",X_val_processed,y_val,treatment_val)
    ]:
        for arm in ARM_IDS:
            diagnostic.append(
                calculate_arm_diagnostic(
                    learner.models[arm],
                    X_split,
                    y_split,
                    treatment_split,
                    arm,
                    split_name
                )
            )

    diagnostic_df=pd.DataFrame(diagnostic)
    print("\nMODEL DIAGNOSTICS")
    print(diagnostic_df.round(6).to_string(index=False))

    probabilities=learner.predict_potential_outcomes(X_val_processed)
    val_predictions=build_validation_predictions(
        val_df,treatment_val,y_val,probabilities
    )

    print("\nVALIDATION PREDICTION SUMMARY")
    print(
        "Mean predicted Men's uplift vs No Email (pp):",
        round(float(val_predictions["predicted_uplift_mens_vs_no_email"].mean() * 100), 4),
    )
    print(
        "Mean predicted Women's uplift vs No Email (pp):",
        round(float(val_predictions["predicted_uplift_womens_vs_no_email"].mean() * 100), 4),
    )
    print("\nPredicted best-campaign distribution:")
    print(val_predictions["predicted_best_campaign"].value_counts().to_string())

    joblib.dump(preprocessor,PREPROCESSOR_PATH)
    joblib.dump(
        {
            "preprocessor":preprocessor,
            "arm_models":learner.models,
            "feature_columns":FEATURE_COLUMNS,
            "arm_labels":ARM_LABELS
        },
        MODEL_PATH
    )
    joblib.dump(
        {
            "C_by_arm":C_BY_ARM,
            "max_iter":MAX_ITER,
            "random_state":RANDOM_STATE
        },
        PARAMS_PATH     
    )

    diagnostic_df.to_csv(DIAGNOSTIC_PATH,index=False)
    val_predictions.to_csv(PREDICTIONS_PATH,index=False)

    print("\nTHREE-ARM LOGISTIC T-LEARNER COMPLETED")
    print("Preprocessor saved at:", PREPROCESSOR_PATH)
    print("Model bundle saved at:", MODEL_PATH)
    print("Parameters saved at:", PARAMS_PATH)
    print("Diagnostics saved at:", DIAGNOSTIC_PATH)
    print("Validation predictions saved at:", PREDICTIONS_PATH)

    return learner, diagnostic_df, val_predictions


if __name__ == "__main__":
    run_three_arm_t_learner()