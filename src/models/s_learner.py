import numpy as np
import pandas as pd
import joblib

from scipy.sparse import issparse,csr_matrix,hstack

from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import log_loss,brier_score_loss

from src.config import(
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
MODEL_PATH=ARTIFACTS_DIR/"s_learner_tuned.joblib"
CV_RESULTS_PATH=REPORTS_DIR/"s_learner_c_tuning.csv"
DIAGNOSTICS_PATH=REPORTS_DIR/"s_learner_tuned_diagnostics.csv"
PREDICTIONS_PATH=(REPORTS_DIR/"s_learner_tuned_validation_predictions.csv")

C_VALUES=[0.001,0.01,0.1,1.0,10]
MAX_ITER=2000
CV_FOLDS=5
TUNED_PARAMS_PATH=ARTIFACTS_DIR/"s_learner_tuned_params.joblib"

class SLearner(ClassifierMixin,BaseEstimator):
    def __init__(self,preprocessor,C=1.0,max_iter=2000):
        self.preprocessor=preprocessor
        self.C=C
        self.max_iter=max_iter

    def create_design_matrix(self,X_processed,treatment):
        treatment=np.asarray(treatment).reshape(-1,1)

        if issparse(X_processed):
            treatment_column=csr_matrix(treatment)

            treatment_interactions=(X_processed.multiply(treatment))

            design_matrix=hstack([
                X_processed,
                treatment_column,
                treatment_interactions
            ],
            format="csr")

        else:
            treatment_interactions=(X_processed*treatment)

            design_matrix=np.hstack([
                X_processed,
                treatment,
                treatment_interactions
            ])

        return design_matrix

    def fit(self,X,y):
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
                "S-Learner expects a pandas Dataframe"
            )

        if "treatment" not in X.columns:
            raise ValueError(
                "Treatment column is missing"
            )

        treatment=X["treatment"].to_numpy(dtype=int)

        y=np.asarray(y,dtype=int)

        if not np.isin(treatment,[0,1]).all():
            raise ValueError(
                "Treatment must contain only 0 and 1"
            )

        if not np.isin(y,[0,1]).all():
            raise ValueError(
                "Conversion must contain only 0 and 1"
            )

        if len(X)!=len(y):
            raise ValueError(
                "X and y must have equal lengths"
            )

        customer_features=X.drop(columns=["treatment"])

        #Clone the preprocessor

        self.preprocessor_=clone(self.preprocessor)

        X_processed=self.preprocessor_.fit_transform(
            customer_features,
            y
        )

        design_matrix=self.create_design_matrix(
            X_processed,
            treatment
        )

        #One shared logistic regression model
        self.model_=LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight=None
        )

        self.model_.fit(
            design_matrix,
            y
        )

        self.classes_=self.model_.classes_
        return self

    def predict_proba(self,X):
        check_is_fitted(
            self,
            ["preprocessor_","model_"]
        )

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
                "S-Learner expects a pandas DataFrame"
            )

        if "treatment" not in X.columns:
            raise ValueError(
                "Treatment column is missing"
            )

        treatment=X["treatment"].to_numpy(dtype=int)

        if not np.isin(treatment,[0,1]).all():
            raise ValueError(
                "Treatment must contain only 0 and 1"
            )

        customer_features=X.drop(columns=["treatment"])

        X_processed=self.preprocessor_.transform(customer_features)
        design_matrix=self.create_design_matrix(X_processed,treatment)

        return self.model_.predict_proba(design_matrix)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return self.classes_[
            np.argmax(probabilities, axis=1)
        ]


def load_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:{TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation data not found:{VALIDATION_PATH}"
        )

    train_df=pd.read_csv(TRAIN_PATH)
    val_df=pd.read_csv(VALIDATION_PATH)

    required_columns=(FEATURE_COLUMNS+[TARGET_COL,"treatment"])

    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(
                f"Missing training column:{col}"
            )

        if col not in val_df.columns:
            raise ValueError(
                f"Missing validqation column:{col}"
            )

    X_train=train_df[FEATURE_COLUMNS].copy()
    y_train=train_df[TARGET_COL].astype(int).to_numpy()

    treatment_train=(
        train_df["treatment"].astype(int).to_numpy()
    )

    X_val=val_df[FEATURE_COLUMNS].copy()
    y_val=val_df[TARGET_COL].astype(int).to_numpy()

    treatment_val=(
        val_df["treatment"].astype(int).to_numpy()
    )

    return (
        train_df,
        val_df,
        X_train,
        y_train,
        treatment_train,
        X_val,
        y_val,
        treatment_val
    )

def run_S_learner():
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

    print("Training Customers:",len(X_train))
    print("Validation Customers:",len(X_val))
    print("Training Conversions:",int(y_train.sum()))
    print("Treatment Customers:",int(np.sum(treatment_train==1)))
    print("Control Customers:",int(np.sum(treatment_train==0)))

    X_train_s=X_train.copy()
    X_train_s["treatment"]=treatment_train
    strata=np.char.add(np.char.add(treatment_train.astype(str),"_"),y_train.astype(str))

    cv=StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_splits=list(cv.split(X_train_s,strata))

    preprocessor=build_preprocessor()

    s_learner=SLearner(
        preprocessor=preprocessor,
        max_iter=MAX_ITER
    )

    print("\nS-Learner Cross Validation")

    param_grid={"C":C_VALUES}

    grid_search=GridSearchCV(
        estimator=s_learner,
        param_grid=param_grid,
        cv=cv_splits,
        scoring="neg_log_loss",
        return_train_score=True,
        n_jobs=1,
        refit=True,
        error_score="raise"
    )

    grid_search.fit(X_train_s,y_train)

    results=grid_search.cv_results_

    cv_df = pd.DataFrame({
        "C": [params["C"] for params in results["params"]],
        "mean_cv_log_loss": -results["mean_test_score"],
        "std_cv_log_loss": results["std_test_score"],
        "mean_train_log_loss": -results["mean_train_score"],
        "rank": results["rank_test_score"]
    })

    cv_df = cv_df.sort_values("C")

    cv_df.to_csv(
        CV_RESULTS_PATH,
        index=False
    )

    best_C = grid_search.best_params_["C"]

    best_index = grid_search.best_index_

    best_cv_loss = -results["mean_test_score"][best_index]

    best_cv_std = results["std_test_score"][best_index]

    best_train_loss = -results["mean_train_score"][best_index]

    print("\nC TUNING RESULTS")
    print(cv_df.round(6).to_string(index=False))

    print("\nBEST S-LEARNER PARAMETERS")

    print("Best C:", best_C)

    print("Mean CV Log Loss:", round(float(best_cv_loss), 6))

    print("CV Log Loss Standard Deviation:", round(float(best_cv_std), 6))

    print("Mean Training Log Loss:", round(float(best_train_loss), 6))

    # Save selected parameters
    joblib.dump(
        {
            "best_C": best_C,
            "best_cv_log_loss": best_cv_loss,
            "cv_folds": CV_FOLDS,
            "scoring": "neg_log_loss"
        },
        TUNED_PARAMS_PATH
    )

    # GridSearchCV automatically refits the best estimator
    # on the complete training dataset when refit=True.
    final_model = grid_search.best_estimator_

    X_val_treated=X_val.copy()
    X_val_treated["treatment"]=1

    X_val_control=X_val.copy()
    X_val_control["treatment"]=0

    probability_treated=(
        final_model.predict_proba(X_val_treated)[:,1]
    )

    probability_control=(
        final_model.predict_proba(X_val_control)[:,1]
    )

    predicted_uplift=probability_treated-probability_control

    treatment_mask=treatment_val==1
    control_mask=treatment_val==0

    treatment_log_loss=log_loss(
        y_val[treatment_mask],
        probability_treated[treatment_mask],
        labels=[0,1]
    )

    control_log_loss=log_loss(
        y_val[control_mask],
        probability_control[control_mask],
        labels=[0,1]
    )

    treatment_brier=brier_score_loss(
        y_val[treatment_mask],
        probability_treated[treatment_mask]
    )

    control_brier=brier_score_loss(
        y_val[control_mask],
        probability_control[control_mask]
    )

    diagnostics_df=pd.DataFrame([
    {
        "model":"S-Learner",
        "arm":"Treatment",
        "sample_size":int(treatment_mask.sum()),
        "conversions":int(y_val[treatment_mask].sum()),
        "log_loss":treatment_log_loss,
        "brier_score":treatment_brier
    },
    {
        "model": "S-Learner",
        "arm": "Control",
        "sample_size": int(control_mask.sum()),
        "conversions": int(y_val[control_mask].sum()),
        "log_loss": control_log_loss,
        "brier_score": control_brier
    }
    ])

    diagnostics_df.to_csv(DIAGNOSTICS_PATH,index=False)

    print("\nVALIDATION DIAGNOSTICS")

    print(
        diagnostics_df.round(6).to_string(index=False)
    )

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

    # --------------------------------------------------
    # UPLIFT SUMMARY
    # --------------------------------------------------

    print("\nS-LEARNER VALIDATION UPLIFT SUMMARY")

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

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_PATH
    )

    print("\nS-LEARNER TRAINING COMPLETED")

    print("Model saved at:", MODEL_PATH)

    print("CV results saved at:", CV_RESULTS_PATH)

    print("Diagnostics saved at:", DIAGNOSTICS_PATH)

    print("Validation predictions saved at:", PREDICTIONS_PATH)

    return (
        final_model,
        cv_df,
        diagnostics_df,
        validation_predictions
    )


if __name__ == "__main__":

    run_S_learner()
