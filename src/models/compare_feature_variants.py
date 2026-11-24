import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_validate

from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COL,
    RANDOM_STATE
)

FEATURE_EXPERIMENTS_DIR=(PROCESSED_DATA_DIR/"feature_experiments")

VARIANTS=[
    "baseline",
    "log_history",
    "interactions",
    "no_history_segment"
]

CATEGORICAL_COLUMNS=[
    "history_segment",
    "zip_code",
    "channel"
]

ID_COLUMN="source_row_index"

C_VALUE=1.0
CV_FOLDS=5
MAX_ITER=2000

RESULTS_PATH=(REPORTS_DIR/"feature_variant_cv_results.csv")
SUMMARY_PATH=(REPORTS_DIR/"feature_variant_summary.csv")

def build_processor(numerical_features,categorical_features):
    numerical_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scalar",StandardScaler())
    ])

    try:
        encoder=OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

    except TypeError:
        encoder=OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )

    categorical_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",encoder)
    ])

    preprocessor=ColumnTransformer(transformers=[
        ("num",numerical_pipeline,numerical_features),
        ("cat",categorical_pipeline,categorical_features)
    ],remainder="drop"
    )

    return preprocessor

def load_feature_variant(variant_name):
    file_path=(
        FEATURE_EXPERIMENTS_DIR/variant_name/"train_data.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:{file_path}"
        )

    df=pd.read_csv(file_path)

    required_metadata=[
        ID_COLUMN,
        TARGET_COL,
        "treatment"
    ]

    missing_columns=[
        col for col in required_metadata
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing metadata in {variant_name}:{missing_columns}"
        )

    metadata_columns=set(required_metadata)

    feature_columns=[
        col for col in df.columns
        if col not in metadata_columns
    ]

    if not feature_columns:
        raise ValueError(
            f"No predictive features found in {variant_name}"
        )

    if df.isna().any().any():
        raise ValueError(
            f"Missing values found in {variant_name}"
        )

    y=df[TARGET_COL].astype(int)

    treatment=df["treatment"].astype(int)

    if not y.isin([0,1]).all():
        raise ValueError("Invalid conversion values")

    if not treatment.isin([0,1]).all():
        raise ValueError("Invalid treatment values")

    X=df[feature_columns].copy()

    categorical_features=[
        col for col in feature_columns
        if col in CATEGORICAL_COLUMNS
    ]

    numerical_features=[
        col for col in feature_columns
        if col not in CATEGORICAL_COLUMNS
    ]

    for col in numerical_features:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise ValueError(
                f"Numerical feature {col} is not numeric"
            )

    if not np.isfinite(X[numerical_features].to_numpy(dtype=float)).all():
        raise ValueError(
            f"Non finite numerical values in {variant_name}"
        )

    return(
        df,
        X,
        y.to_numpy(),
        treatment.to_numpy(),
        numerical_features,
        categorical_features
    )

#Compare One Model Arm
def evaluate_arm(
        X_arm,
        y_arm,
        arm_name,
        variant_name,
        numerical_features,
        categorical_features
):

    if len(np.unique(y_arm))!=2:
        raise ValueError(
            f"{arm_name} in {variant_name} must contain both classes"
        )

    class_counts=np.bincount(y_arm)

    if class_counts.min()<CV_FOLDS:
        raise ValueError(
            f"Insufficient minority-class operations for {CV_FOLDS}-fold CV:{arm_name}"
        )

    preprocessor=build_processor(
        numerical_features,
        categorical_features
    )

    pipeline=Pipeline([
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(
            C=C_VALUE,
            max_iter=MAX_ITER,
            solver="lbfgs",
            class_weight=None
        ))
    ])

    cv=StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_results=cross_validate(
        estimator=pipeline,
        X=X_arm,
        y=y_arm,
        scoring="neg_log_loss",
        cv=cv,
        return_train_score=True,
        n_jobs=-1,
        error_score="raise"
    )

    mean_cv_loss=-np.mean(cv_results["test_score"])

    std_cv_loss=np.std(cv_results["test_score"],ddof=1)

    mean_train_loss=-np.mean(cv_results["train_score"])

    result={
        "variant":variant_name,
        "model":arm_name,
        "sample_size":len(y_arm),
        "conversions":int(y_arm.sum()),
        "conversion_rate":float(y_arm.mean()),
        "feature_count":(
            len(numerical_features)+len(categorical_features)
        ),
        "mean_cv_log_loss":mean_cv_loss,
        "std_cv_log_loss":std_cv_loss,
        "mean_train_log_loss":mean_train_loss
    }

    return result


def run_feature_comparison():
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results=[]

    reference_signature=None

    for variant_name in VARIANTS:
        print("\n"+"="*55)
        print("Feature Variant:",variant_name.upper())

        (df,X,y,treatment,numerical_features,categorical_features)=load_feature_variant(variant_name)

        print("Customers:", len(df))

        print("Numerical features:", numerical_features)

        print("Categorical features:", categorical_features)

        print("Total predictive features:", X.shape[1])

        signature=df[
            [ID_COLUMN,TARGET_COL,"treatment"]
        ].reset_index(drop=True)

        if reference_signature is None:
            reference_signature=signature.copy()

        elif not signature.equals(reference_signature):
            raise ValueError(
                f"Customer order,outcomes or treatment assignments differ in {variant_name}"
            )

        treatment_mask=treatment==1

        treatment_result=evaluate_arm(
            X.loc[treatment_mask].reset_index(drop=True),
            y[treatment_mask],
            "Treatment",
            variant_name,
            numerical_features,
            categorical_features
        )

        print("\nTreatment CV Log Loss:",round(treatment_result["mean_cv_log_loss"],6))

        results.append(treatment_result)

        control_mask=treatment==0

        control_result=evaluate_arm(
            X.loc[control_mask].reset_index(drop=True),
            y[control_mask],
            "Control",
            variant_name,
            numerical_features,
            categorical_features
        )

        print("Control CV Log Loss:",round(control_result["mean_cv_log_loss"],6))

        results.append(control_result)


    results_df=pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH,index=False)

    summary=results_df.pivot(
        index="variant",
        columns="model",
        values="mean_cv_log_loss"
    ).reset_index()

    summary=summary.rename(columns={
        "Treatment":"treatment_cv_log_loss",
        "Control":"control_cv_log_loss"
    })

    treatment_counts=(results_df[
        results_df["model"]=="Treatment"
    ].set_index("variant")["sample_size"])

    control_counts=(
        results_df[results_df["model"]=="Control"].set_index("variant")["sample_size"]
    )

    summary["weighted_cv_log_loss"]=(
        summary["treatment_cv_log_loss"]
        *
        summary["variant"].map(treatment_counts)
        +
        summary["control_cv_log_loss"]
        *
        summary["variant"].map(control_counts)
    )/(
        summary["variant"].map(treatment_counts)
        +
        summary["variant"].map(control_counts)
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print("\n"+"="*55)
    print("\nFeature Comparison Results")
    print(
        results_df.round(6).to_string(index=False)
    )

    print("\nCOMBINED FEATURE SUMMARY")

    print(
        summary.sort_values(
            "weighted_cv_log_loss"
        ).round(6).to_string(index=False)
    )

    print("\nBEST TREATMENT FEATURE VARIANT:")

    best_treatment = results_df[
        results_df["model"] == "Treatment"
    ].sort_values("mean_cv_log_loss").iloc[0]

    print(
        best_treatment["variant"],
        "| Log Loss:",
        round(best_treatment["mean_cv_log_loss"], 6)
    )

    print("\nBEST CONTROL FEATURE VARIANT:")

    best_control = results_df[
        results_df["model"] == "Control"
    ].sort_values("mean_cv_log_loss").iloc[0]

    print(
        best_control["variant"],
        "| Log Loss:",
        round(best_control["mean_cv_log_loss"], 6)
    )

    print("\nDetailed results saved at:", RESULTS_PATH)

    print("Summary saved at:", SUMMARY_PATH)

    return results_df, summary


if __name__ == "__main__":

    run_feature_comparison()
