import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.config import(
    DATA_PATH,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    SEGMENT_COL,
    TARGET_COL,
    TREATMENT_SEGMENT,
    CONTROL_SEGMENT,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE
)

NUMERICAL_FEATURES=[
    "recency",
    "history",
    "mens",
    "womens",
    "newbie"
]

CATEGORICAL_FEATURES=[
    "history_segment",
    "zip_code",
    "channel"
]

FEATURE_COLUMNS=(NUMERICAL_FEATURES+CATEGORICAL_FEATURES)

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:{DATA_PATH}"
        )

    df=pd.read_csv(DATA_PATH)

    required_columns=(FEATURE_COLUMNS+[SEGMENT_COL,TARGET_COL])

    missing_columns=[
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns:{missing_columns}"
        )

    print("Dataset Shape:",df.shape)

    return df

def experiment_dataset(df):
    print("\nEXPERIMENT DATASET")

    experiment_df=df[df[SEGMENT_COL].isin([TREATMENT_SEGMENT,CONTROL_SEGMENT])].copy()

    if experiment_df.empty:
        raise ValueError(
            "No treatment or control records found."
        )

    present_segments=set(experiment_df[SEGMENT_COL].unique())

    expected_segments={TREATMENT_SEGMENT,CONTROL_SEGMENT}

    if present_segments!=expected_segments:
        raise ValueError(
            "Treatment or control group is missing."
        )

    experiment_df["treatment"]=(
        experiment_df[SEGMENT_COL]==TREATMENT_SEGMENT
    ).astype(int)

    print("\nExp Dataset Shape:",experiment_df.shape)
    print("\nCampaign Distribution:",experiment_df[SEGMENT_COL].value_counts())
    print("\nTreatment Distribution:",experiment_df["treatment"].value_counts())

    return experiment_df

#X is customer characteristics
#Y is actual conversion outcome
#treatment(T) whether customer received email

def create_features_and_targets(experiment_df):
    print("\nFEATURE SEPERATION")

    X=experiment_df[FEATURE_COLUMNS].copy()
    Y=experiment_df[TARGET_COL].astype(int).copy()
    treatment=experiment_df["treatment"].astype(int).copy()

    if not set(Y.unique()).issubset({0,1}):
        raise ValueError(
            "Treatment must contain only 0 and 1"
        )

    print("\nFeature shape:",X.shape)
    print("Target shape:",Y.shape)
    print("Treatment shape",treatment.shape)

    print("\nNumerical Features:",NUMERICAL_FEATURES)
    print("\nCategorical Features:",CATEGORICAL_FEATURES)

    return X,Y,treatment

def split_data(X,Y,treatment):
    print("\nDATASET SPLITTING")

    if not(
        0<TEST_SIZE<1
        and 0<VALIDATION_SIZE<1
        and TEST_SIZE+VALIDATION_SIZE<1
    ):
        raise ValueError(
            "Invalid split configuration"
        )

    strata=(treatment.astype(str)+"_"+Y.astype(str))#Creates a combined treatment-outcome strata

    if strata.value_counts().min()<2:
        raise ValueError(
            "Insufficient samples in a treatment-outcome group."
        )

#FIRST SPLIT:80% REMAINING + 20% TEST
    (
        X_train_val,
        X_test,
        Y_train_val,
        Y_test,
        treatment_train_val,
        treatment_test
    )=train_test_split(X,Y,treatment,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=strata)

#SECOND SPLIT:60% TRAIN + 20% VALIDATION
    validation_fraction=(
        VALIDATION_SIZE/(1-TEST_SIZE)
    )

    strata_train_val=(treatment_train_val.astype(str)+"_"+Y_train_val.astype(str))

    if strata_train_val.value_counts().min()<2:
        raise ValueError(
            "Insufficient samples for validation splitting."
        )

    (
        X_train,
        X_val,
        Y_train,
        Y_val,
        treatment_train,
        treatment_val
    )=train_test_split(
        X_train_val,
        Y_train_val,
        treatment_train_val,

        test_size=validation_fraction,

        random_state=RANDOM_STATE,

        stratify=strata_train_val
    )

    total=len(X)
    print("\nTraining dataset:")
    print(
        f"{len(X_train)} customers"
        f"({len(X_train)/total*100:.2f}%)"
    )

    print("\nValidation dataset:")
    print(
        f"{len(X_val)}customers"
        f"({len(X_val)/total*100:.2f}%)"
    )

    print("\nTesting dataset:")
    print(
        f"{len(X_test)}customers"
        f"({len(X_test)/total*100:.2f}%)"
    )

    print("\nTraining treatment-outcome distribution:")
    print(pd.crosstab(
        treatment_train,
        Y_train,
        rownames=["Treatment"],
        colnames=["Conversion"]
    ))

    print("\nValidation treatment-outcome distribution:")
    print(pd.crosstab(
        treatment_val,
        Y_val,
        rownames=["Treatment"],
        colnames=["Conversion"]
    ))

    print("\nTesting treatment-outcome distribution:")
    print(pd.crosstab(
        treatment_test,
        Y_test,
        rownames=["Treatment"],
        colnames=["Conversion"]
    ))

    return (
        X_train,
        X_val,
        X_test,

        Y_train,
        Y_val,
        Y_test,

        treatment_train,
        treatment_val,
        treatment_test
    )

def build_preprocessor():
    numerical_pipeline=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
        ]
    )

    #OneHotEncoder compatibility
    try:
        encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=True)

    except TypeError:
        encoder=OneHotEncoder(handle_unknown="ignore",sparse=True)

    categorical_pipeline=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),

            ("encoder",encoder)
        ]
    )

    preprocessor=ColumnTransformer(
        transformers=[
            ("num",numerical_pipeline,NUMERICAL_FEATURES),
            ("cat",categorical_pipeline,CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor

def save_split(name,features,target,treatment):
    export_df=features.copy()
    export_df[TARGET_COL]=target
    export_df["treatment"]=treatment
    output_path=PROCESSED_DATA_DIR/f"{name}_data.csv"

    export_df.to_csv(output_path,index=True,index_label="source_row_index")

    print(f"{name} dataset saved at:{output_path}")

def preprocess_data():
    df=load_data()
    experiment_df=experiment_dataset(df)

    X,Y,treatment=create_features_and_targets(experiment_df)

    (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,

        treatment_train,
        treatment_val,
        treatment_test
    )=split_data(X,Y,treatment)

    preprocessor=build_preprocessor()

    X_train_processed=preprocessor.fit_transform(X_train)
    X_val_processed=preprocessor.transform(X_val)
    X_test_processed=preprocessor.transform(X_test)

#Gets only transformed features
    feature_names=preprocessor.get_feature_names_out()

    print("\n========== PREPROCESSING RESULTS ==========")

    print("\nOriginal training shape:", X_train.shape)
    print("Processed training shape:", X_train_processed.shape)

    print("\nOriginal validation shape:", X_val.shape)
    print("Processed validation shape:", X_val_processed.shape)

    print("\nOriginal testing shape:", X_test.shape)
    print("Processed testing shape:", X_test_processed.shape)

    print("\nTotal processed features:", len(feature_names))

    experiment_df.to_csv(
        PROCESSED_DATA_DIR / "experiment_dataset.csv",
        index=True,
        index_label="source_row_index"
    )

    print("\nExperiment dataset saved successfully!")


    save_split(
        "train",
        X_train,
        y_train,
        treatment_train
    )

    save_split(
        "validation",
        X_val,
        y_val,
        treatment_val
    )

    save_split(
        "test",
        X_test,
        y_test,
        treatment_test
    )


#Saving transformed features
    feature_names_df = pd.DataFrame({

        "feature_name": feature_names

    })

    feature_names_df.to_csv(
        PROCESSED_DATA_DIR / "feature_names.csv",
        index=False
    )

    preprocessor_path = (
        ARTIFACTS_DIR / "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    print("\nPreprocessor saved at:", preprocessor_path)

    print("\nAll processed datasets saved successfully!")


    return {

        # Original feature datasets
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        # Conversion outcomes
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        # Treatment assignments
        "treatment_train": treatment_train,
        "treatment_val": treatment_val,
        "treatment_test": treatment_test,

        # Transformed feature matrices
        "X_train_processed": X_train_processed,
        "X_val_processed": X_val_processed,
        "X_test_processed": X_test_processed,

        # Fitted preprocessing objects
        "preprocessor": preprocessor,

        "feature_names": feature_names

    }


if __name__=="__main__":
    processed_data=preprocess_data()

