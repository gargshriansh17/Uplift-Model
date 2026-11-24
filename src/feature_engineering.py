import numpy as np
import pandas as pd

from src.config import(
    PROCESSED_DATA_DIR,
    TARGET_COL
)

ENGINEERED_DIR=(PROCESSED_DATA_DIR/"feature_experiments")
SPLIT_FILES={
    "train":PROCESSED_DATA_DIR/"train_data.csv",
    "validation":PROCESSED_DATA_DIR/"validation_data.csv",
    "test":PROCESSED_DATA_DIR/"test_data.csv"
}

BASE_FEATURES=[
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "history_segment",
    "zip_code",
    "channel"
]

REQUIRED_COLUMNS=(
    BASE_FEATURES+[TARGET_COL,"treatment"]
)

def validate_data(df,split_name):
    missing_columns=[
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {split_name}:{missing_columns}"
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            f"Missing values found in {split_name}"
        )

    if (df["history"]<0).any():
        raise ValueError(
            "History cannot be negative for log transformation"
        )

    for col in ["mens","womens"]:
        if not df[col].isin([0,1]).all():
            raise ValueError(
                f"{col} must contain only 0 and 1"
            )

def create_feature_variants(df):
    baseline_df=df.copy()#Original Features
    log_history_df=df.copy()#EXP A-Log transformation of history

    log_history_df["log_history"]=np.log1p(log_history_df["history"])

    log_history_df=log_history_df.drop(columns=["history"])

    interaction_df=log_history_df.copy()

    interaction_df["recency_x_mens"]=(
        interaction_df["recency"]
        *
        interaction_df["mens"]
    )

    interaction_df["recency_x_womens"]=(
        interaction_df["recency"]
        *
        interaction_df["womens"]
    )

    interaction_df["log_history_x_mens"]=(
        interaction_df["log_history"]
        *
        interaction_df["mens"]
    )

    interaction_df["log_history_x_womens"]=(
        interaction_df["log_history"]
        *
        interaction_df["womens"]
    )

    #Removing history_segments
    ablation_df=df.drop(columns=["history_segment"]).copy()

    variants={
        "baseline":baseline_df,
        "log_history":log_history_df,
        "interactions":interaction_df,
        "no_history_segment":ablation_df
    }

    return variants

def run_feature_engineering():
    ENGINEERED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for split_name,file_path in SPLIT_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:{file_path}"
            )

        df=pd.read_csv(file_path)

        validate_data(df,split_name)

        variants=create_feature_variants(df)

        for variant_name,variant_df in variants.items():
            output_dir=ENGINEERED_DIR/variant_name
            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path=(output_dir/f"{split_name}_data.csv")

            variant_df.to_csv(output_path,index=False)

            feature_count=len([
                col for col in variant_df.columns
                if col not in[
                    "source_row_index",
                    TARGET_COL,
                    "treatment"
                ]
            ])

            print(
                f"{variant_name}: "
                f"{feature_count} features | "
                f"{len(variant_df)} customers"
            )

            print("Saved:", output_path)

    print(
        "Engineered datasets saved at:",
        ENGINEERED_DIR
    )


if __name__ == "__main__":

    run_feature_engineering()
