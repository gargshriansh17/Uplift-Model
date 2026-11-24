import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.config import(
    DATA_PATH,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    SEGMENT_COL,
    TREATMENT_SEGMENT,
    CONTROL_SEGMENT,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE
)

from src.data_preprocessing import FEATURE_COLUMNS


WOMENS_SEGMENT="Womens E-Mail"
THREE_ARM_DIR=PROCESSED_DATA_DIR/"three_arm"

LEGACY_SPLIT_PATHS={
    "train":PROCESSED_DATA_DIR/"train_data.csv",
    "validation":PROCESSED_DATA_DIR/"validation_data.csv",
    "test":PROCESSED_DATA_DIR/"test_data.csv"
}

TREATMENT_MAPPING={
    CONTROL_SEGMENT:0,
    TREATMENT_SEGMENT:1,
    WOMENS_SEGMENT:2
}

def load_original_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Original dataset not found:{DATA_PATH}"
        )

    df=pd.read_csv(DATA_PATH)

    required_cols=(FEATURE_COLUMNS+[SEGMENT_COL,TARGET_COL])

    missing_cols=[
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing original dataset columns:{missing_cols}"
        )

    if df[[SEGMENT_COL,TARGET_COL]].isna().any().any():
        raise ValueError(
            "Missing treatment segment or conversion values"
        )

    if not df[TARGET_COL].isin([0,1]).all():
        raise ValueError(
            "Conversion must contain only 0 and 1"
        )

    expected_segments=set(TREATMENT_MAPPING.keys())
    actual_segments=set(df[SEGMENT_COL].unique())

    if actual_segments!=expected_segments:
        raise ValueError(
            "Unexpected campaign segments found.\n"
            f"Expected:{expected_segments}\n"
            f"Found:{actual_segments}"
        )

    df.index.name="source_row_index"

    if df.index.has_duplicates:
        raise ValueError(
            "Duplicate source row indices found."
        )

    print("\nORIGINAL DATASET")

    print("Total customers:",len(df))

    print("\nCampaign distribution:")
    print(df[SEGMENT_COL].value_counts())

    return df

def load_existing_split_assignments(raw_df):
    split_indices={}
    all_existing_indices=[]

    for split_name,file_path in LEGACY_SPLIT_PATHS.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Existing {split_name} file not found:{file_path}"
            )

        existing_df=pd.read_csv(file_path)

        required_cols=[
            "source_row_index",
            "treatment"
        ]

        missing_cols=[
            col for col in required_cols
            if col not in existing_df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing columns in {file_path.name}:{missing_cols}"
            )

        source_indices=pd.to_numeric(
            existing_df["source_row_index"],
            errors="raise"
        )

        if source_indices.isna().any():
            raise ValueError(
                f"Missing source indices in {file_path.name}"
            )

        if not np.equal(
            source_indices,
            np.floor(source_indices)
        ).all():
            raise ValueError(
                f"Non-Integer source indices in {file_path.name}"
            )

        source_indices=source_indices.astype(int).to_numpy()

        if len(np.unique(source_indices))!=len(source_indices):
            raise ValueError(
                f"Duplicate source indices in {file_path.name}"
            )

        if not np.isin(
            pd.to_numeric(
                existing_df["treatment"],
                errors="raise"
            ),
            [0,1]
        ).all():
            raise ValueError(
                f"Invalid legacy treatment assignments in {file_path.name}"
            )

        missing_indices=(set(source_indices)-set(raw_df.index))

        if missing_indices:
            raise ValueError(
                f"Source indices from {file_path.name} are absent from the original dataset"
            )

        original_segments=raw_df.loc[source_indices,SEGMENT_COL]

        if not original_segments.isin([
            CONTROL_SEGMENT,
            TREATMENT_SEGMENT
        ]).all():
            raise ValueError(
                f"Unexpected campaign group in {file_path.name}"
            )

        expected_treatment=(original_segments==TREATMENT_SEGMENT).astype(int).to_numpy()

        actual_treatment=pd.to_numeric(existing_df["treatment"],errors="raise").astype(int).to_numpy()

        if not np.array_equal(
            expected_treatment,
            actual_treatment
        ):
            raise ValueError(
                f"Treatment assignment mismatch in {file_path.name}"
            )

        split_indices[split_name]=source_indices
        all_existing_indices.extend(source_indices)

        print(f"\nExisting {split_name} customers:",len(source_indices))

    if len(all_existing_indices)!=len(set(all_existing_indices)):
        raise ValueError(
            "A customer appears in multiple existing splits"
        )

    expected_legacy_indices=set(
        raw_df.index[
            raw_df[SEGMENT_COL].isin([
                CONTROL_SEGMENT,
                TREATMENT_SEGMENT
            ])
        ]
    )

    actual_legacy_indices=set(all_existing_indices)

    missing=expected_legacy_indices-actual_legacy_indices
    extra=actual_legacy_indices-expected_legacy_indices

    if missing or extra:
        raise ValueError(
            "Existing splits do not cover the original "
            "two-treatment population correctly.\n"
            f"Missing IDs: {len(missing)}\n"
            f"Unexpected IDs: {len(extra)}"
        )

    print("\nExisting Men's Email and No Email assignments verified successfully.")

    return split_indices

def split_womens_customers(raw_df):
    womens_df=raw_df[
        raw_df[SEGMENT_COL]==WOMENS_SEGMENT
    ].copy()

    outcome_counts=womens_df[TARGET_COL].value_counts()
    if len(outcome_counts)!=2 or outcome_counts.min()<2:
        raise ValueError(
            "Insufficient Women's Email conversion classes for stratified splitting"
        )

    print("\nWOMEN'S EMAIL DATASET")
    print("Total customers:", len(womens_df))

    print("\nConversion distribution:")
    print(outcome_counts.sort_index())

    womens_train_val,womens_test=train_test_split(
        womens_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=womens_df[TARGET_COL]
    )

    val_fraction=(VALIDATION_SIZE/(1-TEST_SIZE))

    womens_train,womens_val=train_test_split(
        womens_train_val,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=womens_train_val[TARGET_COL]
    )

    womens_splits={
        "train":womens_train,
        "validation":womens_val,
        "test":womens_test
    }

    print("\nWomen's Email split sizes:")

    for split_name, split_df in womens_splits.items():

        print(
            f"{split_name}: {len(split_df)} customers "
            f"({len(split_df) / len(womens_df) * 100:.2f}%)"
        )

    return womens_splits

def build_three_arm_partitions(
        raw_df,
        existing_indices,
        womens_splits
):

    partitions={}

    for split_name in ["train","validation","test"]:
        old_part=raw_df.loc[
            existing_indices[split_name]
        ].copy()

        old_part["treatment"]=(
            old_part[SEGMENT_COL].map(TREATMENT_MAPPING)
        )

        womens_part=womens_splits[split_name].copy()

        womens_part["treatment"]=(TREATMENT_MAPPING[WOMENS_SEGMENT])

        combined=pd.concat(
            [old_part,womens_part],
            axis=0
        ).sort_index()

        if combined.index.has_duplicates:
            raise ValueError(
                f"Duplicate customers found in {split_name}"
            )

        export_df=combined[
            FEATURE_COLUMNS+[
                TARGET_COL,
                "treatment"
            ]
        ].copy()

        export_df.index.name="source_row_index"

        partitions[split_name]=export_df

    return partitions

def validate_partitions(partitions,raw_df,existing_indices):
    all_indices=[]

    expected_customers=set(raw_df.index)

    for split_name,split_df in partitions.items():
        if split_df.empty:
            raise ValueError(
                f"{split_name} dataset is empty"
            )

        if split_df.index.has_duplicates:
            raise ValueError(
                f"Duplicate source indices in {split_name}"
            )

        if (
            split_df["treatment"].isna().any()
            or split_df[TARGET_COL].isna().any()
        ):
            raise ValueError(
                f"Missing treatment or conversion values in {split_name}"
            )

        if not split_df["treatment"].isin([0,1,2]).all():
            raise ValueError(
                f"Invalid treatment values in {split_name}"
            )

        if not split_df[TARGET_COL].isin([0, 1]).all():
            raise ValueError(
                f"Invalid conversion values in {split_name}."
            )


        distribution=pd.crosstab(
            split_df["treatment"],
            split_df[TARGET_COL]
        ).reindex(
            index=[0,1,2],
            columns=[0,1],
            fill_value=0
        )

        if (distribution==0).any().any():
            raise ValueError(
                f"Missing treatment-outcome combination "
                f"in {split_name}:\n{distribution}"
            )

        all_indices.extend(split_df.index.tolist())

        orignal_ids=set(existing_indices[split_name])

        new_legacy_ids=set(
            split_df.index[
                split_df["treatment"].isin([0,1])
            ]
        )

        if orignal_ids!=new_legacy_ids:
            raise ValueError(
                f"Legacy customer assignments changed in {split_name}"
            )

        print(f"\n{split_name.upper()} DISTRIBUTION")

        print(
            distribution.rename_axis(
                index="Treatment",
                columns="Conversion"
            )
        )

        print("Total customers:", len(split_df))

    if len(all_indices)!=len(set(all_indices)):
        raise ValueError(
            "Customer overlap found between partitions"
        )

    actual_customers=set(all_indices)

    missing=expected_customers-actual_customers
    extra=actual_customers-expected_customers

    if missing or extra:
        raise ValueError(
            "Three-arm partitions do not cover the "
            "original dataset correctly.\n"
            f"Missing customers: {len(missing)}\n"
            f"Unexpected customers: {len(extra)}"
        )

def save_three_arm_partitions(partitions):
    THREE_ARM_DIR.mkdir(parents=True,exist_ok=True)

    for split_name,split_df in partitions.items():
        output_path=(THREE_ARM_DIR/f"{split_name}_data.csv")

        split_df.to_csv(output_path,index=True,index_label="source_row_index")

        print(f"\n{split_name} dataset saved at: {output_path}")


def prepare_three_arm_data():
    raw_df=load_original_data()
    existing_indices=load_existing_split_assignments(raw_df)
    womens_splits=split_womens_customers(raw_df)
    partitions=build_three_arm_partitions(
        raw_df,
        existing_indices,
        womens_splits
    )
    validate_partitions(
        partitions,
        raw_df,
        existing_indices
    )

    save_three_arm_partitions(partitions)

    print("\nOutput directory:", THREE_ARM_DIR)

    return partitions


if __name__ == "__main__":

    prepare_three_arm_data()
