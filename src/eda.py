import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import(
    DATA_PATH,
    REPORTS_DIR,
    FIGURES_DIR,
    SEGMENT_COL,
    TARGET_COL,
    TREATMENT_SEGMENT,
    CONTROL_SEGMENT
)

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df=pd.read_csv(DATA_PATH)

    required_columns=[SEGMENT_COL,"visit",TARGET_COL,"spend"]

    missing_columns=[col for col in required_columns
                     if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns:{missing_columns}")

    print("\nData Set Loaded Successfully")

    return df

def inspect_dataset(df):

    print("DATASET OVERVIEW")

    print("\nShape",df.shape)
    print("\nColumn Names:",df.columns.tolist())
    print("\nFirst Five Rows:",df.head())
    print("\nDataset Information:",df.info())
    print("\nStatistical Summary:",df.describe(include="all").T)

df=load_dataset()
# inspect_dataset(df)

def data_quality(df):

    print("\nDATA QUALITY")

    missing_values=df.isnull().sum()
    print("\nMissing Values:",missing_values)
    print("\nTotal missing values:",missing_values.sum())

    duplicates=df.duplicated().sum()
    print("\nDuplicate Records:",duplicates)
    print("\nUnique Values in Each Column:",df.nunique())

    quality_report=pd.DataFrame({
        "data_type":df.dtypes.astype(str),
        "missing_values":missing_values,
        "missing_percentages":(df.isnull().mean()*100),
        "unique_values":df.nunique()
    })

    quality_report.to_csv(REPORTS_DIR/"data_quality_report.csv")
    print("\nData quality report saved successfully")

def analyze_customer_distribution(df):
    print("\nCUSTOMER DISTRIBUTION")

    distribution=df[SEGMENT_COL].value_counts()
    print("\nCampaign Distribution:",distribution)

    percentages=(df[SEGMENT_COL].value_counts(normalize=True)*100)
    print("\nCustomer Distribution %:",percentages.round(2))

    plt.figure(figsize=(9,5))
    plt.bar(distribution.index,distribution.values)
    plt.title("Customer Distribution by Campaign")
    plt.xlabel("Campaign")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "customer_distribution.png",
        dpi=150
    )
    plt.show()
    plt.close()
    print("\nCustomer distribution chart saved!")


def analyze_campaign_performance(df):
    print("\nCAMPAIGN PERFORMANCE")

    summary=df.groupby(SEGMENT_COL).agg(
        total_customers=(SEGMENT_COL,"size"),
        total_visits=("visit","sum"),
        visit_rate=("visit","mean"),
        total_conversions=(TARGET_COL,"sum"),
        conversion_rate=(TARGET_COL,"mean"),
        total_spend=("spend","sum"),
        average_spend_per_customer=("spend","mean")
    )

    summary["visit_rate"]*=100
    summary["conversion_rate"]*=100

    print("\nCampaign Performance Summary:")
    print(summary.round(4).to_string())

    summary.to_csv(REPORTS_DIR/"campaign_summary.csv")

    plt.figure(figsize=(9,5))
    plt.bar(
        summary.index,
        summary["conversion_rate"]
    )
    plt.title("Conversion Rate by Campaign")
    plt.xlabel("Campaign")
    plt.ylabel("Conversion Rate (%)")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "conversion_by_campaign.png",
        dpi=150
    )
    plt.show()
    plt.close()

    # Visit rate visualization
    plt.figure(figsize=(9, 5))
    plt.bar(
        summary.index,
        summary["visit_rate"]
    )
    plt.title("Visit Rate by Campaign")
    plt.xlabel("Campaign")
    plt.ylabel("Visit Rate (%)")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "visit_rate_by_campaign.png",
        dpi=150
    )
    plt.show()
    plt.close()

    # Average spending visualization
    plt.figure(figsize=(9, 5))
    plt.bar(
        summary.index,
        summary["average_spend_per_customer"]
    )
    plt.title("Average Spend per Customer")
    plt.xlabel("Campaign")
    plt.ylabel("Average Spend")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "average_spend_by_campaign.png",
        dpi=150
    )
    plt.show()
    plt.close()
    print("\nCampaign performance reports saved!")
    return summary

def check_treatment_control_balance(df):
    print("\nTREATMENT-CONTROL BALANCE")

    experiment_df=df[
        df[SEGMENT_COL].isin([
            TREATMENT_SEGMENT,
            CONTROL_SEGMENT
        ])
    ].copy()

    treatment_group=experiment_df[experiment_df[SEGMENT_COL]==TREATMENT_SEGMENT]
    control_group=experiment_df[experiment_df[SEGMENT_COL]==CONTROL_SEGMENT]

    if treatment_group.empty or control_group.empty:
        raise ValueError(
            "Treatment or control group is missing."
        )

    print("\nTreatment Customers:",len(treatment_group))
    print("\nControl Customers:",len(control_group))

    numeric_features=[
        "recency",
        "history",
        "mens",
        "womens",
        "newbie"
    ]

    numeric_features=[
        col for col in numeric_features
        if col in experiment_df.columns
    ]

    balance_results=[]

    for feature in numeric_features:
        treatment_mean=treatment_group[feature].mean()
        control_mean=control_group[feature].mean()

        treatment_std=treatment_group[feature].std()
        control_std=control_group[feature].std()

        pooled_std=np.sqrt((treatment_std**2+control_std**2)/2)

        mean_difference=treatment_mean-control_mean

        if pooled_std>0:
            smd=mean_difference/pooled_std
        elif mean_difference==0:
            smd=0
        else:
            smd=np.nan

        balance_results.append({
            "feature":feature,
            "treatment_mean":treatment_mean,
            "control_mean":control_mean,
            "mean_difference":mean_difference,
            "standardized_mean_difference":smd
        })

    balance_table=pd.DataFrame(balance_results)
    print("\nNumerical Feature Balance:",balance_table.round(4))

    balance_table.to_csv(REPORTS_DIR/"treatment_control_balance.csv",
                         index=False)


    #Categorical feature comparison
    categorical_features=["history_segment","zip_code","channel"]

    for feature in categorical_features:
        if feature not in experiment_df.columns:
            continue

        proportions=pd.crosstab(
            experiment_df[feature],
            experiment_df[SEGMENT_COL],
            normalize="columns"
        )

        proportions=proportions.reindex(
            columns=[TREATMENT_SEGMENT,CONTROL_SEGMENT],
            fill_value=0
        )

        print(f"\n{feature} Distribution:",proportions.round(3))

        proportions.to_csv(REPORTS_DIR/f"{feature}_balance.csv")

    print("\nTreatment-control balance report saved")

def run_eda():
    df=load_dataset()
    inspect_dataset(df)
    data_quality(df)
    analyze_customer_distribution(df)
    analyze_campaign_performance(df)
    check_treatment_control_balance(df)

    print("\nEDA COMPLETED SUCCESSFULLY")

if __name__=="__main__":
    run_eda()