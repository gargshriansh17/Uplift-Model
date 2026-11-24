import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COL,
    RANDOM_STATE
)

TRAIN_PATH=PROCESSED_DATA_DIR/"train_data.csv"
PREDICTION_FILES={
    "Logistic T-Learner":(
        REPORTS_DIR/"t_learner_tuned_validation_predictions.csv"
    ),

    "S-Learner":(
        REPORTS_DIR/"s_learner_tuned_validation_predictions.csv"
    ),

    "LightGBM T-Leaner":(
        REPORTS_DIR/"lightgbm_t_learner_validation_predictions.csv"
    )
}

OUTPUT_DIR=REPORTS_DIR/"model_comparison"

TARGET_PERCENTAGES=[0.10,0.20,0.30]

RANDOM_REPEATS=500

REQUIRED_COLS=[
    "source_row_index",
    "actual_treatment",
    "actual_conversion",
    "predicted_purchase_if_treated",
    "predicted_purchase_if_control",
    "predicted_uplift"
]


def load_prediction_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:{path}"
        )

    df=pd.read_csv(path)

    missing_cols=[
        col for col in REQUIRED_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns in {path.name}:{missing_cols}"
        )

    if df[REQUIRED_COLS].isna().any().any():
        raise ValueError(
            f"Missing values found in {path.name}"
        )

    customer_ids=pd.to_numeric(
        df["source_row_index"],
        errors="raise"
    )

    if not np.isfinite(customer_ids).all():
        raise ValueError(
            f"Invalid customer IDs in {path.name}"
        )

    if not np.equal(
        customer_ids,
        np.floor(customer_ids)
    ).all():
        raise ValueError(
            f"Customer IDs must be integers in {path.name}"
        )

    df["source_row_index"]=customer_ids.astype("int64")

    if df["source_row_index"].duplicated().any():
        raise ValueError(
            f"Duplicate customer IDs found in {path.name}"
        )

    for col in ["actual_treatment","actual_conversion"]:
        df[col]=pd.to_numeric(
            df[col],
            errors="raise"
        )

        if not df[col].isin([0,1]).all():
            raise ValueError(
                f"Invalid binary values in {col}:{path.name}"
            )

        df[col]=df[col].astype(int)

    probability_cols=[
        "predicted_purchase_if_treated",
        "predicted_purchase_if_control"
    ]

    for col in probability_cols:
        df[col]=pd.to_numeric(
            df[col],
            errors="raise"
        )

        if not np.isfinite(df[col]).all():
            raise ValueError(
                f"Invalid probabilities in {col}:{path.name}"
            )

        if not df[col].between(0,1).all():
            raise ValueError(
                f"Probabilities outside [0,1] in {path.name}"
            )

    calculated_uplift=(
        df["predicted_purchase_if_treated"]
        -
        df["predicted_purchase_if_control"]
    )

    if not np.allclose(
        calculated_uplift,
        df["predicted_uplift"],
        atol=1e-8
    ):
        raise ValueError(
            f"Predicted uplift does not match potential outcome probabilities in {path.name}"
        )

    df["predicted_uplift"]=calculated_uplift

    df=df.sort_values("source_row_index").reset_index(drop=True)

    return df


def load_all_predictions():
    predictions={}

    for model_name,path in PREDICTION_FILES.items():
        print(f"\nLoading {model_name}")

        predictions[model_name]=load_prediction_file(path)

        print(
            "Validation customers:",
            len(predictions[model_name])
        )

    reference=predictions["Logistic T-Learner"]

    for model_name,df in predictions.items():
        if not np.array_equal(
            reference["source_row_index"].to_numpy(),
            df["source_row_index"].to_numpy()
        ):

            raise ValueError(
                f"Customer IDs donot match for {model_name}"
            )

        for col in [
            "actual_treatment",
            "actual_conversion"
        ]:
            if not np.array_equal(
                reference[col].to_numpy(),
                df[col].to_numpy()
            ):
                raise ValueError(
                    f"{col} does not match for {model_name}"
                )

    print("\nVALIDATION CONSISTENCY CHECK PASSED")

    print(
        "All models use the same",
        len(reference),
        "validation customers."
    )

    return predictions

def calculate_training_ate():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found:{TRAIN_PATH}")

    train_df=pd.read_csv(TRAIN_PATH)

    treatment_rate=train_df.loc[
        train_df["treatment"]==1,
        TARGET_COL
    ].mean()

    control_rate=train_df.loc[
        train_df["treatment"]==0,
        TARGET_COL
    ].mean()

    ate_pp=(treatment_rate-control_rate)*100

    print(
        "Treatment conversion rate:",
        round(float(treatment_rate * 100), 4),
        "%"
    )

    print(
        "Control conversion rate:",
        round(float(control_rate * 100), 4),
        "%"
    )

    print(
        "Training ATE:",
        round(float(ate_pp), 4),
        "percentage points"
    )

    return ate_pp

def calculate_probability_metrics(validation_df,model_name):

    from sklearn.metrics import log_loss,brier_score_loss

    treatment_mask=(validation_df["actual_treatment"]==1)

    control_mask=(validation_df["actual_treatment"]==0)

    y=validation_df["actual_conversion"].to_numpy()

    treatment_prob=validation_df["predicted_purchase_if_treated"].to_numpy()

    control_prob=validation_df["predicted_purchase_if_control"].to_numpy()

    treatment_log_loss=log_loss(
        y[treatment_mask],
        treatment_prob[treatment_mask],
        labels=[0,1]
    )

    control_log_loss=log_loss(
        y[control_mask],
        control_prob[control_mask],
        labels=[0,1]
    )

    treatment_brier=brier_score_loss(
        y[treatment_mask],
        treatment_prob[treatment_mask]
    )

    control_brier=brier_score_loss(
        y[control_mask],
        control_prob[control_mask]
    )

    return {
        "model":model_name,
        "treatment_log_loss":treatment_log_loss,
        "control_log_loss":control_log_loss,
        "treatment_brier":treatment_brier,
        "control_brier":control_brier
    }

def calculate_group_uplift(selected_df):
    treatment_group=selected_df[
        selected_df["actual_treatment"]==1
    ]

    control_group=selected_df[
        selected_df["actual_treatment"]==0
    ]

    if len(treatment_group)==0 or len(control_group)==0:
        raise ValueError(
            "Both treatment and control customers are required in the selected group"
        )

    treatment_rate=(
        treatment_group["actual_conversion"].mean()
    )

    control_rate=(
        control_group["actual_conversion"].mean()
    )

    uplift_pp=(treatment_rate-control_rate)*100

    return {
        "selected_customers":len(selected_df),
        "treatment_customers":len(treatment_group),
        "control_customers":len(control_group),
        "treatment_conversions":int(treatment_group["actual_conversion"].sum()),
        "control_conversions":int(control_group["actual_conversion"].sum()),
        "treatment_conversion_rate_pct": treatment_rate * 100,
        "control_conversion_rate_pct": control_rate * 100,
        "observed_uplift_pp": uplift_pp
    }


def create_random_samples(total_customers):
    rng=np.random.default_rng(RANDOM_STATE)
    random_samples={}

    for percentage in TARGET_PERCENTAGES:
        k=int(np.ceil(total_customers*percentage))

        random_samples[percentage]=[
            rng.permutation(total_customers)[:k]
            for _ in range(RANDOM_REPEATS)
        ]

    return random_samples

def calculate_uplift_at_k(validation_df,model_name,random_samples):

    ranked_df=validation_df.sort_values(
        "predicted_uplift",
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    total_customers=len(ranked_df)
    results=[]

    for percentage in TARGET_PERCENTAGES:
        k=int(
            np.ceil(total_customers*percentage)
        )

        top_k=ranked_df.iloc[:k]

        model_result=calculate_group_uplift(top_k)

        random_uplifts=[]

        for indices in random_samples[percentage]:
            random_group=validation_df.iloc[indices]

            random_result=calculate_group_uplift(random_group)

            random_uplifts.append(random_result["observed_uplift_pp"])

        random_uplifts=np.asarray(random_uplifts)

        results.append({
            "model":model_name,
            "target_percentage":percentage*100,
            **model_result,
            "random_mean_uplift_pp":np.mean(random_uplifts),
            "random_std_uplift_pp":np.std(random_uplifts,ddof=1),
            "random_2.5_percentile":np.percentile(random_uplifts,2.5),
            "random_97.5_percentile":np.percentile(random_uplifts,97.5),
            "uplift_vs_random_pp":(
                model_result["observed_uplift_pp"]
                -
                np.mean(random_uplifts)
            )
        })

    return pd.DataFrame(results)

def calculate_qini_curve(validation_df):
    ranked_df=validation_df.sort_values(
        "predicted_uplift",
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    treatment=ranked_df["actual_treatment"].to_numpy()
    outcome=ranked_df["actual_conversion"].to_numpy()

    total_customers=len(ranked_df)

    total_treatment=int(np.sum(treatment==1))

    total_control=int(np.sum(treatment==0))

    if total_treatment==0 or total_control==0:
        raise ValueError(
            "Both treatment and control groups are required"
        )

    treatment_conversions=(
        (treatment==1) & (outcome==1)
    ).astype(int)

    control_conversions=(
        (treatment==0) & (outcome==1)
    ).astype(int)

    cumulative_treatment=np.cumsum(treatment_conversions)

    cumulative_control=np.cumsum(control_conversions)

    qini_gain=(
        cumulative_treatment/total_treatment
        -
        cumulative_control/total_control
    )

    qini_gain_pp=qini_gain*100

    target_fraction=(
        np.arange(1,total_customers+1)/total_customers
    )

    observed_ate_pp=(
        outcome[treatment==1].mean()
        -
        outcome[treatment==0].mean()
    )*100

    if not np.isclose(
        qini_gain_pp[-1],
        observed_ate_pp,
        atol=1e-10
    ):
        raise ValueError(
            "Qini endpoint does not match validation ATE"
        )

    random_baseline_pp=(target_fraction*observed_ate_pp)

    qini_df=pd.DataFrame({
        "customer_targeted":np.arange(1,total_customers+1),
        "target_fraction":target_fraction,
        "target_percentage":target_fraction*100,
        "cumulative_treatment_conversions":cumulative_treatment,
        "cumulative_control_conversions":cumulative_control,
        "qini_gain_pp":qini_gain_pp,
        "random_baseline_pp":random_baseline_pp
    })

    qini_df=pd.concat([
        pd.DataFrame([{
            "customers_targeted": 0,
            "target_fraction": 0.0,
            "target_percentage": 0.0,
            "cumulative_treatment_conversions": 0,
            "cumulative_control_conversions": 0,
            "qini_gain_pp": 0.0,
            "random_baseline_pp": 0.0
        }]),
        qini_df
    ],ignore_index=True)

    return qini_df,observed_ate_pp

def calculate_qini_auuc(qini_df):
    x=qini_df["target_fraction"].to_numpy()

    model_gain=qini_df["qini_gain_pp"].to_numpy()

    random_gain=qini_df["random_baseline_pp"].to_numpy()

    if hasattr(np,"trapezoid"):
        integrate=np.trapezoid

    else:
        from scipy.integrate import trapezoid
        integrate=trapezoid

    model_auuc=integrate(
        model_gain,
        x
    )

    random_auuc=integrate(
        random_gain,
        x
    )

    qini_coefficient=(model_auuc-random_auuc)

    return{
        "model_auuc": model_auuc,
        "random_auuc": random_auuc,
        "qini_coefficient": qini_coefficient
    }

def save_model_results(
        model_name,
        validation_df,
        uplift_df,
        qini_df,
        qini_summary
):

    folder_name=(
        model_name.lower().replace(" ","_").replace("-","_")
    )

    model_output_dir=(OUTPUT_DIR/folder_name)

    model_output_dir.mkdir(parents=True,exist_ok=True)

    uplift_df.to_csv(model_output_dir/"uplift_at_k.csv",index=False)

    qini_df.to_csv(model_output_dir/"qini_curve.csv",index=False)

    pd.DataFrame([{
        "model":model_name,
        **qini_summary
    }]).to_csv(model_output_dir/"qini_auuc_summary.csv",index=False)

    plt.figure(figsize=(10, 6))

    plt.plot(
        qini_df["target_percentage"],
        qini_df["qini_gain_pp"],
        label=model_name,
        linewidth=2
    )

    plt.plot(
        qini_df["target_percentage"],
        qini_df["random_baseline_pp"],
        linestyle="--",
        label="Random Targeting"
    )

    plt.axhline(
        y=0,
        linestyle=":",
        linewidth=1
    )

    plt.xlabel("Percentage of Customers Targeted")

    plt.ylabel("Cumulative Gain (percentage points)")

    plt.title(f"Qini Curve - {model_name}")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        model_output_dir / "qini_curve.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

def run_model_comparison():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    predictions=load_all_predictions()

    training_ate=calculate_training_ate()

    print("\nTraining constant ATE baseline:",round(float(training_ate),4),"percentage points")

    total_customers=len(next(iter(predictions.values())))

    random_samples=create_random_samples(total_customers)

    prob_results=[]
    uplift_results=[]
    qini_results=[]

    for model_name,validation_df in predictions.items():
        print("\n"+"="*55)
        print("EVALUATING:",model_name)
        print("="*55)

        probability_metrics=calculate_probability_metrics(
            validation_df,
            model_name
        )

        prob_results.append(probability_metrics)

        uplift_df=calculate_uplift_at_k(
            validation_df,
            model_name,
            random_samples
        )

        uplift_results.append(uplift_df)

        print("\nUPLIFT AT K")

        print(
            uplift_df.round(4).to_string(index=False)
        )

        qini_df,observed_ate=calculate_qini_curve(validation_df)

        print(
            "\nObserved validation ATE:",
            round(float(observed_ate), 4),
            "percentage points"
        )

        # Qini coefficient and AUUC
        qini_summary = calculate_qini_auuc(
            qini_df
        )

        qini_results.append({
            "model": model_name,
            **qini_summary
        })

        print("\nQINI AND AUUC")

        for metric, value in qini_summary.items():

            print(
                metric,
                ":",
                round(float(value), 6)
            )

        # Save individual results
        save_model_results(
            model_name,
            validation_df,
            uplift_df,
            qini_df,
            qini_summary
        )

    # --------------------------------------------------
    # BUILD COMMON COMPARISON TABLES
    # --------------------------------------------------

    probability_df = pd.DataFrame(
        prob_results
    )

    uplift_df = pd.concat(
        uplift_results,
        ignore_index=True
    )

    qini_df = pd.DataFrame(
        qini_results
    )

    # Uplift@K in separate columns
    uplift_wide = uplift_df.pivot(
        index="model",
        columns="target_percentage",
        values="observed_uplift_pp"
    )

    uplift_wide = uplift_wide.rename(
        columns={
            10.0: "uplift_at_10_pp",
            20.0: "uplift_at_20_pp",
            30.0: "uplift_at_30_pp"
        }
    ).reset_index()

    # Combine every metric into one report
    comparison_df = (
        probability_df
        .merge(qini_df, on="model")
        .merge(uplift_wide, on="model")
    )

    # Save master comparison files
    probability_df.to_csv(
        OUTPUT_DIR / "validation_probability_metrics.csv",
        index=False
    )

    uplift_df.to_csv(
        OUTPUT_DIR / "uplift_at_k_comparison.csv",
        index=False
    )

    qini_df.to_csv(
        OUTPUT_DIR / "qini_auuc_comparison.csv",
        index=False
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "model_comparison_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)

    print("FINAL MODEL COMPARISON")

    print("=" * 65)

    print(
        comparison_df.round(6).to_string(index=False)
    )

    print(
        "\nAll comparison reports saved at:",
        OUTPUT_DIR
    )

    return (
        probability_df,
        uplift_df,
        qini_df,
        comparison_df
    )


if __name__ == "__main__":

    run_model_comparison()


    
