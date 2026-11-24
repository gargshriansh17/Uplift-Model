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
VALIDATION_PREDICTIONS_PATH=(REPORTS_DIR/"t_learner_validation_predictions.csv")
UPLIFT_K_PATH=REPORTS_DIR/"uplift_at_k.csv"
CONSTANT_ATE_PATH=REPORTS_DIR/"constant_ate_baseline.csv"

QINI_CURVE_PATH=REPORTS_DIR/"qini_curve.csv"
QINI_PLOT_PATH=REPORTS_DIR/"qini_curve.png"
QINI_COEFFICIENT_PATH=REPORTS_DIR/"qini_coefficient.csv"

AUUC_PATH=REPORTS_DIR/"auuc_results.csv"

TARGET_PERCENTAGES=[0.10,0.20,0.30]
RANDOM_REPEATS=500

def load_evaluation_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:{TRAIN_PATH}"
        )

    if not VALIDATION_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Validation predictions not found:{VALIDATION_PREDICTIONS_PATH}"
        )

    train_df=pd.read_csv(TRAIN_PATH)

    validation_df=pd.read_csv(VALIDATION_PREDICTIONS_PATH)

    required_train_columns=[
        TARGET_COL,
        "treatment"
    ]

    required_validation_columns=[
        "actual_treatment",
        "actual_conversion",
        "predicted_uplift"
    ]

    for col in required_train_columns:
        if col not in train_df.columns:
            raise ValueError(
                f"Missing column in training data:{col}"
            )

    for col in required_validation_columns:
        if col not in validation_df.columns:
            raise ValueError(
                f"Missing column in validation data:{col}"
            )

    if train_df[required_train_columns].isna().any().any():
        raise ValueError("Missing values found in training outcomes")

    if validation_df[required_validation_columns].isna().any().any():
        raise ValueError("Missing values found in validation data.")

    return train_df,validation_df

def calculate_constant_ate(train_df):
    treatment=train_df["treatment"]
    outcome=train_df[TARGET_COL]

    treatment_rate=outcome[treatment==1].mean()
    control_rate=outcome[treatment==0].mean()

    constant_ate=treatment_rate-control_rate

    constant_ate_pp=constant_ate*100

    print("\nConstant ATE Baseline")

    print(f"Treatment conversion rate:{treatment_rate*100:.4f}%")
    print(f"Control conversion rate:{control_rate*100:.4f}%")
    print(f"Constant predicted uplift:{constant_ate_pp:.4f}percentage points")

    baseline_df=pd.DataFrame([{
        "baseline":"Constant ATE",
        "training_treatment_rate":treatment_rate,
        "training_control_rate":control_rate,
        "constant_predicted_uplift_pp":constant_ate_pp
    }])

    baseline_df.to_csv(CONSTANT_ATE_PATH,index=False)

    return constant_ate

def calculate_group_uplift(selected_df):
    treatment_mask=(selected_df["actual_treatment"]==1)

    control_mask=(selected_df["actual_treatment"]==0)

    treatment_group=selected_df[treatment_mask]

    control_group=selected_df[control_mask]

    if len(treatment_group)==0 or len(control_group)==0:
        raise ValueError("Both treatment and control customers are required")

    treatment_rate=(treatment_group["actual_conversion"].mean())

    control_rate=(control_group["actual_conversion"].mean())

    uplift=treatment_rate-control_rate

    return {
        "customers":len(selected_df),
        "treatment_customers":len(treatment_group),
        "control_customers":len(control_group),
        "treatment_conversion":int(treatment_group["actual_conversion"].sum()),
        "control_conversion":int(control_group["actual_conversion"].sum()),
        "treatment_conversion_rate_pct":(treatment_rate*100),
        "control_conversion_rate_pct":(control_rate*100),
        "uplift_pp":uplift*100
    }

def calculate_uplift_at_k(validation_df,random_repeats=RANDOM_REPEATS):
    validation_df=validation_df.sort_values(
        by="predicted_uplift",
        ascending=False
    ).reset_index(drop=True)

    total_customers=len(validation_df)

    rng=np.random.default_rng(RANDOM_STATE)

    results=[]

    for percentage in TARGET_PERCENTAGES:
        k=int(np.ceil(total_customers*percentage))

        top_k=validation_df.iloc[:k]#T learners top K customers

        model_result=calculate_group_uplift(top_k)

        random_uplifts=[]

        for _ in range(random_repeats):
            random_indices=rng.permutation(total_customers)[:k]
            random_group=validation_df.iloc[random_indices]
            random_result=calculate_group_uplift(random_group)
            random_uplifts.append(random_result["uplift_pp"])

        random_uplifts=np.asarray(random_uplifts)

        results.append({
            "target_percentage":percentage*100,
            "selected_customers":k,
            "treatment_customers":(model_result["treatment_customers"]),
            "control_customers":(model_result["control_customers"]),
            "treatment_conversions":(model_result["treatment_conversion"]),
            "control_conversions":(model_result["control_conversion"]),
            "treatment_conversion_rate_pct":(model_result["treatment_conversion_rate_pct"]),
            "control_conversion_rate_pct":(model_result["control_conversion_rate_pct"]),
            "t_learner_uplift_pp":(model_result["uplift_pp"]),
            "random_mean_uplift_pp":(np.mean(random_uplifts)),
            "random_std_uplift_pp":(np.std(random_uplifts,ddof=1)),
            "random_2.5_percentile":(np.percentile(random_uplifts,2.5)),
            "random_97.5_percentile":(np.percentile(random_uplifts,97.5))
        })

    results_df=pd.DataFrame(results)
    results_df.to_csv(UPLIFT_K_PATH,index=False)

    print("\nUPLIFT AT K RESULTS")
    print(results_df.round(4).to_string(index=False))

    return results_df

def calculate_qini_curve(validation_df):
    required_columns=[
        "actual_treatment",
        "actual_conversion",
        "predicted_uplift"
    ]

    for col in required_columns:
        if col not in validation_df.columns:
            raise ValueError(
                f"Missing column for Qini curve:{col}"
            )

    if validation_df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in QINI evaluation data."
        )

    validation_df=validation_df.copy()

    validation_df["predicted_uplift"]=pd.to_numeric(
        validation_df["predicted_uplift"],
        errors="raise"
    )

    if not np.isfinite(
        validation_df["predicted_uplift"]
    ).all():
        raise ValueError(
            "Predicted uplift contains invalid numerical values"
        )

    if not validation_df["actual_treatment"].isin([0,1]).all():
        raise ValueError(
            "Treatment must contain only 0 and 1"
        )

    if not validation_df["actual_conversion"].isin([0,1]).all():
        raise ValueError(
            "Conversion must contain only 0 and 1"
        )

    sorted_df=validation_df.sort_values(
        by="predicted_uplift",
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    treatment=sorted_df["actual_treatment"].to_numpy()
    outcome=sorted_df["actual_conversion"].to_numpy()

    total_customers=len(sorted_df)

    total_treatment=int(np.sum(treatment==1))
    total_control=int(np.sum(treatment==0))

    if total_treatment==0 or total_control==0:
        raise ValueError(
            "Both treatment and control groups are required"
        )

    treatment_conversions=(
        (treatment==1)&(outcome==1)
    ).astype(int)

    control_conversions=(
        (treatment==0)&(outcome==1)
    ).astype(int)

    cumulative_treatment_conversions=np.cumsum(treatment_conversions)
    cumulative_control_conversions=np.cumsum(control_conversions)

    #Calculate qini gain at every ranking position
    qini_gain=(cumulative_treatment_conversions/total_treatment
               -
               cumulative_control_conversions/total_control)

    qini_gain_pp=qini_gain*100

    customers_targeted=np.arange(1,total_customers+1)
    target_fraction=(customers_targeted/total_customers)

    target_percentage=target_fraction*100

    overall_ate_pp=(
        outcome[treatment==1].mean()
        -
        outcome[treatment==0].mean()
    )*100

    # Verify that the final Qini point equals overall ATE

    if not np.isclose(
        qini_gain_pp[-1],
        overall_ate_pp,
        atol=1e-10
    ):
        raise ValueError(
            f"Qini endpoint mismatch: "
            f"{qini_gain_pp[-1]:.6f} vs "
            f"{overall_ate_pp:.6f}"
        )

    print(
        "Qini endpoint verified:",
        round(float(qini_gain_pp[-1]), 4),
        "percentage points"
    )

    random_baseline_pp=(target_fraction*overall_ate_pp)

    customers_targeted=np.insert(customers_targeted,0,0)

    target_fraction=np.insert(target_fraction,0,0)

    target_percentage=np.insert(target_percentage,0,0)

    qini_gain_pp=np.insert(qini_gain_pp,0,0)

    random_baseline_pp=np.insert(random_baseline_pp,0,0)

    cumulative_treatment_conversions=np.insert(cumulative_treatment_conversions,0,0)

    cumulative_control_conversions=np.insert(cumulative_control_conversions,0,0)

    qini_df=pd.DataFrame({
        "customers_targeted":customers_targeted,
        "target_fraction":target_fraction,
        "target_percentage":target_percentage,
        "cumulative_treatment_conversions":cumulative_treatment_conversions,
        "cumulative_control_conversions":cumulative_control_conversions,
        "qini_gain_pp":qini_gain_pp,
        "random_baseline_pp":random_baseline_pp
    })

    qini_df.to_csv(QINI_CURVE_PATH,index=False)

    plt.figure(figsize=(10,6))

    plt.plot(
        qini_df["target_percentage"],
        qini_df["qini_gain_pp"],
        label="T-LEARNER",
        linewidth=2
    )

    plt.plot(
        qini_df["target_percentage"],
        qini_df["random_baseline_pp"],
        label="Random Targeting",
        linestyle='--',
        linewidth=2
    )

    plt.axhline(
        y=0,
        linestyle=":",
        linewidth=1
    )

    plt.xlabel("Percentage of Customers Targeted")

    plt.ylabel("Cumulative Qini Gain (percentage points)")

    plt.title("Qini Curve - T-Learner vs Random Targeting")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        QINI_PLOT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # 10. Print summary.

    print("\nQINI CURVE SUMMARY")

    print(
        "Total validation customers:",
        total_customers
    )

    print(
        "Treatment customers:",
        total_treatment
    )

    print(
        "Control customers:",
        total_control
    )

    print(
        "Overall observed validation ATE:",
        round(overall_ate_pp, 4),
        "percentage points"
    )

    print(
        "\nQini Curve data saved at:",
        QINI_CURVE_PATH
    )

    print(
        "Qini Curve graph saved at:",
        QINI_PLOT_PATH
    )

    return qini_df


def calculate_qini_coefficient(qini_df):
    required_columns=[
        "target_fraction",
        "qini_gain_pp",
        "random_baseline_pp"
    ]

    for col in required_columns:
        if col not in qini_df.columns:
            raise ValueError(
                f"Missing column for qini coefficient:{col}"
            )

    if qini_df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in Qini Curve data."
        )

    target_fraction=qini_df["target_fraction"].to_numpy(dtype=float)

    model_gain=qini_df["qini_gain_pp"].to_numpy(dtype=float)

    random_gain=qini_df["random_baseline_pp"].to_numpy(dtype=float)

    if not (
        np.isfinite(target_fraction).all()
        and np.isfinite(model_gain).all()
        and np.isfinite(random_gain).all()
    ):
        raise ValueError(
            "Non-finite values found in qini curve."
        )

    if not np.all(np.diff(target_fraction)>=0):
        raise ValueError(
            "Target fractions must be in ascending order"
        )

    if not (
        np.isclose(target_fraction[0],0)
        and np.isclose(target_fraction[-1],1)
    ):
        raise ValueError(
            "Qini Curve must cover the range from 0 to 1"
        )

    if hasattr(np,"trapezoid"):
        trapezoid=np.trapezoid

    else:
        from scipy.integrate import trapezoid

    model_area=trapezoid(
        model_gain,
        target_fraction
    )

    random_area=trapezoid(
        random_gain,
        target_fraction
    )

    qini_coefficient=(
        model_area-random_area
    )

    results_df=pd.DataFrame([{
        "model_qini_area_pp":model_area,
        "random_qini_area_pp":random_area,
        "qini_coefficient_pp":qini_coefficient
    }])

    results_df.to_csv(QINI_COEFFICIENT_PATH,index=False)

    print("\nQini Coefficient Results")
    print("T-Learner Qini Area:",round(float(model_area),6))
    print("Random Targeting Qini Area:",round(float(random_area),6))
    print("Qini Coefficient:",round(float(qini_coefficient),6))

    if qini_coefficient>0:
        print("\nInterpretation T-Learner performs better than random targeting overall")

    elif qini_coefficient<0:
        print("\nInterpretation:T-Learner performs worse than random targeting overall")

    else:
        print("\nInterpretation:T-Learner performs approximately equal to random targeting")

    print("\nQini Coefficent saved at:",QINI_COEFFICIENT_PATH)

    return results_df

def calculate_auuc(qini_df):
    required_columns=[
        "target_fraction",
        "qini_gain_pp",
        "random_baseline_pp"
    ]

    for col in required_columns:
        if col not in qini_df.columns:
            raise ValueError(
                f"Missing column for AUUC:{col}"
            )

    if qini_df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in AUUC data"
        )

    target_function=qini_df["target_fraction"].to_numpy(dtype=float)

    model_gain=qini_df["qini_gain_pp"].to_numpy(dtype=float)

    random_gain=qini_df["random_baseline_pp"].to_numpy(dtype=float)

    if not(
        np.isfinite(target_function).all()
        and np.isfinite(model_gain).all()
        and np.isfinite(random_gain).all()
    ):
        raise ValueError(
            "Invalid numerical values found in AUUC data"
        )

    if not np.all(np.diff(target_function)>0):
        raise ValueError(
            "Target fractions must be strictly increasing"
        )

    if not(
        np.isclose(target_function[0],0)
        and np.isclose(target_function[-1],1)
    ):
        raise ValueError(
            "AUUC curve must cover the range from 0 to 1"
        )

    if hasattr(np,"trapezoid"):
        trapezoid=np.trapezoid

    else:
        from scipy.integrate import trapezoid

    auuc_model=trapezoid(model_gain,target_function)

    auuc_random=trapezoid(random_gain,target_function)

    qini_difference=auuc_model-auuc_random

    auuc_df=pd.DataFrame([{
        "t_learner_auuc":auuc_model,
        "random_targeting_auuc":auuc_random,
        "auuc_difference":qini_difference
    }])

    auuc_df.to_csv(AUUC_PATH,index=False)

    print("\nAUUC RESULTS")

    print(
        "T-Learner AUUC:",
        round(float(auuc_model), 6)
    )

    print(
        "Random Targeting AUUC:",
        round(float(auuc_random), 6)
    )

    print(
        "AUUC Difference:",
        round(float(qini_difference), 6)
    )

    print(
        "\nAUUC saved at:",
        AUUC_PATH
    )

    return auuc_df


def run_evaluation():

    train_df, validation_df = load_evaluation_data()

    print("Training shape:", train_df.shape)

    print("Validation shape:", validation_df.shape)

    # Constant ATE baseline

    constant_ate = calculate_constant_ate(
        train_df
    )

    # Uplift at different targeting percentages

    uplift_results = calculate_uplift_at_k(
        validation_df
    )

    qini_curve_df=calculate_qini_curve(
        validation_df
    )

    qini_coefficent_df=calculate_qini_coefficient(
        qini_curve_df
    )

    auuc_df=calculate_auuc(
        qini_curve_df
    )

    print(
        "Constant ATE baseline:",
        round(constant_ate * 100, 4),
        "percentage points"
    )

    return constant_ate, uplift_results,qini_curve_df,qini_coefficent_df,auuc_df


if __name__ == "__main__":

    run_evaluation()
