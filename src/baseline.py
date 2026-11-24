import numpy as np
import pandas as pd
from scipy.stats import norm,t as student_t
from src.config import(
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SEGMENT_COL,
    TARGET_COL,
    TREATMENT_SEGMENT,
    CONTROL_SEGMENT,
)


EXPERIMENT_DATA_PATH=(PROCESSED_DATA_DIR/"experiment_dataset.csv")

CONFIDENCE_INTERVAL=0.95
SIGNIFICANCE_INTERVAL=0.05

def load_experiment_data():
    if not EXPERIMENT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Experiment dataset not found:{EXPERIMENT_DATA_PATH}"
        )

    df=pd.read_csv(EXPERIMENT_DATA_PATH)

    required_cols=[SEGMENT_COL,
                   "treatment",
                   TARGET_COL,
                   "visit",
                   "spend"
                ]

    missing_cols=[
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns:{missing_cols}"
        )

    if df["treatment"].isna().any():
        raise ValueError("Missing treatment assignments found.")

    if not set(df["treatment"].unique()).issubset({0,1}):
        raise ValueError(
            "Treatment must contain only 0 and 1"
        )

    expected_treatment=(
        df[SEGMENT_COL]==TREATMENT_SEGMENT
    ).astype(int)

    if not (expected_treatment==df["treatment"]).all():
        raise ValueError(
            "Treatment assignment does not match campaign segment."
        )

    expected_segments={TREATMENT_SEGMENT,CONTROL_SEGMENT}

    if set(df[SEGMENT_COL].unique()) != expected_segments:
        raise ValueError(
            "Unexpected campaign groups found in experiment dataset"
        )

    print("\nDataset shape:",df.shape)

    return df

#Treatment and Control groups

def get_groups(df,outcome):
    if outcome not in df.columns:
        raise ValueError(
            f"Outcome column '{outcome}' not found"
        )

    treatment_group=df.loc[
        df["treatment"]==1,outcome
    ]

    control_group=df.loc[
        df["treatment"]==0,outcome
    ]

    #Convert outcomes into numerical values
    treatment_group=pd.to_numeric(treatment_group,errors="coerce")
    control_group=pd.to_numeric(control_group,errors="coerce")

    if treatment_group.empty or control_group.empty:
        raise ValueError(
            "Treatment or control group is empty"
        )

    if treatment_group.isna().any() or control_group.isna().any():
        raise ValueError(
            f"Missing or invalid values found in {outcome}"
        )

    if not (
        np.isfinite(treatment_group.to_numpy()).all()
        and np.isfinite(control_group.to_numpy()).all()
    ):

        raise ValueError(
            f"Non-finite values found in {outcome}."
        )

    return treatment_group, control_group

def wilson_interval(successes,sample_size):
    alpha=1-CONFIDENCE_INTERVAL
    z=norm.ppf(1-alpha/2)

    proportion=successes/sample_size

    denominator=1+(z**2/sample_size)

    center=(
        proportion+z**2/(2*sample_size)
    )/denominator

    margin=(z/denominator)*np.sqrt(proportion*(1-proportion)/sample_size+z**2/(4*sample_size**2))

    lower=center-margin
    upper=center+margin

    return lower,upper

def calculate_binary_ate(df,outcome):
    print(f"\nCalculating ATE for:{outcome}")

    treatment_group,control_group=get_groups(df,outcome)

    if not set(treatment_group.unique()).issubset({0,1}):
        raise ValueError(
            f"{outcome} must contain only 0 and 1"
        )

    if not set(control_group.unique()).issubset({0, 1}):
        raise ValueError(
            f"{outcome} must contain only 0 and 1"
        )

#Sample Sizes
    n_treatment=len(treatment_group)
    n_control=len(control_group)

    treatment_successes=int(treatment_group.sum())
    control_successes=int(control_group.sum())

    treatment_rate=treatment_group.mean()
    control_rate=control_group.mean()

    ate=treatment_rate-control_rate

    treatment_lower,treatment_upper=wilson_interval(treatment_successes,n_treatment)

    control_lower,control_upper=wilson_interval(control_successes,n_control)
    ci_lower=ate-np.sqrt(
        (treatment_rate-treatment_lower)**2
        +
        (control_upper-control_rate)**2
    )

    ci_upper=ate+np.sqrt(
        (treatment_upper-treatment_rate)**2
        +
        (control_rate-control_lower)**2
    )

    pooled_rate=(treatment_successes+control_successes)/(n_treatment+n_control)

    null_standard_error=np.sqrt(pooled_rate*(1-pooled_rate)*(1/n_treatment+1/n_control))

    z_statistics=ate/null_standard_error

    p_value=2*norm.sf(abs(z_statistics))

    if control_rate>0:
        relative_lift=(ate/control_rate)*100

    else:
        relative_lift=np.nan

    result={
        "outcome": outcome,

        "effect_type": "binary",

        "treatment_customers": n_treatment,

        "control_customers": n_control,

        "treatment_events": treatment_successes,

        "control_events": control_successes,

        "treatment_mean": treatment_rate,

        "control_mean": control_rate,

        "ate": ate,

        "ci_95_lower": ci_lower,

        "ci_95_upper": ci_upper,

        "p_value": p_value,

        "relative_lift_percent": relative_lift
    }
    return result

def calculate_continuous_ate(df,outcome):
    print(f"\nCalculating ATE for:{outcome}")

    treatment_group,control_group=get_groups(df,outcome)

    n_treatment=len(treatment_group)
    n_control=len(control_group)

    treatment_mean=treatment_group.mean()
    control_mean=control_group.mean()

    ate=treatment_mean-control_mean

    treatment_variance=treatment_group.var(ddof=1)
    control_variance=control_group.var(ddof=1)

    treatment_term=treatment_variance/n_treatment
    control_term=control_variance/n_control

    standard_error=np.sqrt(treatment_term+control_term)

    if standard_error==0:
        raise ValueError(
            f"Cannot calculate a meaningful test for {outcome}"
        )

    degrees_of_freedom=(
        (treatment_term+control_term)**2/(treatment_term**2/(n_treatment-1)+control_term**2/(n_control-1))
    )

    t_statistics=ate/standard_error
    alpha=1-CONFIDENCE_INTERVAL

    t_critical=student_t.ppf(1-alpha/2,degrees_of_freedom)

    margin_of_error=t_critical*standard_error

    ci_lower=ate-margin_of_error
    ci_upper=ate+margin_of_error

    p_value=2*student_t.sf(
        abs(t_statistics),
        degrees_of_freedom
    )

    if control_mean!=0:
        relative_lift=(ate/control_mean)*100

    else:
        relative_lift=np.nan

    result={
        "outcome": outcome,

        "effect_type": "continuous",

        "treatment_customers": n_treatment,

        "control_customers": n_control,

        "treatment_events": np.nan,

        "control_events": np.nan,

        "treatment_mean": treatment_mean,

        "control_mean": control_mean,

        "ate": ate,

        "ci_95_lower": ci_lower,

        "ci_95_upper": ci_upper,

        "p_value": p_value,

        "relative_lift_percent": relative_lift
    }

    return result

def apply_holm_correction(results):
    p_values=np.array([result["p_value"]for result in results])

    number_of_tests=len(p_values)

    sorted_indices=np.argsort(p_values)
    sorted_p_values=p_values[sorted_indices]

    adjusted_sorted=(number_of_tests-np.arange(number_of_tests))*sorted_p_values

    adjusted_sorted=np.maximum.accumulate(adjusted_sorted)

    adjusted_sorted=np.minimum(adjusted_sorted,1.0)

    adjusted_p_values=np.empty(number_of_tests)

    adjusted_p_values[sorted_indices]=adjusted_sorted

    for i,result in enumerate(results):
        result["p_value_holm"]=adjusted_p_values[i]

    return results

def print_ate_results(results):
    print("AVERAGE TREATMENT EFFECT RESULTS")

    for result in results:
        outcome=result["outcome"]
        print(f"\n{outcome.upper()}")

        print("Treatment Customers:",result["treatment_customers"])
        print("Control Customers:",result["control_customers"])

        if result["effect_type"]=="binary":
            print(
                f"Treatment Rate:{result['treatment_mean']*100:.4f}%"
            )

            print(
                f"Control Rate:{result['control_mean']*100:.4f}%"
            )

            print(
                f"ATE: "
                f"{result['ate'] * 100:.4f} percentage points"
            )

            print(
                f"95% Confidence Interval: "
                f"[{result['ci_95_lower'] * 100:.4f}, "
                f"{result['ci_95_upper'] * 100:.4f}] percentage points"
            )


        else:

            print(
                f"Treatment Mean: "
                f"{result['treatment_mean']:.4f}"
            )

            print(
                f"Control Mean: "
                f"{result['control_mean']:.4f}"
            )

            print(
                f"ATE: {result['ate']:.4f} spend units per customer"
            )

            print(
                f"95% Confidence Interval: "
                f"[{result['ci_95_lower']:.4f}, "
                f"{result['ci_95_upper']:.4f}]"
            )

        print(
            f"Relative Lift: "
            f"{result['relative_lift_percent']:.2f}%"
        )

        print(
            f"Raw p-value: {result['p_value']:.6e}"
        )

        print(
            f"Holm-adjusted p-value: "
            f"{result['p_value_holm']:.6e}"
        )

        if result["p_value_holm"] < SIGNIFICANCE_INTERVAL:

            print(
                "Conclusion: Statistically significant "
                "at the 5% level after Holm correction."
            )

        else:

            print(
                "Conclusion: Not statistically significant "
                "at the 5% level after Holm correction."
            )

def run_baseline():

    # Load randomized experiment data
    df = load_experiment_data()

    print("\nGROUP DISTRIBUTION")

    print(df[SEGMENT_COL].value_counts())

    # Conversion is our primary outcome
    conversion_result = calculate_binary_ate(
        df,
        TARGET_COL
    )

    # Visit is a secondary outcome
    visit_result = calculate_binary_ate(
        df,
        "visit"
    )

    # Spending is a secondary monetary outcome
    spend_result = calculate_continuous_ate(
        df,
        "spend"
    )

    # Combine results
    results = [
        conversion_result,
        visit_result,
        spend_result
    ]

    # Adjust for multiple statistical tests
    results = apply_holm_correction(results)

    # Display results
    print_ate_results(results)

    results_df = pd.DataFrame(results)

    output_path = REPORTS_DIR / "ate_summary.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print("Location:", output_path)

if __name__ == "__main__":

    run_baseline()
