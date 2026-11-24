from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR,REPORTS_DIR,TARGET_COL

DATA_DIR=PROCESSED_DATA_DIR/"three_arm"
REPORT_DIR=REPORTS_DIR/"three_arm"
VAL_PATH=DATA_DIR/"validation_data.csv"
LR_PREDICTIONS_PATH=REPORT_DIR/"logistic_t_learner_validation_predictions.csv"
LGBM_PREDICTIONS_PATH=REPORT_DIR/"lightgbm_t_learner_validation_predictions.csv"

ARM_LABELS={0:"No Email",
            1:"Mens E-Mail",
            2:"Womens E-Mail"}

PROPENSITIES={0:1/3,
              1:1/3,
              2:1/3}

BOOTSTRAP_REPS=2000

BOOTSTRAP_RANDOM_STATE=2026

POTENTIAL_OUTCOME_COLS={
    0:"predicted_conversion_no_email",
    1:"predicted_conversion_mens_email",
    2:"predicted_conversion_womens_email"
}

POLICY_LABELS={
    "Always No Email":0,
    "Always Mens Email":1,
    "Always Womens Email":2
}

def _load_and_validate():
    required_files=[VAL_PATH,
                    LR_PREDICTIONS_PATH,
                    LGBM_PREDICTIONS_PATH]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required validation/evaluation file not found:{path}"
            )

    val=pd.read_csv(VAL_PATH)

    if "source_row_index" not in val.columns:
        raise ValueError("Validation data must contain source_row_index")
    if val["source_row_index"].duplicated().any():
        raise ValueError("Duplicate source_row_index values in validation data")

    val=val.set_index("source_row_index").sort_index()
    actual_treatment=val["treatment"].astype(int).to_numpy()
    actual_outcome=val[TARGET_COL].astype(int).to_numpy()

    if not set(np.unique(actual_treatment)).issubset({0,1,2}):
        raise ValueError("Validation treatment must contain only 0,1 and 2")

    if not set(np.unique(actual_outcome)).issubset({0,1}):
        raise ValueError("Validation conversion must contain only 0 and 1")

    prediction_files={
        "Logistic Regression":LR_PREDICTIONS_PATH,
        "LightGBM":LGBM_PREDICTIONS_PATH
    }
    predictions={}

    for model_name,path in prediction_files.items():
        pred=pd.read_csv(path)
        required=[
            "source_row_index",
            "actual_treatment",
            "actual_conversion",
            "predicted_best_treatment",
            *POTENTIAL_OUTCOME_COLS.values()
        ]

        missing=[col for col in required if col not in pred.columns]

        if missing:
            raise ValueError(f"{path.name} is missing columns:{missing}")

        if pred["source_row_index"].duplicated().any():
            raise ValueError(f"Duplicate source_row_index values in {path.name}")

        pred=pred.set_index("source_row_index")
        if not pred.index.equals(val.index):
            if set(pred.index)!=set(val.index):
                raise ValueError(
                    f"Source-row IDs in {path.name} do not match validation data"
                )

            pred=pred.loc[val.index]

        if not np.array_equal(pred["actual_treatment"].astype(int),actual_treatment):
            raise ValueError(f"Treatment values in {path.name} do not match validation data")
        if not np.array_equal(pred["actual_conversion"].astype(int),actual_outcome):
            raise ValueError(f"Conversion values in {path.name} do not match validation data")

        actions=pred["predicted_best_treatment"].astype(int).to_numpy()
        if not set(np.unique(actions)).issubset({0,1,2}):
            raise ValueError(f"Predicted policy actions in {path.name} must be 0,1 or 2")

        mu=np.column_stack([
            pd.to_numeric(pred[POTENTIAL_OUTCOME_COLS[arm]],errors="raise").to_numpy()
            for arm in (0,1,2)
        ])

        if not np.isfinite(mu).all() or np.any((mu<0)|(mu>1)):
            raise ValueError(f"Invalid potential-outcome probabilities in {path.name}")

        predictions[model_name]={"actions":actions,"mu":mu}

    return val,actual_treatment,actual_outcome,predictions

def _build_contribution_columns(
        actual_treatment,
        outcome,
        predictions
):

    n=len(outcome)
    rows=np.arange(n)

    policies={
        name:np.full(n,arm,dtype=int)
        for name,arm in POLICY_LABELS.items()
    }

    policies["Logistic Regression policy"]=predictions["Logistic Regression"]["actions"]
    policies["LightGBM policy"]=predictions["LightGBM"]["actions"]

    contribution_cols={}
    for policy_name,policy_actions in policies.items():
        matched=actual_treatment==policy_actions
        inverse_propensity=np.array(
            [1/PROPENSITIES[int(arm)] for arm in policy_actions],dtype=float
        )

        contribution_cols[("IPW","None",policy_name)]=(
            matched*outcome*inverse_propensity
        )

    for nuisance_model,model_data in predictions.items():
        mu=model_data["mu"]
        mu_observed_arm=mu[rows,actual_treatment]

        for policy_name,policy_actions in policies.items():
            matched=actual_treatment==policy_actions
            mu_recommended_arm=mu[rows,policy_actions]
            inverse_propensity=np.array(
                [1/PROPENSITIES[int(arm)] for arm in policy_actions],dtype=float
            )

            contribution_cols[("DR",nuisance_model,policy_name)]=(
                mu_recommended_arm
                +
                matched*inverse_propensity*(outcome-mu_observed_arm)
            )

    return policies,contribution_cols


def _bootstrap_means(contribution_cols,n):
    keys=list(contribution_cols)
    matrix=np.column_stack([contribution_cols[key]for key in keys])
    point_estimates=matrix.mean(axis=0)

    rng=np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    bootstrap=np.empty((BOOTSTRAP_REPS,len(keys)),dtype=float)
    for b in range(BOOTSTRAP_REPS):
        sample_indices=rng.integers(0,n,size=n)
        bootstrap[b]=matrix[sample_indices].mean(axis=0)

    return keys,point_estimates,bootstrap

def _make_reports(validation,actual_treatment,policies,keys,point,bootstrap):
    key_to_col={key:idx for idx,key in enumerate(keys)}
    value_rows=[]

    for idx,key in enumerate(keys):
        estimator,nuisance_model,policy_name=key
        estimate=float(point[idx])
        lower,upper=np.percentile(bootstrap[:,idx],[2.5,97.5])
        value_rows.append({
            "estimator": estimator,
            "nuisance_model": nuisance_model,
            "policy": policy_name,
            "estimated_conversion_rate": estimate,
            "conversion_rate_percent": estimate * 100,
            "ci_lower_percent": float(lower * 100),
            "ci_upper_percent": float(upper * 100)
        })

    allocation_rows=[]
    for policy_name,actions in policies.items():
        counts=pd.Series(actions).value_counts().reindex([0,1,2],fill_value=0)
        for arm,count in counts.items():
            allocation_rows.append({
                "policy": policy_name,
                "recommended_campaign": ARM_LABELS[int(arm)],
                "customers": int(count),
                "share_percent": float(count / len(actions) * 100)
            })

    comparison_rows=[]
    learned_policies=["Logistic Regression policy","LightGBM policy"]
    fixed_policies=list(POLICY_LABELS)

    comparison_groups=[("IPW","None")]
    comparison_groups.extend([("DR",model_name) for model_name in predictions_names()])

    for estimator,nuisance_model in comparison_groups:
        for learned in learned_policies:
            learned_key=(estimator,nuisance_model,learned)
            learned_idx=key_to_col[learned_key]
            for reference in fixed_policies + [
                p for p in learned_policies if p != learned
            ]:
                reference_key = (estimator, nuisance_model, reference)
                reference_idx = key_to_col[reference_key]
                diff = float(point[learned_idx] - point[reference_idx])
                boot_diff = bootstrap[:, learned_idx] - bootstrap[:, reference_idx]
                lower, upper = np.percentile(boot_diff, [2.5, 97.5])
                comparison_rows.append({
                    "estimator": estimator,
                    "nuisance_model": nuisance_model,
                    "learned_policy": learned,
                    "compared_with": reference,
                    "incremental_conversion_rate": diff,
                    "incremental_uplift_percentage_points": diff * 100,
                    "ci_lower_percentage_points": float(lower * 100),
                    "ci_upper_percentage_points": float(upper * 100),
            })

    return (
        pd.DataFrame(value_rows),
        pd.DataFrame(allocation_rows),
        pd.DataFrame(comparison_rows)
    )

def predictions_names():
    return ("Logistic Regression","LightGBM")

def run_policy_evaluation():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    validation, actual_treatment, outcome, predictions = _load_and_validate()

    print("\nTHREE-ARM POLICY EVALUATION")
    print("Validation customers:", len(validation))
    print("Validation conversions:", int(outcome.sum()))
    print("Randomization propensity used: 1/3 per campaign arm")
    print("Observed validation treatment distribution:")
    print(pd.Series(actual_treatment).map(ARM_LABELS).value_counts().to_string())

    policies, contribution_columns = _build_contribution_columns(
        actual_treatment, outcome, predictions
    )
    keys, point, bootstrap = _bootstrap_means(contribution_columns, len(validation))
    values_df, allocation_df, comparisons_df = _make_reports(
        validation, actual_treatment, policies, keys, point, bootstrap
    )

    values_path = REPORT_DIR / "three_arm_policy_value.csv"
    allocation_path = REPORT_DIR / "three_arm_policy_allocation.csv"
    comparisons_path = REPORT_DIR / "three_arm_policy_comparisons.csv"

    values_df.to_csv(values_path, index=False)
    allocation_df.to_csv(allocation_path, index=False)
    comparisons_df.to_csv(comparisons_path, index=False)

    display = values_df.copy()
    display["estimated_conversion_rate"] = display["conversion_rate_percent"].map(
        lambda x: f"{x:.4f}%"
    )
    display["95% CI"] = display.apply(
        lambda row: f"[{row['ci_lower_percent']:.4f}%, {row['ci_upper_percent']:.4f}%]",
        axis=1,
    )
    print("\nPOLICY VALUE ESTIMATES")
    print(display[["estimator", "nuisance_model", "policy", "estimated_conversion_rate", "95% CI"]].to_string(index=False))

    print("\nPOLICY ALLOCATION")
    print(allocation_df.pivot(
        index="policy", columns="recommended_campaign", values="share_percent"
    ).round(2).to_string())

    print("\nLEARNED POLICY COMPARISONS (percentage points)")
    selected = comparisons_df[
        comparisons_df["compared_with"].isin(fixed_policies_for_print())
    ].copy()
    print(selected[[
        "estimator", "nuisance_model", "learned_policy", "compared_with",
        "incremental_uplift_percentage_points", "ci_lower_percentage_points",
        "ci_upper_percentage_points",
    ]].round(4).to_string(index=False))

    print("\nReports saved:")
    print(values_path)
    print(allocation_path)
    print(comparisons_path)

    return values_df, allocation_df, comparisons_df


def fixed_policies_for_print():
    return list(POLICY_LABELS)


if __name__ == "__main__":
    run_policy_evaluation()
