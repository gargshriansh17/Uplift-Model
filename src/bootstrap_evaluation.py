import itertools
import numpy as np
import pandas as pd

from src.config import REPORTS_DIR,RANDOM_STATE

from src.model_comparison import (
    load_all_predictions,
    calculate_group_uplift,
    calculate_qini_curve,
    calculate_qini_auuc
)

OUTPUT_DIR=REPORTS_DIR/"model_comparison"/"bootstrap"

BOOTSTRAP_REPEATS=2000

POINT_TIE_REPEATS=200

CONFIDENCE_LEVEL=0.95

TARGET_PERCENTAGES=[0.10,0.20,0.30]

METRICES=[
    "qini_coefficient",
    "uplift_at_10_pp",
    "uplift_at_20_pp",
    "uplift_at_30_pp"
]

def calculate_bootstrap_metrices(validation_df,row_order=None):
    if row_order is not None:
        working_df=validation_df.iloc[row_order].reset_index(drop=True)

    else:
        working_df=validation_df.reset_index(drop=True)

    ranked_df=working_df.sort_values(
        "predicted_uplift",
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    results={}

    total_customers=len(ranked_df)

    for percentage in TARGET_PERCENTAGES:
        k=int(np.ceil(total_customers*percentage))

        top_k=ranked_df.iloc[:k]

        try:
            group_result=calculate_group_uplift(top_k)

            results[f"uplift_at_{int(percentage*100)}_pp"]=(group_result["observed_uplift_pp"])

        except ValueError:
            results[f"uplift_at_{int(percentage*100)}_pp"]=np.nan

    qini_df,_=calculate_qini_curve(working_df)

    qini_summary=calculate_qini_auuc(qini_df)

    results["qini_coefficient"]=(qini_summary["qini_coefficient"])

    return results

def calculate_tie_diagnostic(predictions):
    results=[]

    for model_name,df in predictions.items():
        score_counts=df["predicted_uplift"].value_counts()
        tied_groups=score_counts[score_counts>1]
        tied_customers=int(tied_groups.sum())

        results.append({
            "model": model_name,
            "total_customers": len(df),
            "unique_uplift_scores": int(score_counts.size),
            "customers_in_tied_groups": tied_customers,
            "tied_customer_percentage": (
                tied_customers / len(df) * 100
            ),
            "largest_tie_group": (
                int(score_counts.max())
            )
        })

    return pd.DataFrame(results)


def run_bootstrap():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    rng=np.random.default_rng(RANDOM_STATE)

    predictions=load_all_predictions()

    model_names=list(predictions.keys())

    reference_df=predictions[model_names[0]]

    treatment_positions=np.flatnonzero(reference_df["actual_treatment"].to_numpy()==1)

    control_positions=np.flatnonzero(reference_df["actual_treatment"].to_numpy()==0)

    n_treatment=len(treatment_positions)
    n_control=len(control_positions)

    total_customers=len(reference_df)

    print("\nVALIDATION DATA")

    print("Total customers:", total_customers)
    print("Treatment customers:", n_treatment)
    print("Control customers:", n_control)

    print("\nBOOTSTRAP CONFIGURATION")

    print("Bootstrap repetitions:", BOOTSTRAP_REPEATS)
    print("Confidence level:", CONFIDENCE_LEVEL * 100, "%")
    print("Point-estimate tie repetitions:", POINT_TIE_REPEATS)

    tie_df=calculate_tie_diagnostic(predictions)

    tie_df.to_csv(OUTPUT_DIR/"ranking_tie_diagnostics.csv",index=False)

    print("\nRANKING TIE DIAGNOSTICS")

    print(tie_df.round(3).to_string(index=False))

    original_points={}

    point_permutations=[
        rng.permutation(total_customers)
        for _ in range(POINT_TIE_REPEATS)
    ]

    tie_averaged_points={}

    print("\nCALCULATING POINT ESTIMATES")

    for model_name,df in predictions.items():
        original_points[model_name]=(calculate_bootstrap_metrices(df))

        tie_results=[]

        for order in point_permutations:
            tie_results.append(calculate_bootstrap_metrices(df,order))

        tie_results_df=pd.DataFrame(tie_results)

        tie_averaged_points[model_name]=(tie_results_df.mean().to_dict())

    bootstrap_records=[]

    for bootstrap_id in range(1,BOOTSTRAP_REPEATS+1):
        treatment_draw=rng.choice(
            treatment_positions,
            size=n_treatment,
            replace=True
        )

        control_draw=rng.choice(
            control_positions,
            size=n_control,
            replace=True
        )

        sampled_positions=np.concatenate([
            treatment_draw,
            control_draw
        ])

        rng.shuffle(sampled_positions)

        for model_name,df in predictions.items():
            sampled_df=df.iloc[sampled_positions].reset_index(drop=True)

            metrics=calculate_bootstrap_metrices(sampled_df)

            bootstrap_records.append({
                "bootstrap_id":bootstrap_id,
                "model":model_name,
                **metrics
            })

        if bootstrap_id%200==0:
            print(f"Completed {bootstrap_id}/{BOOTSTRAP_REPEATS} bootstrap repetitions")

    bootstrap_df=pd.DataFrame(bootstrap_records)

    bootstrap_df.to_csv(OUTPUT_DIR/"bootstrap_distributions.csv",index=False)

    alpha=(1-CONFIDENCE_LEVEL)/2

    model_summary=[]

    for model_name in model_names:
        model_bootstrap=bootstrap_df[
            bootstrap_df["model"]==model_name
        ]

        for metric in METRICES:

            values=model_bootstrap[metric].dropna().to_numpy()

            if len(values)==0:
                raise ValueError(
                    f"No valid bootstrap results for {model_name}:{metric}"
                )

            lower=np.quantile(values,alpha)
            upper=np.quantile(values,1-alpha)

            model_summary.append({
                "model":model_name,
                "metric":metric,
                "original_point_estimate":(original_points[model_name][metric]),
                "tie_averaged_point_estimate":(tie_averaged_points[model_name][metric]),
                "bootstrap_mean":np.mean(values),
                "bootstrap_std":np.std(values,ddof=1),
                "ci_lower":lower,
                "ci_upper":upper,
                "valid_repetitions":len(values)
            })

    model_summary_df=pd.DataFrame(model_summary)

    model_summary_df.to_csv(
        OUTPUT_DIR/"bootstrap_model_summary.csv",index=False
    )

    pairwise_results=[]

    for model_a,model_b in itertools.combinations(model_names,2):
        for metric in METRICES:
            wide=bootstrap_df.pivot(
                index="bootstrap_id",
                columns="model",
                values=metric
            )

            differences=(
                wide[model_a]-wide[model_b]
            ).dropna().to_numpy()

            if len(differences)==0:
                raise ValueError(
                    f"No valid paired bootstrap differences for {model_a} vs {model_b}:{metric}"
                )

            original_differences=(
                original_points[model_a][metric]
                -
                original_points[model_b][metric]
            )

            tie_avg_difference=(
                tie_averaged_points[model_a][metric]
                -
                tie_averaged_points[model_b][metric]
            )

            pairwise_results.append({
                "model_a": model_a,

                "model_b": model_b,

                "metric": metric,

                "difference_definition": "model_a - model_b",

                "original_point_difference": original_differences,

                "tie_averaged_point_difference": tie_avg_difference,

                "bootstrap_mean_difference": np.mean(differences),

                "bootstrap_std_difference": np.std(
                    differences,
                    ddof=1
                ),

                "ci_lower": np.quantile(differences, alpha),

                "ci_upper": np.quantile(
                    differences,
                    1 - alpha
                ),

                "valid_repetitions": len(differences)

            })

    pairwise_df=pd.DataFrame(pairwise_results)

    pairwise_df.to_csv(OUTPUT_DIR/"bootstrap_pairwise_comparison.csv",index=False)

    print("\n" + "=" * 75)

    print("MODEL-LEVEL BOOTSTRAP CONFIDENCE INTERVALS")

    print("=" * 75)

    print(
        model_summary_df.round(5).to_string(index=False)
    )

    print("\n" + "=" * 75)

    print("PAIRED MODEL DIFFERENCES")

    print("=" * 75)

    print(
        pairwise_df.round(5).to_string(index=False)
    )

    print("Model summary:", OUTPUT_DIR / "bootstrap_model_summary.csv")

    print(
        "Pairwise comparison:",
        OUTPUT_DIR / "bootstrap_pairwise_comparison.csv"
    )

    print(
        "Bootstrap distributions:",
        OUTPUT_DIR / "bootstrap_distributions.csv"
    )

    print(
        "Tie diagnostics:",
        OUTPUT_DIR / "ranking_tie_diagnostics.csv"
    )

    return model_summary_df, pairwise_df, bootstrap_df


if __name__ == "__main__":

    run_bootstrap()



    