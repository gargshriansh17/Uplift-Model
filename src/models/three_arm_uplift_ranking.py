"""Pairwise uplift-ranking evaluation for the Hillstrom three-arm experiment.

Primary contrasts:
    1 = Men's Email vs 0 = No Email
    2 = Women's Email vs 0 = No Email

Secondary contrast:
    1 = Men's Email vs 2 = Women's Email

For each contrast, the script evaluates predicted pairwise CATE rankings using
an inverse-propensity-weighted cumulative gain curve. Because the original
three-arm experiment randomizes equally across arms, the conditional treatment
propensity within any selected pair is 0.5.

The Qini coefficient is defined here as the area between the model's cumulative
gain curve and the straight-line random-targeting baseline. AUUC is the area
under the cumulative gain curve. Tied uplift scores are handled as a tied block:
customers at the same score are treated as exchangeable, avoiding arbitrary
ranking based on row order.

Bootstrap confidence intervals use stratified resampling within the two actual
randomized arms. The same sampled customers are used for both models so their
metric differences are paired. The untouched test split is never loaded.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, TARGET_COL


# ---------------------------------------------------------------------------
# Paths and experiment definitions
# ---------------------------------------------------------------------------
THREE_ARM_DATA_DIR = PROCESSED_DATA_DIR / "three_arm"
THREE_ARM_REPORT_DIR = REPORTS_DIR / "three_arm"
FIGURE_DIR = THREE_ARM_REPORT_DIR / "figures"

VALIDATION_PATH = THREE_ARM_DATA_DIR / "validation_data.csv"
LR_PREDICTIONS_PATH = (
    THREE_ARM_REPORT_DIR / "logistic_t_learner_validation_predictions.csv"
)
LGBM_PREDICTIONS_PATH = (
    THREE_ARM_REPORT_DIR / "lightgbm_t_learner_validation_predictions.csv"
)

ARM_LABELS = {
    0: "No Email",
    1: "Mens E-Mail",
    2: "Womens E-Mail",
}

POTENTIAL_OUTCOME_COLUMNS = {
    0: "predicted_conversion_no_email",
    1: "predicted_conversion_mens_email",
    2: "predicted_conversion_womens_email",
}

MODEL_NAMES = ("Logistic Regression", "LightGBM")

# The order matters only for display and output consistency.
CONTRASTS = (
    {
        "contrast": "Mens_vs_No_Email",
        "treated_arm": 1,
        "control_arm": 0,
        "primary": True,
    },
    {
        "contrast": "Womens_vs_No_Email",
        "treated_arm": 2,
        "control_arm": 0,
        "primary": True,
    },
    {
        "contrast": "Mens_vs_Womens_Email",
        "treated_arm": 1,
        "control_arm": 2,
        "primary": False,
    },
)

# Conditional on being in a given pair, random assignment is 50/50.
PAIRWISE_TREATMENT_PROPENSITY = 0.5
TARGET_FRACTIONS = (0.10, 0.20, 0.30)
CURVE_GRID = np.linspace(0.0, 1.0, 101)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_RANDOM_STATE = 2027

SUMMARY_PATH = THREE_ARM_REPORT_DIR / "three_arm_uplift_ranking_summary.csv"
CURVES_PATH = THREE_ARM_REPORT_DIR / "three_arm_uplift_ranking_curves.csv"
COMPARISONS_PATH = (
    THREE_ARM_REPORT_DIR / "three_arm_uplift_ranking_model_comparisons.csv"
)
METHODOLOGY_PATH = (
    THREE_ARM_REPORT_DIR / "three_arm_uplift_ranking_methodology.txt"
)

METRIC_NAMES = (
    "qini_coefficient_per_1000_customers",
    "auuc_per_1000_customers",
    "overall_pairwise_uplift_pp",
    "uplift_at_10_percent_pp",
    "uplift_at_20_percent_pp",
    "uplift_at_30_percent_pp",
)


# ---------------------------------------------------------------------------
# Loading and input validation
# ---------------------------------------------------------------------------
def _read_source_indexed_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    source_ids = pd.to_numeric(df["source_row_index"], errors="raise")
    if source_ids.isna().any() or not np.equal(source_ids, np.floor(source_ids)).all():
        raise ValueError(f"Invalid source_row_index values in {path.name}.")
    df["source_row_index"] = source_ids.astype(np.int64)

    if df["source_row_index"].duplicated().any():
        raise ValueError(f"Duplicate source_row_index values in {path.name}.")

    return df.set_index("source_row_index").sort_index()


def load_validation_and_predictions():
    validation = _read_source_indexed_csv(
        VALIDATION_PATH,
        ["source_row_index", TARGET_COL, "treatment"],
    )

    treatment_numeric = pd.to_numeric(validation["treatment"], errors="raise")
    outcome_numeric = pd.to_numeric(validation[TARGET_COL], errors="raise")

    if treatment_numeric.isna().any() or not set(treatment_numeric.unique()).issubset({0, 1, 2}):
        raise ValueError("Validation treatment must contain only 0, 1, and 2.")
    if outcome_numeric.isna().any() or not set(outcome_numeric.unique()).issubset({0, 1}):
        raise ValueError("Validation conversion must contain only 0 and 1.")

    actual_treatment = treatment_numeric.astype(int).to_numpy()
    actual_outcome = outcome_numeric.astype(int).to_numpy()

    prediction_paths = {
        "Logistic Regression": LR_PREDICTIONS_PATH,
        "LightGBM": LGBM_PREDICTIONS_PATH,
    }
    predictions = {}

    required_prediction_columns = [
        "source_row_index",
        "actual_treatment",
        "actual_conversion",
        *POTENTIAL_OUTCOME_COLUMNS.values(),
    ]

    for model_name, path in prediction_paths.items():
        pred = _read_source_indexed_csv(path, required_prediction_columns)

        if not pred.index.equals(validation.index):
            if not pred.index.difference(validation.index).empty or not validation.index.difference(pred.index).empty:
                raise ValueError(
                    f"Source-row IDs in {path.name} do not match validation data."
                )
            pred = pred.loc[validation.index]

        pred_treatment = pd.to_numeric(pred["actual_treatment"], errors="raise").astype(int).to_numpy()
        pred_outcome = pd.to_numeric(pred["actual_conversion"], errors="raise").astype(int).to_numpy()

        if not np.array_equal(pred_treatment, actual_treatment):
            raise ValueError(f"Treatment values in {path.name} do not match validation data.")
        if not np.array_equal(pred_outcome, actual_outcome):
            raise ValueError(f"Conversion values in {path.name} do not match validation data.")

        mu = np.column_stack(
            [
                pd.to_numeric(pred[POTENTIAL_OUTCOME_COLUMNS[arm]], errors="raise").to_numpy(dtype=float)
                for arm in (0, 1, 2)
            ]
        )

        if not np.isfinite(mu).all() or np.any((mu < 0.0) | (mu > 1.0)):
            raise ValueError(f"Invalid potential-outcome probabilities in {path.name}.")

        predictions[model_name] = mu

    return validation, actual_treatment, actual_outcome, predictions


# ---------------------------------------------------------------------------
# Uplift / Qini metrics
# ---------------------------------------------------------------------------
def _area_under_curve(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal area without relying on NumPy-version-specific APIs."""
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))


def calculate_ranking_metrics(
    scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    curve_grid: np.ndarray | None = None,
):
    """Calculate tie-aware cumulative gain, AUUC, Qini and uplift@K.

    treatment is binary within the pair: 1 = contrast's treated arm,
    0 = contrast's control arm.

    The per-customer inverse-propensity contribution is:
        I(T=1) * Y / p - I(T=0) * Y / (1-p)
    With p=0.5 this is +2Y for observed treated customers and -2Y for
    observed control customers.
    """
    scores = np.asarray(scores, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    outcome = np.asarray(outcome, dtype=float)

    if not (len(scores) == len(treatment) == len(outcome)):
        raise ValueError("Scores, treatment and outcome must have equal lengths.")
    if len(scores) == 0:
        raise ValueError("Cannot calculate uplift metrics on an empty pairwise sample.")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite uplift scores encountered.")
    if not set(np.unique(treatment)).issubset({0, 1}):
        raise ValueError("Pairwise treatment indicator must contain only 0 and 1.")

    p = PAIRWISE_TREATMENT_PROPENSITY
    contribution = np.where(
        treatment == 1,
        outcome / p,
        -outcome / (1.0 - p),
    )

    # Stable descending sort, then aggregate equal scores into indivisible
    # tie blocks. Linear interpolation inside a tie block represents random
    # ordering among customers whose model scores are identical.
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_contribution = contribution[order]

    group_starts = np.r_[
        0,
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1,
    ]
    group_counts = np.diff(np.r_[group_starts, len(scores)])
    group_gain = np.add.reduceat(sorted_contribution, group_starts)

    cumulative_customers = np.cumsum(group_counts)
    cumulative_gain = np.cumsum(group_gain)

    x_nodes = np.r_[0.0, cumulative_customers / len(scores)]
    gain_nodes = np.r_[0.0, cumulative_gain]

    auuc_raw = _area_under_curve(x_nodes, gain_nodes)
    random_area_raw = 0.5 * float(gain_nodes[-1])
    qini_raw = auuc_raw - random_area_raw
    n = len(scores)

    metrics = {
        "qini_coefficient_per_1000_customers": qini_raw / n * 1000.0,
        "auuc_per_1000_customers": auuc_raw / n * 1000.0,
        "overall_pairwise_uplift_pp": float(gain_nodes[-1] / n * 100.0),
    }

    for fraction in TARGET_FRACTIONS:
        gain_at_fraction = float(np.interp(fraction, x_nodes, gain_nodes))
        uplift_pp = gain_at_fraction / (fraction * n) * 100.0
        key = f"uplift_at_{int(fraction * 100)}_percent_pp"
        metrics[key] = uplift_pp

    curve = None
    if curve_grid is not None:
        interpolated_gain = np.interp(curve_grid, x_nodes, gain_nodes)
        gain_per_1000 = interpolated_gain / n * 1000.0
        random_baseline_per_1000 = (
            curve_grid * float(gain_nodes[-1]) / n * 1000.0
        )
        curve = {
            "gain_per_1000": gain_per_1000,
            "random_baseline_per_1000": random_baseline_per_1000,
        }

    return metrics, curve


# ---------------------------------------------------------------------------
# Pairwise bootstrap evaluation
# ---------------------------------------------------------------------------
def evaluate_contrast(
    contrast: dict,
    actual_treatment: np.ndarray,
    actual_outcome: np.ndarray,
    predictions: dict[str, np.ndarray],
    rng: np.random.Generator,
):
    treated_arm = contrast["treated_arm"]
    control_arm = contrast["control_arm"]
    contrast_name = contrast["contrast"]

    pair_indices = np.flatnonzero(
        (actual_treatment == treated_arm) | (actual_treatment == control_arm)
    )
    pair_actual_treatment = actual_treatment[pair_indices]
    pair_outcome = actual_outcome[pair_indices]
    pair_treatment_binary = (pair_actual_treatment == treated_arm).astype(int)

    treated_local_indices = np.flatnonzero(pair_treatment_binary == 1)
    control_local_indices = np.flatnonzero(pair_treatment_binary == 0)
    n_treated = len(treated_local_indices)
    n_control = len(control_local_indices)

    if n_treated == 0 or n_control == 0:
        raise ValueError(f"Both arms are required for {contrast_name}.")

    score_by_model = {
        model_name: mu[pair_indices, treated_arm] - mu[pair_indices, control_arm]
        for model_name, mu in predictions.items()
    }

    point_metrics = {}
    point_curves = {}
    for model_name, scores in score_by_model.items():
        metrics, curve = calculate_ranking_metrics(
            scores,
            pair_treatment_binary,
            pair_outcome,
            curve_grid=CURVE_GRID,
        )
        point_metrics[model_name] = metrics
        point_curves[model_name] = curve

    bootstrap_metrics = {
        model_name: {metric: np.empty(BOOTSTRAP_REPS, dtype=float) for metric in METRIC_NAMES}
        for model_name in MODEL_NAMES
    }
    bootstrap_curves = {
        model_name: np.empty((BOOTSTRAP_REPS, len(CURVE_GRID)), dtype=float)
        for model_name in MODEL_NAMES
    }

    for b in range(BOOTSTRAP_REPS):
        # Stratify by randomized arm to preserve the observed arm sample sizes.
        sampled_treated = rng.choice(
            treated_local_indices, size=n_treated, replace=True
        )
        sampled_control = rng.choice(
            control_local_indices, size=n_control, replace=True
        )
        sampled = np.concatenate([sampled_treated, sampled_control])

        treatment_boot = np.r_[
            np.ones(n_treated, dtype=int),
            np.zeros(n_control, dtype=int),
        ]
        outcome_boot = pair_outcome[sampled]

        # Use exactly the same sampled customers for both models: paired CIs.
        for model_name in MODEL_NAMES:
            score_boot = score_by_model[model_name][sampled]
            metrics_boot, curve_boot = calculate_ranking_metrics(
                score_boot,
                treatment_boot,
                outcome_boot,
                curve_grid=CURVE_GRID,
            )
            for metric in METRIC_NAMES:
                bootstrap_metrics[model_name][metric][b] = metrics_boot[metric]
            bootstrap_curves[model_name][b] = curve_boot["gain_per_1000"]

    summary_rows = []
    curve_rows = []
    for model_name in MODEL_NAMES:
        row = {
            "contrast": contrast_name,
            "primary_contrast": bool(contrast["primary"]),
            "model": model_name,
            "treated_campaign": ARM_LABELS[treated_arm],
            "control_campaign": ARM_LABELS[control_arm],
            "pairwise_customers": int(len(pair_indices)),
            "treated_customers": int(n_treated),
            "control_customers": int(n_control),
            "treated_conversions": int(pair_outcome[pair_treatment_binary == 1].sum()),
            "control_conversions": int(pair_outcome[pair_treatment_binary == 0].sum()),
        }

        for metric in METRIC_NAMES:
            estimate = float(point_metrics[model_name][metric])
            lower, upper = np.percentile(
                bootstrap_metrics[model_name][metric], [2.5, 97.5]
            )
            row[metric] = estimate
            row[f"{metric}_ci_lower"] = float(lower)
            row[f"{metric}_ci_upper"] = float(upper)

        summary_rows.append(row)

        gain_ci_lower, gain_ci_upper = np.percentile(
            bootstrap_curves[model_name], [2.5, 97.5], axis=0
        )
        for idx, fraction in enumerate(CURVE_GRID):
            curve_rows.append({
                "contrast": contrast_name,
                "primary_contrast": bool(contrast["primary"]),
                "model": model_name,
                "targeted_share": float(fraction),
                "targeted_share_percent": float(fraction * 100.0),
                "cumulative_gain_per_1000_customers": float(
                    point_curves[model_name]["gain_per_1000"][idx]
                ),
                "random_targeting_baseline_per_1000": float(
                    point_curves[model_name]["random_baseline_per_1000"][idx]
                ),
                "gain_ci_lower_per_1000": float(gain_ci_lower[idx]),
                "gain_ci_upper_per_1000": float(gain_ci_upper[idx]),
            })

    # Paired model differences: Logistic Regression minus LightGBM.
    comparison_rows = []
    for metric in METRIC_NAMES:
        difference = float(point_metrics["Logistic Regression"][metric]
                           - point_metrics["LightGBM"][metric])
        bootstrap_difference = (
            bootstrap_metrics["Logistic Regression"][metric]
            - bootstrap_metrics["LightGBM"][metric]
        )
        lower, upper = np.percentile(bootstrap_difference, [2.5, 97.5])
        comparison_rows.append({
            "contrast": contrast_name,
            "primary_contrast": bool(contrast["primary"]),
            "metric": metric,
            "difference_lr_minus_lightgbm": difference,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "interpretation": "Positive favors Logistic Regression; negative favors LightGBM.",
        })

    diagnostics = {
        "contrast": contrast_name,
        "pairwise_customers": int(len(pair_indices)),
        "treated_arm": ARM_LABELS[treated_arm],
        "control_arm": ARM_LABELS[control_arm],
        "treated_customers": int(n_treated),
        "control_customers": int(n_control),
        "treated_conversions": int(pair_outcome[pair_treatment_binary == 1].sum()),
        "control_conversions": int(pair_outcome[pair_treatment_binary == 0].sum()),
    }

    return summary_rows, curve_rows, comparison_rows, diagnostics


# ---------------------------------------------------------------------------
# Plotting and reporting
# ---------------------------------------------------------------------------
def _contrast_title(contrast_name: str) -> str:
    titles = {
        "Mens_vs_No_Email": "Men's Email vs No Email",
        "Womens_vs_No_Email": "Women's Email vs No Email",
        "Mens_vs_Womens_Email": "Men's Email vs Women's Email",
    }
    return titles[contrast_name]


def save_qini_plots(curves_df: pd.DataFrame):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for contrast in CONTRASTS:
        contrast_name = contrast["contrast"]
        subset = curves_df[curves_df["contrast"] == contrast_name]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 6))
        random_line_drawn = False

        for model_name in MODEL_NAMES:
            model_curve = subset[subset["model"] == model_name].sort_values(
                "targeted_share"
            )
            if model_curve.empty:
                continue

            ax.plot(
                model_curve["targeted_share_percent"],
                model_curve["cumulative_gain_per_1000_customers"],
                linewidth=2,
                label=model_name,
            )
            ax.fill_between(
                model_curve["targeted_share_percent"].to_numpy(),
                model_curve["gain_ci_lower_per_1000"].to_numpy(),
                model_curve["gain_ci_upper_per_1000"].to_numpy(),
                alpha=0.15,
            )

            if not random_line_drawn:
                ax.plot(
                    model_curve["targeted_share_percent"],
                    model_curve["random_targeting_baseline_per_1000"],
                    linestyle="--",
                    linewidth=1.5,
                    label="Random targeting",
                )
                random_line_drawn = True

        ax.axhline(0, linewidth=0.8, alpha=0.6)
        ax.set_title(f"Qini Curve: {_contrast_title(contrast_name)}")
        ax.set_xlabel("Customers targeted (%)")
        ax.set_ylabel("Cumulative incremental conversions per 1,000 pairwise customers")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        figure_path = FIGURE_DIR / f"qini_{contrast_name.lower()}.png"
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)
        print("Qini plot saved:", figure_path)


def write_methodology_note():
    note = """THREE-ARM UPLIFT RANKING EVALUATION

Data:
- Validation split only; the untouched test split is never loaded.
- Men's Email vs No Email and Women's Email vs No Email are primary contrasts.
- Men's Email vs Women's Email is included as a secondary pairwise diagnostic.

Ranking score:
- For each contrast, rank customers by the model's predicted potential-outcome
  difference: predicted outcome under the treated campaign minus predicted
  outcome under the comparator campaign.

Pairwise randomization:
- The original experiment assigned customers equally to all three arms.
- Conditional on belonging to a selected pair, the assignment probability is
  0.5 for each of the two arms.

Cumulative gain:
- Each ranked customer contributes I(T=1)*Y/0.5 - I(T=0)*Y/0.5.
- The cumulative sum estimates incremental conversions from targeting the
  highest-scored fraction with the treated campaign instead of the comparator.

Metrics:
- AUUC: area under the cumulative gain curve, using the trapezoidal rule.
- Qini coefficient: area under the model curve minus the area under the straight
  random-targeting baseline connecting the origin to the full-population gain.
- Qini and AUUC are additionally scaled per 1,000 pairwise customers.
- Uplift@10%, @20%, and @30%: cumulative estimated incremental conversions
  divided by the number of customers targeted, expressed in percentage points.
- Equal prediction scores are treated as a tie block, with linear interpolation
  through the block. This avoids arbitrary ordering of tied LightGBM scores.

Uncertainty:
- 2,000 stratified bootstrap replications, resampling separately within each
  randomized arm while preserving the observed arm sample sizes.
- Logistic Regression and LightGBM use the same bootstrap samples, so model
  differences are paired.
- Intervals are conditional on the already-trained models; they do not include
  model-refitting uncertainty.

Interpretation:
- Positive Qini coefficient means the model ranking's cumulative gain area is
  above the random-targeting baseline under this definition.
- A positive Qini coefficient does not by itself prove positive policy value,
  profitable campaign selection, or superiority over a fixed campaign.
- Results are exploratory because conversion events are sparse.
"""
    METHODOLOGY_PATH.write_text(note, encoding="utf-8")


def run_uplift_ranking_evaluation():
    THREE_ARM_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    validation, actual_treatment, actual_outcome, predictions = (
        load_validation_and_predictions()
    )

    print("\nTHREE-ARM UPLIFT RANKING EVALUATION")
    print("Validation customers:", len(validation))
    print("Validation conversions:", int(actual_outcome.sum()))
    print("Bootstrap replications:", BOOTSTRAP_REPS)
    print("Conditional pairwise treatment propensity: 0.50")
    print("The untouched test split is not loaded.")

    all_summary_rows = []
    all_curve_rows = []
    all_comparison_rows = []
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)

    for contrast in CONTRASTS:
        print("\n" + "=" * 64)
        print("CONTRAST:", _contrast_title(contrast["contrast"]))
        print("Treated campaign:", ARM_LABELS[contrast["treated_arm"]])
        print("Comparison campaign:", ARM_LABELS[contrast["control_arm"]])
        print("=" * 64)

        summary_rows, curve_rows, comparison_rows, diagnostics = evaluate_contrast(
            contrast,
            actual_treatment,
            actual_outcome,
            predictions,
            rng,
        )

        print("Pairwise customers:", diagnostics["pairwise_customers"])
        print(
            "Treated:", diagnostics["treated_customers"],
            "customers,", diagnostics["treated_conversions"], "conversions",
        )
        print(
            "Comparator:", diagnostics["control_customers"],
            "customers,", diagnostics["control_conversions"], "conversions",
        )

        all_summary_rows.extend(summary_rows)
        all_curve_rows.extend(curve_rows)
        all_comparison_rows.extend(comparison_rows)

    summary_df = pd.DataFrame(all_summary_rows)
    curves_df = pd.DataFrame(all_curve_rows)
    comparisons_df = pd.DataFrame(all_comparison_rows)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    curves_df.to_csv(CURVES_PATH, index=False)
    comparisons_df.to_csv(COMPARISONS_PATH, index=False)
    save_qini_plots(curves_df)
    write_methodology_note()

    display_columns = [
        "contrast",
        "model",
        "pairwise_customers",
        "treated_conversions",
        "control_conversions",
        "qini_coefficient_per_1000_customers",
        "qini_coefficient_per_1000_customers_ci_lower",
        "qini_coefficient_per_1000_customers_ci_upper",
        "auuc_per_1000_customers",
        "overall_pairwise_uplift_pp",
        "uplift_at_10_percent_pp",
        "uplift_at_20_percent_pp",
        "uplift_at_30_percent_pp",
    ]

    print("\nUPLIFT RANKING SUMMARY")
    print(summary_df[display_columns].round(4).to_string(index=False))

    print("\nPAIRED MODEL COMPARISONS (Logistic Regression minus LightGBM)")
    print(
        comparisons_df[[
            "contrast",
            "metric",
            "difference_lr_minus_lightgbm",
            "ci_lower",
            "ci_upper",
        ]].round(4).to_string(index=False)
    )

    print("\nREPORTS SAVED")
    print("Summary:", SUMMARY_PATH)
    print("Curves:", CURVES_PATH)
    print("Model comparisons:", COMPARISONS_PATH)
    print("Methodology:", METHODOLOGY_PATH)
    print("Figures directory:", FIGURE_DIR)

    return summary_df, curves_df, comparisons_df


if __name__ == "__main__":
    run_uplift_ranking_evaluation()