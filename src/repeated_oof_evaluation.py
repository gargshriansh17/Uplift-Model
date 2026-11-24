import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV
from sklearn.metrics import log_loss,brier_score_loss
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from src.config import(
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COL,
    RANDOM_STATE
)

from src.data_preprocessing import(
    FEATURE_COLUMNS,
    build_preprocessor
)

from src.models.s_learner import SLearner


TRAIN_PATH=PROCESSED_DATA_DIR/"train_data.csv"
VAL_PATH=PROCESSED_DATA_DIR/"validation_data.csv"

OUTPUT_DIR=REPORTS_DIR/"repeated_oof"

N_SPLITS=5
N_REPEATS=5

TIE_REPEATS=50

TREATMENT_C=0.01
CONTROL_C=0.1

S_LEARNER_C=0.1

MAX_ITER=2000

X_LEARNER=[0.1,1.0,10.0,100.0]

TARGET_PERCENTAGES=[0.10,0.20,0.30]

#LightGBM parameters
LIGHTGBM_TREATMENT_PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7
}

LIGHTGBM_CONTROL_PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.05,
    "max_depth": 2,
    "num_leaves": 3
}


def load_development_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:{TRAIN_PATH}"
        )

    if not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Validation data not found:{VAL_PATH}"
        )

    train_df=pd.read_csv(TRAIN_PATH)
    val_df=pd.read_csv(VAL_PATH)

    development_df=pd.concat(
        [train_df,val_df],
        ignore_index=True
    )

    required_cols=(
        FEATURE_COLUMNS+[
            TARGET_COL,
            "treatment",
            "source_row_index"
        ]
    )

    missing_cols=[
        col for col in required_cols
        if col not in development_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing development columns:{missing_cols}"
        )

    if development_df[required_cols].isna().any().any():
        raise ValueError(
            "Missing columns found in development data"
        )

    if development_df["source_row_index"].duplicated().any():
        raise ValueError(
            "Duplicate source_row_index found in development data"
        )

    development_df=development_df.sort_values(
        "source_row_index"
    ).reset_index(drop=True)

    X=development_df[FEATURE_COLUMNS].copy()

    y=development_df[TARGET_COL].astype(int).to_numpy()

    treatment=development_df["treatment"].astype(int).to_numpy()

    if not np.isin(y,[0,1]).all():
        raise ValueError("Conversion must contain only 0 and 1")

    if not np.isin(treatment,[0,1]).all():
        raise ValueError("Treatment must contain only 0 and 1")

    return development_df,X,y,treatment

def build_lightgbm_pipeline(params,seed):
    model_params={
        **params,
        "objective":"binary",
        "min_child_samples":50,
        "reg_lambda":5.0,
        "random_state":seed,
        "n_jobs":1,
        "verbosity":-1
    }

    return Pipeline([
        ("preprocessor",build_preprocessor()),
        ("model",LGBMClassifier(**model_params))
    ])


def fit_predict_fold(
        X_train,
        y_train,
        treatment_train,
        X_holdout,
        seed
):

    predictions={}

    treatment_mask=treatment_train==1
    control_mask=treatment_train==0

    #T-LEARNER
    preprocessor=build_preprocessor()

    X_train_processed=preprocessor.fit_transform(X_train)

    X_holdout_processed=preprocessor.transform(X_holdout)

    treatment_model=LogisticRegression(
        C=TREATMENT_C,
        max_iter=MAX_ITER,
        solver="lbfgs",
        class_weight=None
    )

    control_model=LogisticRegression(
        C=CONTROL_C,
        max_iter=MAX_ITER,
        solver="lbfgs",
        class_weight=None
    )

    treatment_model.fit(
        X_train_processed[treatment_mask],
        y_train[treatment_mask]
    )

    control_model.fit(
        X_train_processed[control_mask],
        y_train[control_mask]
    )

    logistic_p1=treatment_model.predict_proba(X_holdout_processed)[:,1]

    logistic_p0=control_model.predict_proba(X_holdout_processed)[:,1]

    predictions["Logistic T-Learner"]={
        "p1":logistic_p1,
        "p0":logistic_p0,
        "uplift":logistic_p1-logistic_p0
    }

    #X-Learner

    # Imputed treatment effects for treated customers.
    # The control model predicts their counterfactual
    # outcome without email.
    control_prediction_for_treated=(
        control_model.predict_proba(
            X_train_processed[treatment_mask]
        )[:,1]
    )

    imputed_effect_treated=(
        y_train[treatment_mask]
        -
        control_prediction_for_treated
    )

    # Imputed treatment effects for control customers.
    # The treatment model predicts their counterfactual
    # outcome with email.

    treatment_prediction_for_control=(
        treatment_model.predict_proba(
            X_train_processed[control_mask]
        )[:,1]
    )

    imputed_effect_control=(
        treatment_prediction_for_control
        -
        y_train[control_mask]
    )


    #Training Second stage models independently
    tau1_model,alpha1=fit_effect_model(
        X_train_processed[treatment_mask],
        imputed_effect_treated,
        seed
    )

    tau0_model,alpha2=fit_effect_model(
        X_train_processed[control_mask],
        imputed_effect_control,
        seed+1
    )

    tau1_holdout=tau1_model.predict(X_holdout_processed)
    tau0_holdout=tau0_model.predict(X_holdout_processed)

    propensity=float(np.mean(treatment_train))

    x_uplift=(
        propensity*tau0_holdout
        +
        (1-propensity)*tau1_holdout
    )

    predictions["X-Learner"]={
        "p1":logistic_p1,
        "p0":logistic_p0,
        "uplift":x_uplift
    }

    #S-LEARNER
    X_train_s=X_train.copy()
    X_train_s["treatment"]=treatment_train

    s_learner=SLearner(
        preprocessor=build_preprocessor(),
        C=S_LEARNER_C,
        max_iter=MAX_ITER
    )

    s_learner.fit(
        X_train_s,
        y_train
    )

    X_holdout_treated=X_holdout.copy()
    X_holdout_treated["treatment"]=1

    X_holdout_control=X_holdout.copy()
    X_holdout_control["treatment"]=0

    s_p1=s_learner.predict_proba(X_holdout_treated)[:,1]

    s_p0=s_learner.predict_proba(X_holdout_control)[:,1]

    predictions["S-Learner"]={
        "p1":s_p1,
        "p0":s_p0,
        "uplift":s_p1-s_p0
    }


    #LightGBM
    treatment_indices=np.flatnonzero(treatment_mask)
    control_indices=np.flatnonzero(control_mask)

    lightgbm_treatment=build_lightgbm_pipeline(
        LIGHTGBM_TREATMENT_PARAMS,
        seed
    )

    lightgbm_control=build_lightgbm_pipeline(
        LIGHTGBM_CONTROL_PARAMS,
        seed
    )

    lightgbm_treatment.fit(
        X_train.iloc[treatment_indices],
        y_train[treatment_mask]
    )

    lightgbm_control.fit(
        X_train.iloc[control_indices],
        y_train[control_mask]
    )

    lightgbm_p1=lightgbm_treatment.predict_proba(X_holdout)[:,1]
    lightgbm_p0=lightgbm_control.predict_proba(X_holdout)[:,1]

    predictions["LightGBM T-Learner"]={
        "p1":lightgbm_p1,
        "p0":lightgbm_p0,
        "uplift":lightgbm_p1-lightgbm_p0
    }

    return predictions

def fit_effect_model(X,imputed_effect,seed):
    inner_cv=KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    grid_search=GridSearchCV(
        estimator=Ridge(solver="lsqr"),
        param_grid={"alpha":X_LEARNER},
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
        error_score="raise"
    )

    grid_search.fit(X,imputed_effect)

    return(
        grid_search.best_estimator_,
        float(grid_search.best_params_["alpha"])
    )

def calculate_probability_metrics(df):
    treatment_mask=df["actual_treatment"].to_numpy()==1
    control_mask=df["actual_treatment"].to_numpy()==0

    y=df["actual_conversion"].to_numpy()

    p1=df["predicted_purchase_if_treated"].to_numpy()
    p0=df["predicted_purchase_if_control"].to_numpy()

    return {
        "treatment_log_loss":log_loss(
            y[treatment_mask],
            p1[treatment_mask],
            labels=[0,1]
        ),

        "control_log_loss":log_loss(
            y[control_mask],
            p0[control_mask],
            labels=[0,1]
        ),

        "treatment_brier":brier_score_loss(
            y[treatment_mask],
            p1[treatment_mask]
        ),

        "control_brier":brier_score_loss(
            y[control_mask],
            p0[control_mask]
        )
    }


def calculate_group_uplift(df):
    treatment_group=df[df["actual_treatment"]==1]
    control_group=df[df["actual_treatment"]==0]

    if len(treatment_group)==0 or len(control_group)==0:
        raise ValueError(
            "Both treatment and control are required"
        )

    treatment_rate=treatment_group["actual_conversion"].mean()
    control_rate=control_group["actual_conversion"].mean()

    return (treatment_rate-control_rate)*100


#QINI and UPLIFT AT K ONE TIE ORDER
def calculate_rank_metrics_once(df,row_order):

    working_df=df.iloc[row_order].reset_index(drop=True)
    ranked_df=working_df.sort_values(
        "predicted_uplift",
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    total_customers=len(ranked_df)

    treatment=ranked_df["actual_treatment"].to_numpy()

    outcome=ranked_df["actual_conversion"].to_numpy()

    total_treatment=int(np.sum(treatment==1))
    total_control=int(np.sum(treatment==0))

    results={}

    for percentage in TARGET_PERCENTAGES:
        k=int(np.ceil(total_customers*percentage))

        selected=ranked_df.iloc[:k]

        results[
            f"uplift_at_{int(percentage*100)}_pp"
        ]=calculate_group_uplift(selected)


    #Qini Curve
    treatment_conversion=(
        (treatment==1) & (outcome==1)
    ).astype(int)

    control_conversion=(
        (treatment==0) & (outcome==1)
    ).astype(int)

    cumulative_treatment=np.cumsum(treatment_conversion)
    cumulative_control=np.cumsum(control_conversion)

    qini_gain=(
        cumulative_treatment/total_treatment
        -
        cumulative_control/total_control
    )*100

    target_fraction=(
        np.arange(1,total_customers+1)/total_customers
    )

    observed_ate=(
        outcome[treatment==1].mean()
        -
        outcome[treatment==0].mean()
    )*100

    if not np.isclose(
        qini_gain[-1],
        observed_ate,
        atol=1e-10
    ):
        raise ValueError(
            "Qini endpoint does not match observed ATE"
        )

    x=np.insert(target_fraction,0,0.0)
    gain=np.insert(qini_gain,0,0.0)

    if hasattr(np,"trapezoid"):
        integrate=np.trapezoid

    else:
        from scipy.integrate import trapezoid
        integrate=trapezoid

    model_auuc=integrate(gain,x)

    random_auuc=0.5*observed_ate

    results["model_auuc"]=model_auuc

    results["random_auuc"]=random_auuc

    results["qini_coefficient"]=(model_auuc-random_auuc)

    results["observed_ate_pp"]=observed_ate

    return results


#AVG OVER RANDOMIZED TIE ORDERS
def calculate_rank_metrics(df,tie_orders):
    tie_results=[]
    for order in tie_orders:
        result=calculate_rank_metrics_once(df,order)

        tie_results.append(result)

    tie_df=pd.DataFrame(tie_results)

    summary={}

    for col in tie_df.columns:
        summary[col]=float(tie_df[col].mean())

        summary[f"{col}_tie_std"]=float(tie_df[col].std(ddof=1))

    return summary

#BUILD OFF PREDICTION DATAFRAME
def build_oof_dataframe(
        development_df,
        treatment,
        y,
        oof_predictions
):

    results_df=pd.DataFrame({
        "source_row_index":(
            development_df["source_row_index"].to_numpy()
        ),

        "actual_treatment":treatment,

        "actual_conversion":y
    })

    for model_name,prediction in oof_predictions.items():

        prefix=(
            model_name.lower().replace(" ","_").replace("-","_")
        )

        results_df[
            f"{prefix}_p1"
        ] = prediction["p1"]

        results_df[
            f"{prefix}_p0"
        ] = prediction["p0"]

        results_df[
            f"{prefix}_uplift"
        ] = prediction["uplift"]

    return results_df


def evaluate_one_repeat(
        oof_df,
        repeat_numbers,
        tie_orders
):

    model_names=[
        "Logistic T-Learner",
        "S-Learner",
        "LightGBM T-Learner",
        "X-Learner"
    ]

    metric_records=[]

    for model_name in model_names:
        prefix=(
            model_name.lower().replace(" ","_").replace("-","_")
        )

        model_df=pd.DataFrame({
            "actual_treatment": oof_df["actual_treatment"],

            "actual_conversion": oof_df["actual_conversion"],

            "predicted_purchase_if_treated": oof_df[
                f"{prefix}_p1"
            ],

            "predicted_purchase_if_control": oof_df[
                f"{prefix}_p0"
            ],

            "predicted_uplift": oof_df[
                f"{prefix}_uplift"
            ]
        })

        probab_metrics=calculate_probability_metrics(model_df)

        rank_metrics=calculate_rank_metrics(model_df,tie_orders)

        record={
            "repeat": repeat_numbers,

            "model": model_name,

            **probab_metrics,

            **rank_metrics
        }

        metric_records.append(record)

    return metric_records

def run_repeated_oof_evaluation():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    development_df, X, y, treatment = (
        load_development_data()
    )

    total_customers = len(X)

    # Stratify by treatment-outcome combination.
    strata = (
        treatment.astype(str)
        + "_"
        + y.astype(str)
    )

    strata_counts = pd.Series(strata).value_counts()

    if strata_counts.min() < N_SPLITS:
        raise ValueError(
            "A treatment-outcome stratum has fewer "
            "observations than the number of folds."
        )

    print("\nREPEATED OUT-OF-FOLD EVALUATION")

    print("\nDEVELOPMENT DATA")

    print("Customers:", total_customers)

    print("Conversions:", int(y.sum()))

    print("Treatment customers:", int(np.sum(treatment == 1)))

    print("Control customers:", int(np.sum(treatment == 0)))

    print("\nTREATMENT-OUTCOME STRATA")

    print(
        pd.crosstab(
            treatment,
            y,
            rownames=["Treatment"],
            colnames=["Conversion"]
        )
    )

    print("\nCROSS-VALIDATION CONFIGURATION")

    print("Folds:", N_SPLITS)

    print("Repeats:", N_REPEATS)

    print("Total fold evaluations:", N_SPLITS * N_REPEATS)

    print("Tie-order repetitions:", TIE_REPEATS)

    metric_records = []

    # Each repeat produces one OOF prediction per customer.
    for repeat_number in range(1, N_REPEATS + 1):

        seed = RANDOM_STATE + repeat_number * 100

        print("\n" + "=" * 60)

        print(
            f"REPEAT {repeat_number}/{N_REPEATS} "
            f"| Random seed: {seed}"
        )

        print("=" * 60)

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed
        )

        oof_predictions ={}

        model_names=[
            "Logistic T-Learner",
            "S-Learner",
            "LightGBM T-Learner",
            "X-Learner"
        ]

        for model_name in model_names:
            oof_predictions[model_name]={
                "p1":np.full(total_customers,np.nan),
                "p0":np.full(total_customers,np.nan),
                "uplift":np.full(total_customers,np.nan)
            }


        # ------------------------------------------
        # FIVE CV FOLDS
        # ------------------------------------------

        for fold_number, (train_idx, holdout_idx) in enumerate(
            cv.split(X, strata),
            start=1
        ):

            print(
                f"\nRepeat {repeat_number} - "
                f"Fold {fold_number}/{N_SPLITS}"
            )

            X_train_fold = X.iloc[train_idx].copy()

            X_holdout = X.iloc[holdout_idx].copy()

            y_train_fold = y[train_idx]

            treatment_train_fold = treatment[train_idx]

            fold_predictions = fit_predict_fold(
                X_train_fold,
                y_train_fold,
                treatment_train_fold,
                X_holdout,
                seed
            )

            # Store predictions only for held-out customers.
            for model_name, prediction in fold_predictions.items():

                oof_predictions[model_name]["p1"][
                    holdout_idx
                ] = prediction["p1"]

                oof_predictions[model_name]["p0"][
                    holdout_idx
                ] = prediction["p0"]

                oof_predictions[model_name]["uplift"][
                    holdout_idx
                ] = prediction["uplift"]

        # ------------------------------------------
        # VERIFY OOF COMPLETENESS
        # ------------------------------------------

        for model_name, prediction in oof_predictions.items():

            if not (
                np.isfinite(prediction["p1"]).all()
                and np.isfinite(prediction["p0"]).all()
                and np.isfinite(prediction["uplift"]).all()
            ):

                raise ValueError(
                    f"Incomplete OOF predictions: {model_name}"
                )

            if not (
                np.all((prediction["p1"] >= 0) & (prediction["p1"] <= 1))
                and np.all((prediction["p0"] >= 0) & (prediction["p0"] <= 1))
            ):

                raise ValueError(
                    f"Invalid potential-outcome probabilities: {model_name}"
                )

        # ------------------------------------------
        # SAVE OOF PREDICTIONS
        # ------------------------------------------

        oof_df = build_oof_dataframe(
            development_df,
            treatment,
            y,
            oof_predictions
        )

        oof_path = (
            OUTPUT_DIR
            / f"oof_predictions_repeat_{repeat_number}.csv"
        )

        oof_df.to_csv(
            oof_path,
            index=False
        )

        print(
            "\nOOF predictions saved:",
            oof_path
        )

        # ------------------------------------------
        # COMMON RANDOMIZED TIE ORDERS
        # ------------------------------------------

        tie_rng = np.random.default_rng(
            seed + 999
        )

        tie_orders = [
            tie_rng.permutation(total_customers)
            for _ in range(TIE_REPEATS)
        ]

        # Use identical ordering permutations for all
        # candidate models in this repeat.
        repeat_records = evaluate_one_repeat(
            oof_df,
            repeat_number,
            tie_orders
        )

        metric_records.extend(repeat_records)

    # ==================================================
    # SAVE REPEAT-LEVEL METRICS
    # ==================================================

    metrics_by_repeat = pd.DataFrame(
        metric_records
    )

    metrics_by_repeat.to_csv(
        OUTPUT_DIR / "metrics_by_repeat.csv",
        index=False
    )

    # ==================================================
    # SUMMARIZE PERFORMANCE ACROSS FIVE REPEATS
    # ==================================================

    metric_columns = [
        "treatment_log_loss",
        "control_log_loss",
        "treatment_brier",
        "control_brier",
        "qini_coefficient",
        "model_auuc",
        "random_auuc",
        "uplift_at_10_pp",
        "uplift_at_20_pp",
        "uplift_at_30_pp"
    ]

    summary_df = (
        metrics_by_repeat
        .groupby("model")[metric_columns]
        .agg(["mean", "std", "min", "max"])
    )

    summary_df.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary_df.columns
    ]

    summary_df = summary_df.reset_index()

    summary_df.to_csv(
        OUTPUT_DIR / "metrics_summary.csv",
        index=False
    )

    # ==================================================
    # FINAL OUTPUT
    # ==================================================

    print("\n" + "=" * 75)

    print("REPEATED OOF MODEL COMPARISON")

    print("=" * 75)

    print(
        summary_df.round(6).to_string(index=False)
    )

    print("\nREPEAT-LEVEL METRICS")

    print(
        metrics_by_repeat[
            [
                "repeat",
                "model",
                "qini_coefficient",
                "uplift_at_10_pp",
                "uplift_at_20_pp",
                "uplift_at_30_pp"
            ]
        ].round(5).to_string(index=False)
    )

    print("\nREPEATED OOF EVALUATION COMPLETED")

    print(
        "Metrics by repeat:",
        OUTPUT_DIR / "metrics_by_repeat.csv"
    )

    print(
        "Metrics summary:",
        OUTPUT_DIR / "metrics_summary.csv"
    )

    print(
        "OOF prediction files:",
        OUTPUT_DIR
    )

    return (
        metrics_by_repeat,
        summary_df
    )


if __name__ == "__main__":

    run_repeated_oof_evaluation()



