"""One-time held-out test evaluation for all three Hillstrom campaign arms.

Models are selected on development data only, refit on train + validation,
and scored once on the untouched three-arm test partition. Policy value is
reported with IPW and doubly-robust estimators under the experiment's 1/3
randomization propensity. No test-set tuning is performed.
"""
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklift.metrics import qini_auc_score, uplift_auc_score

from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, TARGET_COL, RANDOM_STATE
from src.data_preprocessing import FEATURE_COLUMNS, build_preprocessor

DATA_DIR = PROCESSED_DATA_DIR / "three_arm"
ART_DIR = ARTIFACTS_DIR / "three_arm"
OUT_DIR = REPORTS_DIR / "three_arm" / "final_test"
SPLITS = {name: DATA_DIR / f"{name}_data.csv" for name in ("train", "validation", "test")}
ARM_LABELS = {0: "No Email", 1: "Mens E-Mail", 2: "Womens E-Mail"}
PROPENSITY = np.array([1/3, 1/3, 1/3], dtype=float)
BOOTSTRAP_REPEATS = 2000
SEED = RANDOM_STATE + 9000


def _load_splits():
    frames = {name: pd.read_csv(path) for name, path in SPLITS.items()}
    required = FEATURE_COLUMNS + [TARGET_COL, "treatment", "source_row_index"]
    for name, frame in frames.items():
        missing = [col for col in required if col not in frame]
        if missing:
            raise ValueError(f"{name} split missing columns: {missing}")
        if frame["source_row_index"].duplicated().any():
            raise ValueError(f"Duplicate source IDs in {name} split.")
        if not frame[TARGET_COL].isin([0, 1]).all():
            raise ValueError(f"Non-binary outcome in {name} split.")
        if not frame["treatment"].isin([0, 1, 2]).all():
            raise ValueError(f"Invalid treatment arms in {name} split.")
    ids = {name: set(frame["source_row_index"]) for name, frame in frames.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise ValueError("Three-arm train, validation and test splits overlap.")
    for name, frame in frames.items():
        counts = pd.crosstab(frame["treatment"], frame[TARGET_COL]).reindex(
            index=[0, 1, 2], columns=[0, 1], fill_value=0
        )
        if (counts == 0).any().any():
            raise ValueError(f"{name} split lacks an arm/outcome combination: {counts}")
    return frames


def _fit_logistic(fit_df):
    bundle = {}
    for arm, c_value in ((0, 0.1), (1, 0.01), (2, 0.01)):
        rows = fit_df[fit_df["treatment"] == arm]
        bundle[arm] = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs"))
        ]).fit(rows[FEATURE_COLUMNS], rows[TARGET_COL].astype(int))
    return bundle


def _fit_lightgbm(fit_df):
    saved_path = ART_DIR / "lightgbm_t_learner_bundle.joblib"
    if not saved_path.exists():
        raise FileNotFoundError(
            "Missing validation-selected three-arm LightGBM bundle. "
            "Run python -m src.models.three_arm_lightgbm_t_learner first."
        )
    saved = joblib.load(saved_path)
    arm_models = saved.get("arm_models")
    if arm_models is None:
        raise ValueError("Saved three-arm LightGBM bundle has no arm_models.")
    fitted = {}
    for arm in (0, 1, 2):
        rows = fit_df[fit_df["treatment"] == arm]
        fitted[arm] = clone(arm_models[arm]).fit(
            rows[FEATURE_COLUMNS], rows[TARGET_COL].astype(int)
        )
    return fitted


def _predict(models, test):
    X = test[FEATURE_COLUMNS]
    return np.column_stack([models[arm].predict_proba(X)[:, 1] for arm in (0, 1, 2)])


def _policy_contributions(t, y, mu, actions):
    rows = np.arange(len(y))
    matched = t == actions
    propensity = PROPENSITY[actions]
    ipw = matched * y / propensity
    dr = mu[rows, actions] + matched * (y - mu[rows, t]) / propensity
    return ipw, dr


def _evaluate_policy_set(t, y, mu):
    learned = np.argmax(mu, axis=1)
    policies = {
        "Always No Email": np.zeros(len(y), dtype=int),
        "Always Mens Email": np.ones(len(y), dtype=int),
        "Always Womens Email": np.full(len(y), 2, dtype=int),
        "Logistic learned policy": None,
        "LightGBM learned policy": learned,
    }
    rows = []
    for name, actions in policies.items():
        if name == "Logistic learned policy":
            continue
        ipw, dr = _policy_contributions(t, y, mu, actions)
        rows.append({"policy": name, "ipw_value": float(ipw.mean()), "dr_value": float(dr.mean())})
    return learned, pd.DataFrame(rows)


def _bootstrap_policy_values(t, y, mu, actions_by_model):
    rng = np.random.default_rng(SEED)
    names = list(actions_by_model)
    contribution = {}
    for model_name, policies in actions_by_model.items():
        for policy_name, actions in policies.items():
            ipw, dr = _policy_contributions(t, y, mu[model_name], actions)
            contribution[(model_name, policy_name, "IPW")] = ipw
            contribution[(model_name, policy_name, "DR")] = dr
    matrix = np.column_stack(list(contribution.values()))
    samples = np.empty((BOOTSTRAP_REPEATS, matrix.shape[1]))
    for b in range(BOOTSTRAP_REPEATS):
        idx = rng.integers(0, len(y), size=len(y))
        samples[b] = matrix[idx].mean(axis=0)
    keys = list(contribution)
    records = []
    for j, key in enumerate(keys):
        records.append({
            "model": key[0], "policy": key[1], "estimator": key[2],
            "estimate": float(matrix[:, j].mean()),
            "ci_lower": float(np.quantile(samples[:, j], 0.025)),
            "ci_upper": float(np.quantile(samples[:, j], 0.975)),
        })
    return pd.DataFrame(records)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = _load_splits()
    fit_df = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
    test = frames["test"]
    t = test["treatment"].astype(int).to_numpy()
    y = test[TARGET_COL].astype(int).to_numpy()

    models = {
        "Logistic T-Learner": _fit_logistic(fit_df),
        "LightGBM T-Learner": _fit_lightgbm(fit_df),
    }
    metrics = []
    predictions = {}
    policy_actions = {}
    potential_outcomes = {}
    for name, model in models.items():
        mu = _predict(model, test)
        potential_outcomes[name] = mu
        predictions[name] = mu
        policy_actions[name] = {
            "Always No Email": np.zeros(len(y), dtype=int),
            "Always Mens Email": np.ones(len(y), dtype=int),
            "Always Womens Email": np.full(len(y), 2, dtype=int),
            "Learned argmax policy": np.argmax(mu, axis=1),
        }
        for arm in (1, 2):
            mask = np.isin(t, [0, arm])
            binary_t = (t[mask] == arm).astype(int)
            uplift = mu[mask, arm] - mu[mask, 0]
            metrics.append({
                "model": name, "contrast": f"{ARM_LABELS[arm]} vs No Email",
                "n": int(mask.sum()), "conversions": int(y[mask].sum()),
                "sklift_qini_auc": float(qini_auc_score(y[mask], uplift, binary_t)),
                "sklift_uplift_auc": float(uplift_auc_score(y[mask], uplift, binary_t)),
            })
        out = test[["source_row_index"]].copy()
        out["actual_treatment"] = t
        out["actual_conversion"] = y
        for arm in (0, 1, 2):
            out[f"predicted_conversion_arm_{arm}"] = mu[:, arm]
        out["predicted_best_arm"] = np.argmax(mu, axis=1)
        out.to_csv(OUT_DIR / (name.lower().replace(" ", "_") + "_predictions.csv"), index=False)

    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(OUT_DIR / "three_arm_test_uplift_metrics.csv", index=False)

    # The learned policy's nuisance predictions are supplied by each model.
    policy_records = []
    for model_name, mu in potential_outcomes.items():
        for policy_name, actions in policy_actions[model_name].items():
            ipw, dr = _policy_contributions(t, y, mu, actions)
            policy_records.append({
                "model": model_name, "policy": policy_name,
                "ipw_value": float(ipw.mean()), "dr_value": float(dr.mean())
            })
    policy_df = pd.DataFrame(policy_records)
    policy_df.to_csv(OUT_DIR / "three_arm_test_policy_values.csv", index=False)

    # Bootstrap CIs use customer-level resampling and preserve pairing across
    # all policies within a model. Randomized arm assignment probabilities are 1/3.
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for model_name, mu in potential_outcomes.items():
        names = list(policy_actions[model_name])
        contributions = {}
        for policy_name, actions in policy_actions[model_name].items():
            contributions[policy_name] = _policy_contributions(t, y, mu, actions)
        boot = {name: {"IPW": [], "DR": []} for name in names}
        arm_positions = {arm: np.flatnonzero(t == arm) for arm in (0, 1, 2)}
        for _ in range(BOOTSTRAP_REPEATS):
            # Stratify by randomized assignment arm to preserve the experiment design.
            idx = np.concatenate([
                rng.choice(arm_positions[arm], size=len(arm_positions[arm]), replace=True)
                for arm in (0, 1, 2)
            ])
            for name, (ipw, dr) in contributions.items():
                boot[name]["IPW"].append(float(ipw[idx].mean()))
                boot[name]["DR"].append(float(dr[idx].mean()))
        for name in names:
            for estimator in ("IPW", "DR"):
                values = np.asarray(boot[name][estimator])
                point = float(contributions[name][0 if estimator == "IPW" else 1].mean())
                boot_rows.append({
                    "model": model_name, "policy": name, "estimator": estimator,
                    "estimate": point,
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_repetitions": BOOTSTRAP_REPEATS,
                })
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "three_arm_test_policy_bootstrap.csv", index=False)

    print("THREE-ARM FINAL TEST")
    print("Fit customers:", len(fit_df), "| Test customers:", len(test))
    print(metric_df.round(5).to_string(index=False))
    print(policy_df.round(5).to_string(index=False))
    print("Reports:", OUT_DIR)
    print("Test set scored. Do not tune models based on these results.")
    return metric_df, policy_df


if __name__ == "__main__":
    run()
