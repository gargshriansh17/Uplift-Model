"""Nested cross-validation for the four two-arm uplift learners.

Outer folds estimate generalisation; every tunable parameter is selected using
only the corresponding outer-training fold. The untouched final test split is
never read here. Scores include scikit-uplift's normalized Qini/AUUC metrics.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

from sklift.metrics import qini_auc_score, uplift_auc_score

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, TARGET_COL, RANDOM_STATE
from src.data_preprocessing import FEATURE_COLUMNS, build_preprocessor
from src.models.s_learner import SLearner

TRAIN_PATH = PROCESSED_DATA_DIR / "train_data.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_data.csv"
OUTPUT_DIR = REPORTS_DIR / "nested_cv"
OUTER_FOLDS = 5
INNER_FOLDS = 3
MODEL_NAMES = ("Logistic T-Learner", "S-Learner", "X-Learner", "LightGBM T-Learner")


def load_development_data():
    frames = [pd.read_csv(path) for path in (TRAIN_PATH, VALIDATION_PATH)]
    df = pd.concat(frames, ignore_index=True)
    required = FEATURE_COLUMNS + [TARGET_COL, "treatment", "source_row_index"]
    missing = [col for col in required if col not in df]
    if missing:
        raise ValueError(f"Development data missing columns: {missing}")
    if df["source_row_index"].duplicated().any():
        raise ValueError("Duplicate customer IDs across train/validation.")
    if not df[TARGET_COL].isin([0, 1]).all() or not df["treatment"].isin([0, 1]).all():
        raise ValueError("Outcome and treatment must be binary.")
    return df.reset_index(drop=True)


def _logistic_pipeline(c):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(C=c, max_iter=2000, solver="lbfgs"))
    ])


def _fit_logistic_t(X, y, t, seed):
    models = {}
    for arm in (0, 1):
        mask = t == arm
        counts = np.bincount(y[mask], minlength=2)
        folds = min(INNER_FOLDS, int(counts.min()))
        if folds < 2:
            raise ValueError(f"Insufficient outcome counts in arm {arm} for inner CV.")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + arm)
        search = GridSearchCV(
            _logistic_pipeline(1.0),
            {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            scoring="neg_log_loss", cv=cv, n_jobs=1, refit=True
        )
        search.fit(X.loc[mask], y[mask])
        models[arm] = search.best_estimator_
    return models


def _fit_s_learner(X, y, t, seed):
    train = X.copy()
    train["treatment"] = t
    strata = np.char.add(np.char.add(t.astype(str), "_"), y.astype(str))
    counts = pd.Series(strata).value_counts()
    folds = min(INNER_FOLDS, int(counts.min()))
    cv = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(train, strata))
    estimator = SLearner(preprocessor=build_preprocessor(), max_iter=2000)
    search = GridSearchCV(
        estimator, {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        scoring="neg_log_loss", cv=cv, n_jobs=1, refit=True
    )
    search.fit(train, y)
    return search.best_estimator_


def _fit_x_learner(X, y, t, seed):
    base = _fit_logistic_t(X, y, t, seed)
    pre = build_preprocessor()
    x = pre.fit_transform(X)
    m1, m0 = base[1], base[0]
    tmask, cmask = t == 1, t == 0
    d1 = y[tmask] - m0.predict_proba(X.loc[tmask])[:, 1]
    d0 = m1.predict_proba(X.loc[cmask])[:, 1] - y[cmask]
    # Ridge alpha selection is inner CV, separately within each imputed-effect arm.
    def tune_effect(rows, target, offset):
        folds = min(INNER_FOLDS, len(target))
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + offset) if len(np.unique(target)) == 2 else folds
        search = GridSearchCV(
            Ridge(solver="lsqr"), {"alpha": [0.1, 1.0, 10.0, 100.0]},
            scoring="neg_mean_squared_error", cv=cv, n_jobs=1, refit=True
        )
        search.fit(rows, target)
        return search.best_estimator_
    tau1 = tune_effect(x[tmask], d1, 11)
    tau0 = tune_effect(x[cmask], d0, 17)
    return {"preprocessor": pre, "m1": m1, "m0": m0, "tau1": tau1, "tau0": tau0, "propensity": float(t.mean())}


def _predict_x(bundle, X):
    x = bundle["preprocessor"].transform(X)
    p1 = bundle["m1"].predict_proba(X)[:, 1]
    p0 = bundle["m0"].predict_proba(X)[:, 1]
    tau = bundle["propensity"] * bundle["tau0"].predict(x) + (1-bundle["propensity"]) * bundle["tau1"].predict(x)
    return p1, p0, tau


def _fit_lightgbm_t(X, y, t, seed):
    from src.models.lightgbm_t_learner import PARAM_GRID
    models = {}
    for arm in (0, 1):
        mask = t == arm
        counts = np.bincount(y[mask], minlength=2)
        folds = min(INNER_FOLDS, int(counts.min()))
        if folds < 2:
            raise ValueError(f"Insufficient outcome counts in arm {arm} for inner CV.")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + arm)
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LGBMClassifier(
                objective="binary", min_child_samples=50, reg_lambda=5.0,
                random_state=seed, n_jobs=1, verbosity=-1
            ))
        ])
        search = GridSearchCV(pipe, PARAM_GRID, scoring="neg_log_loss", cv=cv, n_jobs=1, refit=True)
        search.fit(X.loc[mask], y[mask])
        models[arm] = search.best_estimator_
    return models


def _score_model(name, fitted, X):
    if name in ("Logistic T-Learner", "LightGBM T-Learner"):
        p1 = fitted[1].predict_proba(X)[:, 1]
        p0 = fitted[0].predict_proba(X)[:, 1]
        return p1, p0, p1-p0
    if name == "S-Learner":
        treated, control = X.copy(), X.copy()
        treated["treatment"], control["treatment"] = 1, 0
        p1 = fitted.predict_proba(treated)[:, 1]
        p0 = fitted.predict_proba(control)[:, 1]
        return p1, p0, p1-p0
    return _predict_x(fitted, X)


def run_nested_cv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_development_data()
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    t = df["treatment"].astype(int).to_numpy()
    strata = np.char.add(np.char.add(t.astype(str), "_"), y.astype(str))
    min_count = int(pd.Series(strata).value_counts().min())
    if min_count < OUTER_FOLDS:
        raise ValueError("Not enough observations in a treatment/outcome stratum for outer CV.")

    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for fold, (train_idx, hold_idx) in enumerate(outer.split(X, strata), start=1):
        Xtr, Xho = X.iloc[train_idx], X.iloc[hold_idx]
        ytr, tho = y[train_idx], t[train_idx]
        yho, t_hold = y[hold_idx], t[hold_idx]
        seed = RANDOM_STATE + fold * 100
        fitted = {
            "Logistic T-Learner": _fit_logistic_t(Xtr, ytr, tho, seed),
            "S-Learner": _fit_s_learner(Xtr, ytr, tho, seed),
            "X-Learner": _fit_x_learner(Xtr, ytr, tho, seed),
            "LightGBM T-Learner": _fit_lightgbm_t(Xtr, ytr, tho, seed),
        }
        for name in MODEL_NAMES:
            p1, p0, uplift = _score_model(name, fitted[name], Xho)
            # scikit-uplift uses the same randomized binary contrast for every learner.
            records.append({
                "outer_fold": fold, "model": name,
                "n_holdout": len(hold_idx),
                "conversions": int(yho.sum()),
                "log_loss_treated": log_loss(yho[t_hold == 1], p1[t_hold == 1], labels=[0, 1]),
                "log_loss_control": log_loss(yho[t_hold == 0], p0[t_hold == 0], labels=[0, 1]),
                "sklift_qini_auc": qini_auc_score(yho, uplift, t_hold),
                "sklift_uplift_auc": uplift_auc_score(yho, uplift, t_hold),
            })
        print(f"Completed outer fold {fold}/{OUTER_FOLDS}")

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / "outer_fold_metrics.csv", index=False)
    summary = result.groupby("model").agg(
        folds=("outer_fold", "count"),
        mean_qini_auc=("sklift_qini_auc", "mean"),
        std_qini_auc=("sklift_qini_auc", "std"),
        mean_uplift_auc=("sklift_uplift_auc", "mean"),
        std_uplift_auc=("sklift_uplift_auc", "std"),
        mean_treated_log_loss=("log_loss_treated", "mean"),
        mean_control_log_loss=("log_loss_control", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "nested_cv_summary.csv", index=False)
    print(summary.round(5).to_string(index=False))
    return result, summary


if __name__ == "__main__":
    run_nested_cv()
