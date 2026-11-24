"""Unit tests for the evaluation metrics and the data-split safety checks.

Run from the project root:
    python -m pytest tests
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import final_test_evaluation as fte
from src.data_preprocessing import FEATURE_COLUMNS
from src.config import TARGET_COL


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_predictions(n=20000, seed=0):
    """Randomised experiment where only 'responders' (half the customers)
    are helped by treatment: +20 percentage points of conversion."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, size=n)
    responder = rng.integers(0, 2, size=n)

    p_convert = 0.05 + 0.20 * treatment * responder
    conversion = (rng.random(n) < p_convert).astype(int)

    return pd.DataFrame({
        "actual_treatment": treatment,
        "actual_conversion": conversion,
        "responder": responder,
    })


def qini_for_scores(df, scores):
    scored = df.copy()
    scored["predicted_uplift"] = scores
    return fte.metrics_from_predictions(scored)["qini_coefficient"]


# ----------------------------------------------------------------------
# Metric tests
# ----------------------------------------------------------------------

def test_good_ranking_beats_random_beats_reversed():
    df = make_predictions()
    rng = np.random.default_rng(1)

    perfect = qini_for_scores(df, df["responder"].to_numpy() + rng.random(len(df)) * 1e-6)
    random_ = qini_for_scores(df, rng.random(len(df)))
    reversed_ = qini_for_scores(df, -df["responder"].to_numpy() + rng.random(len(df)) * 1e-6)

    assert perfect > 0
    assert reversed_ < 0
    assert perfect > random_ > reversed_


def test_uplift_at_k_matches_hand_calculation():
    # 10 customers, best-ranked 2 are one treated (converts) and one control (does not)
    df = pd.DataFrame({
        "actual_treatment":  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "actual_conversion": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "predicted_uplift":  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    })
    result = fte.metrics_from_predictions(df)

    # top 20% = first two rows: treated rate 100%, control rate 0% -> 100 pp
    assert abs(result["uplift_at_20_pp"] - 100.0) < 1e-9


def test_qini_curve_ends_at_observed_effect():
    df = make_predictions(n=5000, seed=3)
    df["predicted_uplift"] = np.random.default_rng(4).random(len(df))

    qini_df, observed_ate_pp = fte.calculate_qini_curve(df)

    assert np.isclose(qini_df["qini_gain_pp"].iloc[-1], observed_ate_pp)
    assert qini_df["target_fraction"].iloc[0] == 0.0
    assert qini_df["target_fraction"].iloc[-1] == 1.0


def test_bootstrap_returns_requested_repeats_and_sensible_interval():
    df = make_predictions(n=3000, seed=5)
    df["predicted_uplift"] = df["responder"] + np.random.default_rng(6).random(len(df)) * 1e-6

    point = fte.metrics_from_predictions(df)
    boot = fte.stratified_bootstrap(df, repeats=100)
    summary = pd.DataFrame(fte.summarise_bootstrap(boot, point, "test"))

    assert len(boot) == 100
    row = summary[summary["metric"] == "qini_coefficient"].iloc[0]
    assert row["ci_lower"] <= row["point_estimate"] <= row["ci_upper"]


# ----------------------------------------------------------------------
# Split-safety tests
# ----------------------------------------------------------------------

def _split_frame(ids):
    data = {col: [0] * len(ids) for col in FEATURE_COLUMNS}
    data.update({
        TARGET_COL: [0] * len(ids),
        "treatment": [0] * len(ids),
        "source_row_index": list(ids),
    })
    return pd.DataFrame(data)


def _run_load_splits(train_ids, val_ids, test_ids):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = {
            "TRAIN_PATH": tmp / "train.csv",
            "VALIDATION_PATH": tmp / "val.csv",
            "TEST_PATH": tmp / "test.csv",
        }
        _split_frame(train_ids).to_csv(paths["TRAIN_PATH"], index=False)
        _split_frame(val_ids).to_csv(paths["VALIDATION_PATH"], index=False)
        _split_frame(test_ids).to_csv(paths["TEST_PATH"], index=False)

        with patch.multiple(fte, **paths):
            return fte.load_splits()


def test_load_splits_accepts_disjoint_splits():
    train, val, test = _run_load_splits([1, 2, 3], [4, 5], [6, 7])
    assert len(train) == 3 and len(val) == 2 and len(test) == 2


def test_load_splits_rejects_test_leakage():
    for train_ids, val_ids in (([1, 2, 6], [4, 5]), ([1, 2], [4, 6])):
        try:
            _run_load_splits(train_ids, val_ids, [6, 7])
        except ValueError:
            continue
        raise AssertionError("Leakage between test and train/validation was not caught")
