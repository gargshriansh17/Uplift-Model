"""Reproducible access to the public Hillstrom dataset via scikit-uplift.

The raw source is cached by scikit-uplift. A normalized local CSV is also written
to data/raw/hillstrom.csv so the rest of the original pipeline can keep using
the same file contract.
"""
from __future__ import annotations

import pandas as pd

from src.config import DATA_PATH, RAW_DATA_DIR, SEGMENT_COL, TARGET_COL

REQUIRED_FEATURES = [
    "recency", "history", "mens", "womens", "newbie",
    "history_segment", "zip_code", "channel",
]
TARGET_COLUMNS = ["visit", "conversion", "spend"]


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in frame.columns
    ]
    return frame


def load_hillstrom_frame(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch Hillstrom using scikit-uplift and return the canonical dataframe.

    scikit-uplift caches the downloaded source under data/raw/scikit_uplift.
    Calling this function repeatedly therefore does not repeatedly download
    the dataset. force_refresh is reserved for an explicit manual refresh.
    """
    try:
        from sklift.datasets import fetch_hillstrom
    except ImportError as exc:
        raise ImportError(
            "scikit-uplift is required to fetch Hillstrom. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = RAW_DATA_DIR / "scikit_uplift"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_hillstrom(
        target_col="all",
        data_home=str(cache_dir),
        download_if_missing=True,
        return_X_y_t=False,
    )

    features = _normalise_columns(pd.DataFrame(dataset.data))
    targets = _normalise_columns(pd.DataFrame(dataset.target))
    treatment = pd.Series(dataset.treatment, name=SEGMENT_COL).reset_index(drop=True)

    missing_features = [name for name in REQUIRED_FEATURES if name not in features.columns]
    missing_targets = [name for name in TARGET_COLUMNS if name not in targets.columns]
    if missing_features or missing_targets:
        raise ValueError(
            "Unexpected scikit-uplift Hillstrom schema. "
            f"Missing features={missing_features}; missing targets={missing_targets}; "
            f"features={features.columns.tolist()}; targets={targets.columns.tolist()}"
        )

    if len(features) != len(targets) or len(features) != len(treatment):
        raise ValueError("Hillstrom features, targets and treatment have different lengths.")

    frame = pd.concat(
        [
            features[REQUIRED_FEATURES].reset_index(drop=True),
            targets[TARGET_COLUMNS].reset_index(drop=True),
            treatment,
        ],
        axis=1,
    )

    if frame[[SEGMENT_COL, TARGET_COL]].isna().any().any():
        raise ValueError("Hillstrom contains missing treatment or conversion values.")
    if not frame[TARGET_COL].isin([0, 1]).all():
        raise ValueError("Hillstrom conversion target must be binary.")

    # Preserve the canonical local path expected by existing project stages.
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)

    return frame


if __name__ == "__main__":
    result = load_hillstrom_frame(force_refresh=True)
    print(f"Loaded Hillstrom through scikit-uplift: {result.shape}")
    print(result[SEGMENT_COL].value_counts().to_string())
