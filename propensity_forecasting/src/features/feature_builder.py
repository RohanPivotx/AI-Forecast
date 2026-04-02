"""
Feature Engineering Module
============================
Performs all feature engineering steps in sequence:
  1. Label-encode categorical GROUP_COLS columns
  2. Compute per-group z-score scalers (fit on train period only)
  3. Apply per-group z-score normalisation to the full dataset
  4. Build lag, rolling-stat, trend, momentum, and calendar features
  5. Split data into train / test / historical sets

Encoders and scalers are persisted to disk for reproducibility and
potential GCS upload.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.constants import CAT_COLS, GROUP_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Label Encoding
# ─────────────────────────────────────────────────────────────────────────────

def create_encodings(
    df_monthly: pd.DataFrame, run_dir: str
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode all CAT_COLS columns. Save encoders to disk."""
    logger.info("Label encoding categorical columns…")
    encoders: Dict[str, LabelEncoder] = {}
    for col in CAT_COLS:
        if col in df_monthly.columns:
            enc = LabelEncoder()
            df_monthly[f"{col}_enc"] = enc.fit_transform(
                df_monthly[col].astype(str)
            )
            encoders[col] = enc
            logger.info(f"  {col}: {len(enc.classes_)} unique values")

    encoders_path = os.path.join(run_dir, "models", "label_encoders.pkl")
    try:
        with open(encoders_path, "wb") as f:
            pickle.dump(encoders, f)
        logger.info(f"Label encoders saved → {encoders_path}")
    except Exception as exc:
        logger.warning(f"Could not save label encoders to '{encoders_path}': {exc}")
    return df_monthly, encoders


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Per-Group Z-Score Scalers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_group_scalers(
    df_monthly: pd.DataFrame,
    train_cutoff: pd.Timestamp,
) -> Dict[tuple, Tuple[float, float]]:
    """Fit per-group (mu, std) using only training-period data."""
    df_train_only = df_monthly[df_monthly["date"] <= train_cutoff]
    group_scalers: Dict[tuple, Tuple[float, float]] = {}
    failed = 0
    for key, grp in df_train_only.groupby(GROUP_COLS):
        key = key if isinstance(key, tuple) else (key,)
        try:
            amounts = grp["amount"].values
            mu = float(amounts.mean())
            std = float(amounts.std()) if amounts.std() > 1e-6 else 1.0
            group_scalers[key] = (mu, std)
        except Exception as exc:
            logger.warning(f"Scaler computation failed for group {key}: {exc} — defaulting to (0, 1)")
            group_scalers[key] = (0.0, 1.0)
            failed += 1
    if failed:
        logger.warning(f"{failed} group(s) fell back to default scaler (0, 1)")
    logger.info(f"Computed scalers for {len(group_scalers)} groups (train period up to {train_cutoff.date()})")
    return group_scalers


def _apply_normalisation(
    df_monthly: pd.DataFrame,
    group_scalers: Dict[tuple, Tuple[float, float]],
) -> pd.DataFrame:
    """Apply per-group z-score normalisation. Unknown groups default to (0, 1)."""

    def _norm_group(grp: pd.DataFrame) -> pd.DataFrame:
        key = grp.name if isinstance(grp.name, tuple) else (grp.name,)
        mu, std = group_scalers.get(key, (0.0, 1.0))
        grp = grp.copy()
        grp["amount_norm"] = (grp["amount"].values - mu) / std
        return grp

    return df_monthly.groupby(GROUP_COLS, group_keys=False).apply(_norm_group)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Valid Lag Computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_valid_lags(
    n_months: int, max_lag_data_fraction: float
) -> List[int]:
    candidates = [1, 2, 3, 6, 12]
    cutoff = n_months * max_lag_data_fraction
    valid = [lag for lag in candidates if lag < cutoff]
    logger.info(f"Valid lags (< {cutoff:.0f} months): {valid}")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Feature Construction
# ─────────────────────────────────────────────────────────────────────────────

def _slope(vals: np.ndarray) -> float:
    """Linear trend slope over an array of values."""
    if len(vals) < 3:
        return 0.0
    t = np.arange(len(vals), dtype=float)
    return float(np.polyfit(t, vals, 1)[0])


def build_features(
    df_monthly: pd.DataFrame,
    cfg: dict,
    valid_lags: List[int],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build all model features on a normalised monthly DataFrame.

    Features constructed:
      - Lag features            : lag_Xm  (X in valid_lags)
      - Rolling mean/std/max    : roll_*_{2,3,6}m
      - Trend slope (3m window) : trend_slope_3m
      - Momentum                : momentum (m/m pct change)
      - Zero streak             : zero_streak
      - Activity rate           : activity_rate_3m
      - History length          : months_of_history
      - Calendar                : month, quarter, year, is_quarter_end
      - Encoded categoricals    : {col}_enc for each GROUP_COL

    Returns (df_with_features, feature_cols_list).
    """
    logger.info("Building lag, rolling, trend, and calendar features…")
    df = df_monthly.copy()
    nz = cfg["model"]["nonzero_threshold"]
    g = GROUP_COLS

    # Lag features
    for lag in valid_lags:
        df[f"lag_{lag}m"] = df.groupby(g)["amount_norm"].shift(lag)

    # Rolling mean / std / max (shift-1 to avoid look-ahead)
    for w in [2, 3, 6]:
        df[f"roll_mean_{w}m"] = df.groupby(g)["amount_norm"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean()
        )
        df[f"roll_std_{w}m"] = df.groupby(g)["amount_norm"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std().fillna(0)
        )
        df[f"roll_max_{w}m"] = df.groupby(g)["amount_norm"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).max()
        )

    # Trend slope over a 3-month window
    df["trend_slope_3m"] = df.groupby(g)["amount_norm"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).apply(_slope, raw=True)
    )

    # Month-over-month momentum
    df["momentum"] = df.groupby(g)["amount_norm"].transform(
        lambda x: x.shift(1).pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    )

    # Zero streak — how many consecutive near-zero months before current row
    df["is_nonzero"] = (df["amount_raw"].abs() > nz).astype(int)
    df["zero_streak"] = df.groupby(g)["is_nonzero"].transform(
        lambda x: x.eq(0).groupby(x.ne(0).cumsum()).cumcount()
    )

    # Activity rate — 3-month rolling mean of is_nonzero (lag-1)
    df["activity_rate_3m"] = df.groupby(g)["is_nonzero"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # How many months of history exist up to this row
    df["months_of_history"] = df.groupby(g).cumcount()

    # Calendar features
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["is_quarter_end"] = df["date"].dt.month.isin([3, 6, 9, 12]).astype(int)

    # Encoded categorical columns (created by create_encodings)
    enc_cols = [f"{c}_enc" for c in GROUP_COLS if f"{c}_enc" in df.columns]

    lag_cols = [f"lag_{lag}m" for lag in valid_lags]
    roll_cols = (
        [f"roll_mean_{w}m" for w in [2, 3, 6]]
        + [f"roll_std_{w}m" for w in [2, 3, 6]]
        + [f"roll_max_{w}m" for w in [2, 3, 6]]
    )
    feature_cols = (
        lag_cols
        + roll_cols
        + [
            "trend_slope_3m",
            "momentum",
            "zero_streak",
            "activity_rate_3m",
            "months_of_history",
        ]
        + ["month", "quarter", "year", "is_quarter_end"]
        + enc_cols
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    logger.info(f"Features built: {len(feature_cols)}")
    logger.info(f"Feature list  : {feature_cols}")
    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Public Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    df_monthly: pd.DataFrame,
    cfg: dict,
    run_dir: str,
) -> Tuple[pd.DataFrame, List[str], Dict[tuple, Tuple[float, float]], List[int]]:
    """
    Full feature engineering pipeline:
      1. Label-encode categoricals (fit & persist encoders)
      2. Compute per-group z-score scalers (train period only)
      3. Persist scalers to disk
      4. Apply normalisation to full dataset
      5. Compute valid lag lengths
      6. Build all features

    Returns: (df_with_features, feature_cols, group_scalers, valid_lags)
    """
    logger.info("=" * 60)
    logger.info("Feature Engineering")
    logger.info("=" * 60)

    train_cutoff = pd.Timestamp(cfg["dates"]["train_cutoff"])
    max_lag_frac = cfg["model"]["max_lag_data_fraction"]

    # Step 1: encode
    df_monthly, _ = create_encodings(df_monthly, run_dir)

    # Step 2: compute scalers
    group_scalers = _compute_group_scalers(df_monthly, train_cutoff)
    scalers_path = os.path.join(run_dir, "models", "group_scalers.pkl")
    try:
        with open(scalers_path, "wb") as f:
            pickle.dump(group_scalers, f)
        logger.info(
            f"Group scalers saved → {scalers_path}  ({len(group_scalers)} groups)"
        )
    except Exception as exc:
        logger.warning(f"Could not save group scalers to '{scalers_path}': {exc}")

    # Step 3: normalise
    df_monthly = _apply_normalisation(df_monthly, group_scalers)
    # reset_index after groupby-apply to avoid multi-index ambiguity
    df_monthly = df_monthly.reset_index(drop=True)

    # Step 4: valid lags
    n_months = df_monthly["date"].nunique()
    valid_lags = _compute_valid_lags(n_months, max_lag_frac)

    # Step 5: build features
    df_monthly, feature_cols = build_features(df_monthly, cfg, valid_lags)

    return df_monthly, feature_cols, group_scalers, valid_lags


# ─────────────────────────────────────────────────────────────────────────────
# Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    df_monthly: pd.DataFrame,
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the feature-enriched DataFrame into:
      - df_train    : rows up to train_cutoff (inclusive)
      - df_test     : rows between train_cutoff (exclusive) and test_cutoff (inclusive)
      - df_historical : all rows up to test_cutoff (union of train + test)
    """
    logger.info("=" * 60)
    logger.info("Train / Test Split")
    logger.info("=" * 60)

    train_cut = pd.Timestamp(cfg["dates"]["train_cutoff"])
    test_cut = pd.Timestamp(cfg["dates"]["test_cutoff"])
    nz = cfg["model"]["nonzero_threshold"]

    df_train = df_monthly[df_monthly["date"] <= train_cut].copy()
    df_test = df_monthly[
        (df_monthly["date"] > train_cut) & (df_monthly["date"] <= test_cut)
    ].copy()
    df_hist = df_monthly[df_monthly["date"] <= test_cut].copy()

    nz_train = (df_train["amount_raw"].abs() > nz).sum()
    nz_test = (df_test["amount_raw"].abs() > nz).sum()

    logger.info(
        f"Train   : {len(df_train):,} rows  "
        f"(non-zero: {nz_train:,} = {nz_train / max(len(df_train), 1) * 100:.1f}%)"
    )
    logger.info(
        f"Test    : {len(df_test):,} rows  "
        f"(non-zero: {nz_test:,} = {nz_test / max(len(df_test), 1) * 100:.1f}%)"
    )
    logger.info(f"History : {len(df_hist):,} rows")

    return df_train, df_test, df_hist
