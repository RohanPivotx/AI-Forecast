"""
ETL — Validate Module
======================
Lightweight data quality checks on raw and monthly-aggregated DataFrames.
All checks return bool.  run_all_checks() aggregates results without raising
so the caller decides whether to abort or continue with warnings.
"""

from typing import Dict, List, Tuple

import pandas as pd

from src.constants import GROUP_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_RAW_COLUMNS: List[str] = [
    "value_datetime",
    "transaction_amount_usd",
] + GROUP_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_required_columns(df: pd.DataFrame, required: List[str]) -> bool:
    """Verify all required columns are present in the raw DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    logger.info(
        f"Column check passed — all {len(required)} required columns present."
    )
    return True


def check_date_range(df_monthly: pd.DataFrame, cfg: dict) -> bool:
    """Verify the data covers the training cutoff date."""
    train_cutoff = pd.Timestamp(cfg["dates"]["train_cutoff"])
    min_date = df_monthly["date"].min()
    max_date = df_monthly["date"].max()
    if max_date < train_cutoff:
        logger.error(
            f"Data max date {max_date.date()} is before "
            f"train_cutoff {train_cutoff.date()}."
        )
        return False
    logger.info(
        f"Date range check passed: {min_date.date()} → {max_date.date()} "
        f"covers train_cutoff {train_cutoff.date()}"
    )
    return True


def check_no_all_zero_series(df_monthly: pd.DataFrame) -> bool:
    """Warn if any series has all-zero amounts (completely empty series)."""
    series_nonzero = df_monthly.groupby(GROUP_COLS)["amount"].apply(
        lambda x: (x != 0).any()
    )
    all_zero = series_nonzero[~series_nonzero]
    if len(all_zero) > 0:
        logger.warning(f"{len(all_zero)} series have all-zero amounts.")
        return False
    logger.info("No all-zero series found.")
    return True


def check_min_series_length(
    df_monthly: pd.DataFrame, min_months: int
) -> Tuple[bool, pd.DataFrame]:
    """Return (ok, short_series_df) for series below min_months threshold."""
    series_lengths = (
        df_monthly.groupby(GROUP_COLS).size().reset_index(name="n_months")
    )
    short = series_lengths[series_lengths["n_months"] < min_months]
    if len(short) > 0:
        logger.warning(
            f"{len(short)} series have < {min_months} months of history."
        )
    else:
        logger.info(f"All series have >= {min_months} months of history.")
    return len(short) == 0, short


def check_no_duplicate_rows(df_monthly: pd.DataFrame) -> bool:
    """Ensure no duplicate (date, series) combinations exist after aggregation."""
    dupes = df_monthly.duplicated(subset=["date"] + GROUP_COLS).sum()
    if dupes > 0:
        logger.warning(f"{dupes} duplicate (date, series) rows found.")
        return False
    logger.info("No duplicate (date, series) rows found.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_checks(
    df_raw: pd.DataFrame,
    df_monthly: pd.DataFrame,
    cfg: dict,
) -> Dict[str, bool]:
    """
    Run all validation checks and return a results dict {check_name: passed}.
    Does NOT raise — caller decides whether to abort on failure.
    """
    results: Dict[str, bool] = {}

    results["required_columns"] = check_required_columns(
        df_raw, REQUIRED_RAW_COLUMNS
    )
    results["date_range"] = check_date_range(df_monthly, cfg)
    results["no_all_zero_series"] = check_no_all_zero_series(df_monthly)
    results["no_duplicate_rows"] = check_no_duplicate_rows(df_monthly)

    ok, _ = check_min_series_length(
        df_monthly, cfg["model"].get("min_history_months", 6)
    )
    results["min_series_length"] = ok

    passed = sum(results.values())
    total = len(results)
    logger.info(
        f"Validation summary: {passed}/{total} checks passed — {results}"
    )
    return results
