"""
ETL — Ingest Module
====================
Loads raw transaction data from:
  - Local CSV (default, used for local development)
  - Google Cloud Storage (enabled via cfg['gcp']['use_gcs'] = true)
  - BigQuery (enabled via cfg['gcp']['use_gcs'] = true + bq_table set)

After loading, filters by source and aggregates to monthly net cash flow
per series (7-level hierarchy).  Missing months are gap-filled with zeros.
"""

import gc
import io
import os

import pandas as pd

from src.constants import GROUP_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Raw Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_from_local(cfg: dict) -> pd.DataFrame:
    """Load raw transaction data from a local CSV file."""
    data_file = cfg["data"]["data_file"]
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    logger.info(f"Loading data from local file: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df):,} raw rows")
    return df


def load_from_gcs(cfg: dict) -> pd.DataFrame:
    """
    Load raw transaction data from a GCS blob as CSV.
    Requires: pip install google-cloud-storage
    """
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required for GCS ingestion. "
            "Install it with: pip install google-cloud-storage"
        )
    gcp = cfg["gcp"]
    client = storage.Client(project=gcp["project_id"])
    bucket = client.bucket(gcp["bucket_name"])
    blob_name = gcp["data_prefix"] + cfg["data"]["data_file"]
    blob = bucket.blob(blob_name)
    logger.info(f"Downloading from GCS: gs://{gcp['bucket_name']}/{blob_name}")
    try:
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download or parse GCS blob '{blob_name}': {exc}"
        ) from exc
    logger.info(
        f"Loaded {len(df):,} rows from gs://{gcp['bucket_name']}/{blob_name}"
    )
    return df


def load_from_bigquery(cfg: dict) -> pd.DataFrame:
    """
    Load raw transaction data from BigQuery.
    Requires: pip install google-cloud-bigquery
    """
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError:
        raise ImportError(
            "google-cloud-bigquery is required for BQ ingestion. "
            "Install it with: pip install google-cloud-bigquery"
        )
    gcp = cfg["gcp"]
    client = bigquery.Client(project=gcp["project_id"])
    query = (
        f"SELECT * FROM `{gcp['project_id']}.{gcp['bq_dataset']}.{gcp['bq_table']}`"
    )
    logger.info(f"Running BigQuery query on {gcp['bq_dataset']}.{gcp['bq_table']}")
    try:
        df = client.query(query).to_dataframe()
    except Exception as exc:
        raise RuntimeError(
            f"BigQuery query failed on {gcp['bq_dataset']}.{gcp['bq_table']}: {exc}"
        ) from exc
    logger.info(f"Loaded {len(df):,} rows from BigQuery")
    return df


def ingest(cfg: dict) -> pd.DataFrame:
    """
    Route to the correct ingestion method based on config.
    Local CSV by default; GCS or BigQuery when gcp.use_gcs = true.
    """
    gcp_cfg = cfg.get("gcp", {})
    if gcp_cfg.get("use_gcs", False):
        if gcp_cfg.get("bq_table"):
            return load_from_bigquery(cfg)
        return load_from_gcs(cfg)
    return load_from_local(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Source Filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_sources(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply source_filter and exclude_sources from config."""
    data_cfg = cfg["data"]
    src_col = "source"

    source_filter = data_cfg.get("source_filter")
    if source_filter and src_col in df.columns:
        available = df[src_col].unique()
        if source_filter in available:
            df = df[df[src_col] == source_filter].copy()
            logger.info(f"Filtered to source='{source_filter}': {len(df):,} rows")
        else:
            logger.warning(
                f"Source '{source_filter}' not found. Available: {list(available)}"
            )

    exclude = data_cfg.get("exclude_sources", [])
    if exclude and src_col in df.columns:
        n_before = len(df)
        df = df[~df[src_col].isin(exclude)].copy()
        logger.info(f"Excluded sources {exclude}: {n_before:,} → {len(df):,} rows")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Level5 Category Filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_level5_categories(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Drop level5 categories that are entirely zero-valued or have too few
    non-zero transaction rows.  Applied on the raw (pre-aggregation) DataFrame
    so the decision is made on actual transaction_amount_usd values.

    Config key  data.level5_min_nonzero_rows  (default: 10)
    keeps categories where:
        - Zero_Percentage < 100  (at least one real transaction)
        - Non_zero_count  > level5_min_nonzero_rows
    """
    amount_col = "transaction_amount_usd"
    level5_col = "level5"
    min_nonzero: int = cfg["data"].get("level5_min_nonzero_rows", 10)

    if amount_col not in df.columns or level5_col not in df.columns:
        logger.warning(
            "filter_level5: required columns missing "
            f"({amount_col!r} or {level5_col!r}) — skipping filter."
        )
        return df

    # Drop rows missing key fields before computing per-category stats
    key_cols = [c for c in [amount_col, "value_datetime", "opcos", level5_col]
                if c in df.columns]
    n_before_drop = len(df)
    df = df.dropna(subset=key_cols).copy()
    n_dropped_na = n_before_drop - len(df)
    if n_dropped_na:
        logger.info(
            f"filter_level5: dropped {n_dropped_na:,} rows with missing key fields "
            f"({', '.join(key_cols)})"
        )

    # Compute per-level5: non-zero count and zero percentage
    zero_stats = (
        df.groupby(level5_col)[amount_col]
        .agg(
            nonzero_count=lambda x: (x != 0).sum(),
            zero_pct=lambda x: (x == 0).mean() * 100,
        )
        .reset_index()
    )

    # Keep: NOT all-zero AND enough non-zero rows
    keep_mask = (
        (zero_stats["zero_pct"] < 100) &
        (zero_stats["nonzero_count"] > min_nonzero)
    )
    keep_cats = zero_stats.loc[keep_mask, level5_col].tolist()
    drop_cats = zero_stats.loc[~keep_mask, level5_col].tolist()

    n_cats_before = df[level5_col].nunique()
    if drop_cats:
        preview = drop_cats[:5]
        extra = f" … +{len(drop_cats) - 5} more" if len(drop_cats) > 5 else ""
        logger.info(
            f"filter_level5: removing {len(drop_cats)} level5 categories "
            f"(all-zero or non_zero ≤ {min_nonzero}): {preview}{extra}"
        )

    df = df[df[level5_col].isin(keep_cats)].copy()
    n_cats_after = df[level5_col].nunique()
    logger.info(
        f"filter_level5: {n_cats_before} → {n_cats_after} level5 categories kept "
        f"({n_cats_before - n_cats_after} removed, min_nonzero_rows={min_nonzero})"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Monthly Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_monthly(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Aggregate transaction-level rows to monthly net cash flow per series.
    Inflows are positive, outflows are negative → summing gives net CF directly.
    Applies a minimum data-point filter to drop sparse series.
    """
    df = df.copy()
    try:
        df["value_datetime"] = pd.to_datetime(df["value_datetime"], utc=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse 'value_datetime' column: {exc}. "
            "Ensure the column contains valid datetime strings."
        ) from exc
    df["date"] = (
        df["value_datetime"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
    )

    df_monthly = (
        df.groupby(["date"] + GROUP_COLS)["transaction_amount_usd"]
        .sum()
        .reset_index()
        .rename(columns={"transaction_amount_usd": "amount"})
    )
    df_monthly["amount_raw"] = df_monthly["amount"].copy()

    # Drop series with fewer than min_series_data_points months of data
    min_pts = cfg["data"].get("min_series_data_points", 0)
    series_counts = (
        df_monthly.groupby(GROUP_COLS).size().reset_index(name="data_points")
    )
    series_keep = series_counts[series_counts["data_points"] > min_pts]
    n_before = df_monthly[GROUP_COLS].drop_duplicates().shape[0]
    df_monthly = df_monthly.merge(series_keep[GROUP_COLS], on=GROUP_COLS, how="inner")
    n_after = df_monthly[GROUP_COLS].drop_duplicates().shape[0]
    logger.info(
        f"Series filter (>{min_pts} data points): {n_before:,} → {n_after:,} kept"
    )

    return df_monthly


# ─────────────────────────────────────────────────────────────────────────────
# Gap Fill
# ─────────────────────────────────────────────────────────────────────────────

def gap_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every series has one row for every month in the full date range.
    Missing months are filled with amount = 0.
    """
    if df.empty:
        raise ValueError(
            "No series remaining after filtering — check min_series_data_points in config."
        )
    min_d = df["date"].min()
    max_d = df["date"].max()
    all_months = pd.date_range(min_d, max_d, freq="MS")
    n_expected = len(df[GROUP_COLS].drop_duplicates()) * len(all_months)
    parts = []
    skipped = 0
    for key, gdf in df.groupby(GROUP_COLS):
        try:
            ri = (
                gdf.set_index("date")
                .reindex(all_months)
                .reset_index()
                .rename(columns={"index": "date"})
            )
            for i, col in enumerate(GROUP_COLS):
                ri[col] = key[i] if isinstance(key, tuple) else key
            ri["amount"] = ri["amount"].fillna(0)
            ri["amount_raw"] = ri["amount_raw"].fillna(0)
            parts.append(ri[["date"] + GROUP_COLS + ["amount", "amount_raw"]])
        except Exception as exc:
            sk = "__".join(str(v) for v in (key if isinstance(key, tuple) else (key,)))
            logger.warning(f"gap_fill: skipping series '{sk}' due to error: {exc}")
            skipped += 1
    n_filled = n_expected - len(df)
    logger.info(
        f"Gap-fill: {n_filled:,} rows added across "
        f"{len(parts)} series ({skipped} skipped)"
    )
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date"] + GROUP_COLS)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def load_and_aggregate(cfg: dict) -> pd.DataFrame:
    """
    Full ingestion pipeline:
      1. Load raw data (local CSV or GCS/BQ)
      2. Filter sources
      3. Aggregate to monthly net cash flow
      4. Gap-fill missing months

    Returns the monthly-aggregated DataFrame ready for feature engineering.
    """
    logger.info("=" * 60)
    logger.info("ETL: Load & Monthly Net-Cash-Flow Aggregate")
    logger.info("=" * 60)

    df_raw = ingest(cfg)
    df_raw = filter_sources(df_raw, cfg)
    df_raw = filter_level5_categories(df_raw, cfg)
    df_monthly = aggregate_monthly(df_raw, cfg)
    del df_raw
    gc.collect()

    df_monthly = gap_fill(df_monthly)

    n_months = df_monthly["date"].nunique()
    logger.info(f"After gap-fill : {len(df_monthly):,} records")
    logger.info(
        f"Date range     : {df_monthly['date'].min().date()} "
        f"→ {df_monthly['date'].max().date()}"
    )
    logger.info(f"Unique months  : {n_months}")
    logger.info(f"Series (groups): {df_monthly.groupby(GROUP_COLS).ngroups:,}")

    return df_monthly
