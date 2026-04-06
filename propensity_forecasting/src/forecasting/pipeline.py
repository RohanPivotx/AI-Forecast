"""
Main Pipeline Orchestrator
===========================
`ForecastingPipeline` is the single class that owns all state and wires every
module together.  Steps mirror the original script with clean boundaries:

  Step 1  — ETL: load, filter, aggregate, gap-fill          (etl/ingest.py)
  Step 2  — Validation checks                               (etl/validate.py)
  Step 3  — Feature engineering + normalization             (features/feature_builder.py)
  Step 4  — Train / test split                              (features/feature_builder.py)
  Step 5  — Parallel per-series ensemble training           (utils/parallel.py)
  Step 6  — Batch-flush forecast generation (during train)
  Step 7  — Remaining-series forecast generation
  Step 8  — Forecast analysis summary
  Step 9  — Forecast vs actuals comparison
  Step 10 — Test-period h=1 validation
  Step 11 — Persist config snapshot

GCP upload stubs are present but gated behind cfg['gcp']['use_gcs'].
"""

import gc
import json
import math
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.constants import GROUP_COLS
from src.etl.ingest import load_and_aggregate
from src.etl.validate import run_all_checks
from src.features.feature_builder import (
    run_feature_engineering,
    split_data,
)
from src.utils.logger import get_logger
from src.utils.metrics import (
    batch_predict,
    batch_predict_proba,
    medape_safe,
    series_key_str,
    wma,
    wmape_safe,
)
from src.utils.parallel import run_parallel_training

logger = get_logger(__name__)


def _slope(vals: np.ndarray) -> float:
    if len(vals) < 3:
        return 0.0
    t = np.arange(len(vals), dtype=float)
    return float(np.polyfit(t, vals, 1)[0])


class ForecastingPipeline:
    """End-to-end cash-flow forecasting pipeline."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = cfg["output"]["output_root"]
        self.run_dir = os.path.join(output_root, f"run_{self.run_id}")

        for sub in [
            "data",
            "models",
            "reports",
            os.path.join("reports", "series_forecasts"),
        ]:
            os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True)

        # Runtime state
        self.df_monthly: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.df_historical: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.group_scalers: Dict = {}
        self.valid_lags: List[int] = []
        self.series_models: Dict = {}
        self.forecast_df: Optional[pd.DataFrame] = None
        self._flushed_forecasts: List[dict] = []
        self._hist_groups: Dict = {}

        logger.info("=" * 70)
        logger.info(
            f"  FORECASTING PIPELINE  run={self.run_id}  "
            f"output={self.run_dir}"
        )
        logger.info("=" * 70)
        # Log key config values for traceability
        dates = cfg.get("dates", {})
        model = cfg.get("model", {})
        perf = cfg.get("performance", {})
        logger.info(
            f"  Config: train_cutoff={dates.get('train_cutoff')}  "
            f"test_cutoff={dates.get('test_cutoff')}  "
            f"forecast_start={dates.get('forecast_start')}"
        )
        logger.info(
            f"  Config: horizons={model.get('forecast_horizons')}  "
            f"min_pts={cfg.get('data', {}).get('min_series_data_points')}  "
            f"workers={perf.get('parallel_workers')}  "
            f"ensemble={model.get('enable_ensemble')}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — ETL
    # ─────────────────────────────────────────────────────────────────────────

    def _step_etl(self):
        logger.info("STEP 1: ETL — Load & Aggregate")
        df_monthly = load_and_aggregate(self.cfg)
        n_months = df_monthly["date"].nunique()
        self.valid_lags = [
            lag
            for lag in [1, 2, 3, 6, 12]
            if lag < n_months * self.cfg["model"]["max_lag_data_fraction"]
        ]
        df_monthly.to_csv(
            os.path.join(self.run_dir, "data", "monthly_aggregated.csv"), index=False
        )
        self.df_monthly = df_monthly

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _step_validate(self):
        logger.info("STEP 2: Data Validation")
        # We pass an empty placeholder for df_raw since ingest already cleaned it
        import pandas as _pd
        df_raw_placeholder = _pd.DataFrame(
            columns=[
                "value_datetime",
                "transaction_amount_usd",
            ]
            + GROUP_COLS
        )
        results = run_all_checks(df_raw_placeholder, self.df_monthly, self.cfg)
        # Do not abort on warnings — just surface them via the logger
        critical = ["date_range", "no_duplicate_rows"]
        for key in critical:
            if not results.get(key, True):
                logger.error(
                    f"Critical validation check FAILED: {key}. Aborting."
                )
                sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────

    def _step_features(self):
        logger.info("STEP 3: Feature Engineering")
        (
            self.df_monthly,
            self.feature_cols,
            self.group_scalers,
            self.valid_lags,
        ) = run_feature_engineering(self.df_monthly, self.cfg, self.run_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Train / Test Split
    # ─────────────────────────────────────────────────────────────────────────

    def _step_split(self):
        logger.info("STEP 4: Train / Test / History Split")
        self.df_train, self.df_test, self.df_historical = split_data(
            self.df_monthly, self.cfg
        )
        # Build history lookup cache used later in forecast generation
        self._hist_groups = {
            grp_key: grp_df.sort_values("date").reset_index(drop=True)
            for grp_key, grp_df in self.df_historical.groupby(GROUP_COLS)
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Parallel Training + Batch-Flush (Step 6)
    # ─────────────────────────────────────────────────────────────────────────

    def _step_train(self):
        logger.info("STEP 5: Parallel Per-Series Ensemble Training")

        _, series_metrics_rows, series_registry_rows = run_parallel_training(
            df_train=self.df_train,
            df_test=self.df_test,
            feature_cols=self.feature_cols,
            group_scalers=self.group_scalers,
            cfg=self.cfg,
            run_dir=self.run_dir,
            flush_callback=self._flush_batch,
        )

        # Save aggregated metric files
        metrics_df = pd.DataFrame(series_metrics_rows)
        metrics_df.to_csv(
            os.path.join(self.run_dir, "reports", "series_metrics.csv"), index=False
        )

        registry_df = pd.DataFrame(series_registry_rows)
        registry_df.to_csv(
            os.path.join(self.run_dir, "reports", "series_model_registry.csv"),
            index=False,
        )

        if not metrics_df.empty:
            agg_h = (
                metrics_df.groupby("horizon")[["WMAPE", "MedAPE", "MAE", "OccAcc%"]]
                .mean()
                .reset_index()
            )
            agg_h.to_csv(
                os.path.join(self.run_dir, "reports", "horizon_metrics.csv"),
                index=False,
            )
            logger.info("Global horizon averages:")
            for _, hrow in agg_h.iterrows():
                wm_s = (
                    f"{hrow['WMAPE']:.1f}%"
                    if not math.isnan(hrow["WMAPE"])
                    else "N/A"
                )
                logger.info(f"  h={int(hrow['horizon'])}  avg WMAPE={wm_s}")

        self.series_metrics_df = metrics_df

    # ─────────────────────────────────────────────────────────────────────────
    # Batch Flush — called by the parallel runner after every N series
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_batch(self, keys_to_flush: list):
        """Generate and persist forecasts for a batch of series keyed by grp_key."""
        self._generate_forecasts_for_keys(keys_to_flush)

    def _generate_forecasts_for_keys(self, keys_to_forecast: list):
        """Core forecast-generation loop for a list of grp_keys."""
        forecast_start = pd.Timestamp(self.cfg["dates"]["forecast_start"])
        H = self.cfg["model"]["forecast_horizons"]
        nz = self.cfg["model"]["nonzero_threshold"]
        occur_threshold = self.cfg["model"].get("occur_threshold", 0.45)
        sparse_wma_window = self.cfg["model"].get("sparse_wma_window", 2)
        forecast_batch_size = self.cfg["performance"].get("forecast_batch_size", 512)
        series_forecast_dir = os.path.join(
            self.run_dir, "reports", "series_forecasts"
        )

        for grp_key in keys_to_forecast:
            sk = series_key_str(grp_key)
            grp_dict = dict(zip(GROUP_COLS, grp_key))

            hist_grp = self._hist_groups.get(grp_key)
            if hist_grp is None or hist_grp.empty:
                continue

            mu, std = self.group_scalers.get(grp_key, (0.0, 1.0))
            rolling_df = hist_grp.copy()

            # Prefer in-memory models; fall back to disk-loaded models
            if grp_key in self.series_models:
                series_reg_models = self.series_models[grp_key]["reg"]
                series_clf_models = self.series_models[grp_key]["clf"]
            else:
                try:
                    reg_path = os.path.join(
                        self.run_dir, "models", sk, "reg_models.pkl"
                    )
                    clf_path = os.path.join(
                        self.run_dir, "models", sk, "clf_models.pkl"
                    )
                    with open(reg_path, "rb") as f:
                        series_reg_models = pickle.load(f)
                    with open(clf_path, "rb") as f:
                        series_clf_models = pickle.load(f)
                except Exception:
                    series_reg_models = {}
                    series_clf_models = {}

            series_forecasts = []
            prev_amounts: list = []

            for h in range(1, H + 1):
                fdate = forecast_start + pd.DateOffset(months=h - 1)

                valid_h_keys = [
                    k for k, v in series_reg_models.items() if v is not None
                ]
                h_model = (
                    None
                    if not valid_h_keys
                    else h
                    if h in valid_h_keys
                    else min(valid_h_keys, key=lambda k: abs(k - h))
                )

                y_norm_nw = 0.0

                if h_model is None:
                    wma_val = wma(rolling_df["amount"], sparse_wma_window)
                    amount = float(wma_val)
                    y_norm_nw = (amount - mu) / std if std > 1e-8 else 0.0
                    method = "wma_fallback"
                else:
                    try:
                        feat_row = (
                            rolling_df.iloc[-1][self.feature_cols]
                            .fillna(0)
                            .infer_objects(copy=False)
                            .copy()
                        )
                        if "month" in feat_row.index:
                            feat_row["month"] = fdate.month
                        if "quarter" in feat_row.index:
                            feat_row["quarter"] = fdate.quarter
                        if "year" in feat_row.index:
                            feat_row["year"] = fdate.year
                        if "is_quarter_end" in feat_row.index:
                            feat_row["is_quarter_end"] = int(
                                fdate.month in [3, 6, 9, 12]
                            )
                        X_row = feat_row.values.reshape(1, -1)
                        clf_model = series_clf_models.get(h_model)
                        p_occur = (
                            float(
                                batch_predict_proba(
                                    clf_model, X_row, forecast_batch_size
                                )[0]
                            )
                            if clf_model is not None
                            else 1.0
                        )
                        y_norm = float(
                            batch_predict(
                                series_reg_models[h_model], X_row, forecast_batch_size
                            )[0]
                        )
                        y_raw = y_norm * std + mu
                        if p_occur < occur_threshold:
                            amount = 0.0
                            y_norm_nw = (0.0 - mu) / std if std > 1e-8 else 0.0
                        else:
                            amount = float(y_raw)
                            y_norm_nw = y_norm
                        method = "model"
                    except Exception:
                        wma_val = wma(rolling_df["amount"], sparse_wma_window)
                        amount = float(wma_val)
                        y_norm_nw = (amount - mu) / std if std > 1e-8 else 0.0
                        method = "wma_fallback"
                        logger.debug(
                            f"Forecast model predict failed for '{sk}' h={h}; "
                            f"falling back to WMA"
                        )

                # Constant-output collapse detection
                if method == "model" and len(prev_amounts) >= 1:
                    if all(abs(amount - pa) < 1.0 for pa in prev_amounts):
                        wma_val = wma(rolling_df["amount"], sparse_wma_window)
                        amount = float(wma_val)
                        y_norm_nw = (amount - mu) / std if std > 1e-8 else 0.0
                        method = "wma_fallback(collapse)"
                prev_amounts.append(amount)

                series_forecasts.append(
                    {
                        **grp_dict,
                        "date": fdate,
                        "forecast_amount": amount,
                        "method": method,
                    }
                )

                # Roll forward one month in the feature window
                new_row = rolling_df.iloc[-1].copy()
                new_row["date"] = fdate
                new_row["amount"] = amount
                new_row["amount_raw"] = abs(amount)
                new_row["amount_norm"] = y_norm_nw
                rolling_df = pd.concat(
                    [rolling_df, pd.DataFrame([new_row])], ignore_index=True
                )
                s = rolling_df["amount_norm"]
                for lag in self.valid_lags:
                    rolling_df[f"lag_{lag}m"] = s.shift(lag).fillna(0)
                for win in [2, 3, 6]:
                    rolling_df[f"roll_mean_{win}m"] = (
                        s.shift(1).rolling(win, min_periods=2).mean().fillna(0)
                    )
                    rolling_df[f"roll_std_{win}m"] = (
                        s.shift(1).rolling(win, min_periods=2).std().fillna(0)
                    )
                    rolling_df[f"roll_max_{win}m"] = (
                        s.shift(1).rolling(win, min_periods=2).max().fillna(0)
                    )
                rolling_df["trend_slope_3m"] = (
                    s.shift(1)
                    .rolling(3, min_periods=2)
                    .apply(_slope, raw=True)
                    .fillna(0)
                )
                rolling_df["momentum"] = (
                    s.shift(1)
                    .pct_change()
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )
                rolling_df["is_nonzero"] = (
                    rolling_df["amount_raw"].abs() > nz
                ).astype(int)
                rolling_df["zero_streak"] = (
                    rolling_df["is_nonzero"]
                    .eq(0)
                    .groupby(rolling_df["is_nonzero"].ne(0).cumsum())
                    .cumcount()
                )
                rolling_df["activity_rate_3m"] = (
                    rolling_df["is_nonzero"]
                    .shift(1)
                    .rolling(3, min_periods=1)
                    .mean()
                    .fillna(0)
                )
                rolling_df["months_of_history"] = range(len(rolling_df))
                rolling_df["month"] = rolling_df["date"].dt.month
                rolling_df["quarter"] = rolling_df["date"].dt.quarter
                rolling_df["year"] = rolling_df["date"].dt.year
                rolling_df["is_quarter_end"] = (
                    rolling_df["date"].dt.month.isin([3, 6, 9, 12]).astype(int)
                )
                rolling_df = rolling_df.replace([np.inf, -np.inf], 0).fillna(0)

            sf_df = pd.DataFrame(series_forecasts)
            sf_df.to_csv(
                os.path.join(series_forecast_dir, f"{sk}_forecast.csv"), index=False
            )
            self._flushed_forecasts.extend(series_forecasts)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7 — Generate forecasts for any remaining series
    # ─────────────────────────────────────────────────────────────────────────

    def _step_forecasts(self):
        logger.info("STEP 7: Generating Forecasts for Remaining Series")

        all_series = self.df_historical[GROUP_COLS].drop_duplicates().reset_index(
            drop=True
        )
        already_flushed: set = set()
        if self._flushed_forecasts:
            flushed_df = pd.DataFrame(self._flushed_forecasts)
            already_flushed = {
                tuple(row[c] for c in GROUP_COLS)
                for _, row in flushed_df[GROUP_COLS].drop_duplicates().iterrows()
            }

        remaining = [
            tuple(row[c] for c in GROUP_COLS)
            for _, row in all_series.iterrows()
            if tuple(row[c] for c in GROUP_COLS) not in already_flushed
        ]

        logger.info(
            f"Already flushed: {len(already_flushed):,}   "
            f"Remaining: {len(remaining):,}"
        )
        if remaining:
            self._generate_forecasts_for_keys(remaining)

        # Combine and save
        all_forecasts = list(self._flushed_forecasts)
        forecast_df = (
            pd.DataFrame(all_forecasts)
            .sort_values(GROUP_COLS + ["date"])
            .reset_index(drop=True)
        )

        if forecast_df.empty:
            logger.error(
                "No forecasts generated. Check training completed successfully."
            )
        else:
            H = self.cfg["model"]["forecast_horizons"]
            logger.info(
                f"Total forecast records : {len(forecast_df):,}"
            )
            logger.info(
                f"Unique series          : "
                f"{forecast_df[GROUP_COLS].drop_duplicates().shape[0]:,}"
            )
            logger.info(
                f"Date range             : "
                f"{forecast_df['date'].min().date()} → "
                f"{forecast_df['date'].max().date()}"
            )
            logger.info(
                f"Methods used           : "
                f"{forecast_df['method'].value_counts().to_dict()}"
            )
            nan_count = forecast_df["forecast_amount"].isna().sum()
            if nan_count > 0:
                logger.warning(f"{nan_count} NaN forecast values — filling with 0")
                forecast_df["forecast_amount"] = forecast_df[
                    "forecast_amount"
                ].fillna(0.0)

            out = os.path.join(self.run_dir, "forecasts_3months.csv")
            forecast_df.to_csv(out, index=False)
            logger.info(f"Combined forecast saved → {out}")

        self.forecast_df = forecast_df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8 — Forecast Analysis Summary
    # ─────────────────────────────────────────────────────────────────────────

    def _step_analysis(self):
        logger.info("STEP 8: Forecast Analysis Summary")
        if self.forecast_df is None or self.forecast_df.empty:
            logger.warning("No forecasts to analyse.")
            return

        df = self.forecast_df.copy()
        H = self.cfg["model"]["forecast_horizons"]

        def _make_top_level_agg(df_fc, group_keys, filename):
            agg = (
                df_fc.groupby(group_keys + ["date"])["forecast_amount"]
                .sum()
                .reset_index()
            )
            agg["year_month"] = agg["date"].dt.to_period("M").astype(str)
            wide = agg.pivot_table(
                index=group_keys,
                columns="year_month",
                values="forecast_amount",
                aggfunc="sum",
                fill_value=0,
            ).reset_index()
            wide.columns.name = None
            month_cols = [c for c in wide.columns if c not in group_keys]
            wide["total_3m_netcf"] = wide[month_cols].sum(axis=1)
            wide.to_csv(
                os.path.join(self.run_dir, "reports", filename), index=False
            )
            return wide

        _make_top_level_agg(df, ["opcos"], "forecast_top_opcos.csv")
        _make_top_level_agg(
            df, ["opcos", "entity_region"], "forecast_top_entity_region.csv"
        )
        _make_top_level_agg(
            df, ["opcos", "entity_region", "level2"], "forecast_top_level2.csv"
        )

        # Full hierarchy + per-series summary
        hierarchy_summary = (
            df.groupby(GROUP_COLS)
            .agg(
                mean_forecast=("forecast_amount", "mean"),
                std_forecast=("forecast_amount", "std"),
                total_netcf=("forecast_amount", "sum"),
                months=("date", "count"),
                method=("method", lambda x: x.iloc[0]),
            )
            .reset_index()
            .sort_values("total_netcf", ascending=False)
        )
        hierarchy_summary.to_csv(
            os.path.join(self.run_dir, "reports", "forecast_by_hierarchy.csv"),
            index=False,
        )

        # Account-name level summary (opcos → account_name rollup, matching original)
        ea_summary = (
            df.groupby(
                ["opcos", "entity_region", "entity_country", "entity_name", "account_name"]
            )
            .agg(
                total_netcf=("forecast_amount", "sum"),
                mean_forecast=("forecast_amount", "mean"),
                n_level2=("level2", "nunique"),
                n_level5=("level5", "nunique"),
                records=("date", "count"),
            )
            .reset_index()
            .sort_values("total_netcf", ascending=False)
        )
        ea_summary.to_csv(
            os.path.join(self.run_dir, "reports", "forecast_by_account_name.csv"),
            index=False,
        )

        bottom_level_df = (
            df[GROUP_COLS + ["date", "forecast_amount", "method"]]
            .sort_values(GROUP_COLS + ["date"])
            .reset_index(drop=True)
        )
        bottom_level_df.to_csv(
            os.path.join(
                self.run_dir, "reports", "forecast_bottom_level_full_hierarchy.csv"
            ),
            index=False,
        )

        inconsistent = (df.groupby(GROUP_COLS).size() != H).sum()
        logger.info(
            f"Forecast Coverage: {len(hierarchy_summary):,} series, "
            f"{len(df):,} records, "
            f"{H} months per series"
            + (f"  [{inconsistent} inconsistent]" if inconsistent else "")
        )
        logger.info(
            f"Total net-CF: ${df['forecast_amount'].sum():,.0f}  "
            f"Mean: ${df['forecast_amount'].mean():,.0f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 9 — Forecast vs Actuals
    # ─────────────────────────────────────────────────────────────────────────

    def _step_compare_actuals(self):
        logger.info("STEP 9: Forecast vs Actuals Comparison")
        if self.forecast_df is None or self.forecast_df.empty:
            logger.warning("No forecasts available.")
            return

        forecast_start = pd.Timestamp(self.cfg["dates"]["forecast_start"])
        forecast_end = forecast_start + pd.DateOffset(
            months=self.cfg["model"]["forecast_horizons"] - 1
        )
        actuals_fc = self.df_monthly[
            (self.df_monthly["date"] >= forecast_start)
            & (self.df_monthly["date"] <= forecast_end)
        ].copy()

        if len(actuals_fc) > 0:
            self._run_actuals_comparison(actuals_fc, label="forecast-period")
        else:
            logger.info(
                "No actuals found for forecast period — "
                "comparing against test period instead"
            )
            train_cut = pd.Timestamp(self.cfg["dates"]["train_cutoff"])
            test_cut = pd.Timestamp(self.cfg["dates"]["test_cutoff"])
            actuals_te = self.df_monthly[
                (self.df_monthly["date"] > train_cut)
                & (self.df_monthly["date"] <= test_cut)
            ][GROUP_COLS + ["date", "amount", "amount_raw"]].copy()
            self._step_test_period_validation(actuals_te)

    def _run_actuals_comparison(self, actuals: pd.DataFrame, label: str = ""):
        """Merge forecasts with actuals and compute + persist metric reports."""
        if "amount_raw" not in actuals.columns and "amount" in actuals.columns:
            actuals = actuals.copy()
            actuals["amount_raw"] = actuals["amount"]

        actuals = actuals[GROUP_COLS + ["date", "amount_raw"]].rename(
            columns={"amount_raw": "actual_amount"}
        )
        merged = self.forecast_df.merge(actuals, on=GROUP_COLS + ["date"], how="left")
        merged_valid = merged.dropna(subset=["actual_amount"]).copy()

        if merged_valid.empty:
            logger.warning("No overlapping dates between forecasts and actuals.")
            return

        y_true = merged_valid["actual_amount"].values
        y_pred = merged_valid["forecast_amount"].values
        ow = wmape_safe(y_true, y_pred)
        om = medape_safe(y_true, y_pred)
        oa = float(mean_absolute_error(y_true, y_pred))
        ob = float(np.mean(y_pred - y_true))

        logger.info(
            f"Overall Metrics ({label}): "
            f"WMAPE={ow:.1f}%  MedAPE={om:.1f}%  "
            f"MAE={oa:,.0f}  Bias={ob:+,.0f}"
        )

        # Per-horizon
        forecast_start = pd.Timestamp(self.cfg["dates"]["forecast_start"])
        merged_valid = merged_valid.copy()
        merged_valid["horizon"] = (
            (
                merged_valid["date"].dt.to_period("M")
                - forecast_start.to_period("M")
            )
            .apply(lambda x: x.n)
            + 1
        ).astype(int)
        horizon_rows = []
        for h, hdf in merged_valid.groupby("horizon"):
            yt = hdf["actual_amount"].values
            yp = hdf["forecast_amount"].values
            horizon_rows.append(
                {
                    "horizon": h,
                    "wmape": wmape_safe(yt, yp),
                    "medape": medape_safe(yt, yp),
                    "mae": float(mean_absolute_error(yt, yp)) if len(yt) > 0 else np.nan,
                    "bias": float(np.mean(yp - yt)),
                    "n_groups": len(hdf),
                    "nonzero_actual": int(
                        (np.abs(yt) > self.cfg["model"]["nonzero_threshold"]).sum()
                    ),
                }
            )
        pd.DataFrame(horizon_rows).to_csv(
            os.path.join(
                self.run_dir, "reports", "forecast_vs_actuals_by_horizon.csv"
            ),
            index=False,
        )

        # Per-hierarchy
        hier_rows = []
        for grp_key, gdf in merged_valid.groupby(GROUP_COLS):
            gk = grp_key if isinstance(grp_key, tuple) else (grp_key,)
            yt = gdf["actual_amount"].values
            yp = gdf["forecast_amount"].values
            hier_rows.append(
                {
                    **dict(zip(GROUP_COLS, gk)),
                    "wmape": wmape_safe(yt, yp),
                    "medape": medape_safe(yt, yp),
                    "mae": float(mean_absolute_error(yt, yp)) if len(yt) > 0 else np.nan,
                    "bias": float(np.mean(yp - yt)),
                    "n_months": len(gdf),
                    "total_actual": float(yt.sum()),
                    "total_forecast": float(yp.sum()),
                    "method": gdf["method"].iloc[0] if "method" in gdf.columns else "unknown",
                }
            )
        pd.DataFrame(hier_rows).sort_values("wmape").to_csv(
            os.path.join(
                self.run_dir, "reports", "forecast_vs_actuals_by_hierarchy.csv"
            ),
            index=False,
        )

        # Per-hierarchy monthly breakdown
        nz = self.cfg["model"]["nonzero_threshold"]
        merged_valid["year_month"] = merged_valid["date"].dt.to_period("M").astype(str)
        monthly_rows = []
        for grp_key, gmdf in merged_valid.groupby(GROUP_COLS + ["year_month"]):
            hier_vals = grp_key[: len(GROUP_COLS)]
            ym_val = grp_key[-1]
            yt = gmdf["actual_amount"].values
            yp = gmdf["forecast_amount"].values
            total_actual = float(yt.sum())
            total_forecast = float(yp.sum())
            pct_diff = (
                (total_forecast - total_actual) / abs(total_actual) * 100
                if abs(total_actual) > nz
                else float("nan")
            )
            monthly_rows.append(
                {
                    **dict(zip(GROUP_COLS, hier_vals)),
                    "year_month": ym_val,
                    "n_months": len(gmdf),
                    "total_actual": total_actual,
                    "total_forecast": total_forecast,
                    "pct_diff": pct_diff,
                    "wmape": wmape_safe(yt, yp),
                    "medape": medape_safe(yt, yp),
                    "mae": float(mean_absolute_error(yt, yp)) if len(yt) > 0 else float("nan"),
                    "bias": float(np.mean(yp - yt)),
                    "method": gmdf["method"].iloc[0] if "method" in gmdf.columns else "unknown",
                }
            )
        pd.DataFrame(monthly_rows).sort_values(
            GROUP_COLS + ["year_month"]
        ).reset_index(drop=True).to_csv(
            os.path.join(
                self.run_dir,
                "reports",
                "forecast_vs_actuals_by_hierarchy_monthly.csv",
            ),
            index=False,
        )

        # Per-hierarchy date-level detail
        monthly_detail_rows = []
        for grp_key, gwdf in merged_valid.groupby(GROUP_COLS + ["date"]):
            hier_vals = grp_key[: len(GROUP_COLS)]
            date_val = grp_key[-1]
            yt = gwdf["actual_amount"].values
            yp = gwdf["forecast_amount"].values
            total_actual = float(yt.sum())
            total_forecast = float(yp.sum())
            abs_var = total_forecast - total_actual
            pct_var = (
                abs_var / abs(total_actual) * 100
                if abs(total_actual) > nz
                else float("nan")
            )
            accuracy = max(0.0, 100.0 - wmape_safe(yt, yp)) if len(yt) > 0 else float("nan")
            monthly_detail_rows.append(
                {
                    **dict(zip(GROUP_COLS, hier_vals)),
                    "date": date_val,
                    "actual_amount": total_actual,
                    "forecast_amount": total_forecast,
                    "absolute_variance": abs_var,
                    "pct_variance": pct_var,
                    "accuracy": accuracy,
                    "method": gwdf["method"].iloc[0] if "method" in gwdf.columns else "unknown",
                }
            )
        monthly_detail_df = (
            pd.DataFrame(monthly_detail_rows)
            .sort_values(GROUP_COLS + ["date"])
            .reset_index(drop=True)
        )
        monthly_detail_df.to_csv(
            os.path.join(
                self.run_dir,
                "reports",
                "forecast_vs_actuals_by_hierarchy_monthly_detail.csv",
            ),
            index=False,
        )
        monthly_detail_rows_df = monthly_detail_df.rename(
            columns={
                "date": "period_year_month",
                "opcos": "opcos",
                "entity_region": "region",
                "level5": "cash_flow_category",
                "actual_amount": "actuals",
                "forecast_amount": "ai_forecast_numbers",
                "absolute_variance": "aifc__actuals",
                "pct_variance": "variance",
            },
        )
        monthly_detail_rows_df.to_csv(
            os.path.join(
                self.run_dir,
                "forecast_vs_actuals_by_hierarchy_monthly_upload.csv",
            ),
            index=False,
        )


        merged_valid.to_csv(
            os.path.join(self.run_dir, "reports", "forecast_vs_actuals.csv"),
            index=False,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 10 — Test-period h=1 validation
    # ─────────────────────────────────────────────────────────────────────────

    def _step_test_period_validation(self, actuals_te: pd.DataFrame):
        logger.info("STEP 10: Test-Period h=1 Validation")
        occur_threshold = self.cfg["model"].get("occur_threshold", 0.45)
        forecast_batch_size = self.cfg["performance"].get("forecast_batch_size", 512)
        test_dates = sorted(actuals_te["date"].unique())

        # Index actuals for O(1) lookup
        actuals_idx = {}
        for row in actuals_te.itertuples(index=False):
            key = tuple(getattr(row, c) for c in GROUP_COLS) + (row.date,)
            actuals_idx[key] = getattr(row, "amount_raw", np.nan)

        # Collect all trained series keys (in-memory + disk)
        all_trained_keys = set(self.series_models.keys())
        models_dir = os.path.join(self.run_dir, "models")
        disk_keys_found = 0
        for entry in os.scandir(models_dir):
            if not entry.is_dir():
                continue
            if os.path.exists(
                os.path.join(entry.path, "reg_models.pkl")
            ) and os.path.exists(os.path.join(entry.path, "clf_models.pkl")):
                parts = entry.name.split("__")
                if len(parts) == len(GROUP_COLS):
                    all_trained_keys.add(tuple(parts))
                    disk_keys_found += 1
        logger.info(
            f"Test-period validation: {len(all_trained_keys)} series keys "
            f"({len(self.series_models)} in-memory, {disk_keys_found} from disk)"
        )

        rows = []
        for grp_key in all_trained_keys:
            grp_dict = dict(zip(GROUP_COLS, grp_key))
            mu, std = self.group_scalers.get(grp_key, (0.0, 1.0))

            if grp_key in self.series_models:
                sdata = self.series_models[grp_key]
                reg_h1 = sdata["reg"].get(1)
                clf_h1 = sdata["clf"].get(1)
            else:
                sk = series_key_str(grp_key)
                try:
                    with open(
                        os.path.join(
                            self.run_dir, "models", sk, "reg_models.pkl"
                        ),
                        "rb",
                    ) as f:
                        reg_models = pickle.load(f)
                    with open(
                        os.path.join(
                            self.run_dir, "models", sk, "clf_models.pkl"
                        ),
                        "rb",
                    ) as f:
                        clf_models = pickle.load(f)
                    reg_h1 = reg_models.get(1)
                    clf_h1 = clf_models.get(1)
                except Exception:
                    continue

            if reg_h1 is None:
                continue

            hist_grp = self._hist_groups.get(grp_key)
            if hist_grp is None or hist_grp.empty:
                continue

            for tdate in test_dates:
                context = hist_grp[hist_grp["date"] < tdate]
                if context.empty:
                    continue
                X_row = (
                    context.iloc[-1][self.feature_cols]
                    .fillna(0)
                    .infer_objects(copy=False)
                    .values.reshape(1, -1)
                )
                try:
                    p_occur = (
                        float(
                            batch_predict_proba(
                                clf_h1, X_row, forecast_batch_size
                            )[0]
                        )
                        if clf_h1 is not None
                        else 1.0
                    )
                    y_norm = float(
                        batch_predict(reg_h1, X_row, forecast_batch_size)[0]
                    )
                    y_raw = y_norm * std + mu
                    pred = 0.0 if p_occur < occur_threshold else float(y_raw)
                except Exception:
                    pred = 0.0
                    p_occur = np.nan

                actual = actuals_idx.get(grp_key + (tdate,), np.nan)
                rows.append(
                    {
                        **grp_dict,
                        "date": tdate,
                        "actual_amount": actual,
                        "forecast_amount": pred,
                        "p_occur": p_occur if clf_h1 is not None else np.nan,
                    }
                )

        if not rows:
            logger.warning("No test-period predictions generated.")
            return

        cmp_df = (
            pd.DataFrame(rows)
            .sort_values(GROUP_COLS + ["date"])
            .reset_index(drop=True)
        )
        valid = cmp_df.dropna(subset=["actual_amount", "forecast_amount"])
        if valid.empty:
            logger.warning("No matched actuals/predictions for test period.")
            return

        y_true = valid["actual_amount"].values
        y_pred = valid["forecast_amount"].values
        ow = wmape_safe(y_true, y_pred)
        om = medape_safe(y_true, y_pred)
        oa = float(mean_absolute_error(y_true, y_pred))
        ob = float(np.mean(y_pred - y_true))

        logger.info(
            f"Test-Period (h=1) Metrics: "
            f"WMAPE={ow:.1f}%  MedAPE={om:.1f}%  "
            f"MAE={oa:,.0f}  Bias={ob:+,.0f}"
        )

        cmp_df.to_csv(
            os.path.join(self.run_dir, "reports", "test_period_comparison.csv"),
            index=False,
        )

        # Per-series breakdown
        series_rows = []
        for grp_key, gdf in valid.groupby(GROUP_COLS):
            gk = grp_key if isinstance(grp_key, tuple) else (grp_key,)
            yt = gdf["actual_amount"].values
            yp = gdf["forecast_amount"].values
            series_rows.append(
                {
                    **dict(zip(GROUP_COLS, gk)),
                    "n_months": len(gdf),
                    "wmape": wmape_safe(yt, yp),
                    "medape": medape_safe(yt, yp),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "bias": float(np.mean(yp - yt)),
                    "total_actual": float(yt.sum()),
                    "total_forecast": float(yp.sum()),
                }
            )
        pd.DataFrame(series_rows).sort_values("wmape").to_csv(
            os.path.join(
                self.run_dir, "reports", "test_period_by_series.csv"
            ),
            index=False,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 11 — Persist config snapshot + optional GCS upload
    # ─────────────────────────────────────────────────────────────────────────

    def _step_save_config(self):
        logger.info("STEP 11: Persisting Config Snapshot")
        config_path = os.path.join(self.run_dir, "config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(self.cfg, f, indent=2, default=str)
            logger.info(f"Config snapshot saved → {config_path}")
        except Exception as exc:
            logger.error(f"Failed to write config snapshot to '{config_path}': {exc}")

        if self.cfg.get("gcp", {}).get("use_gcs", False):
            self._upload_run_to_gcs()

    def _upload_run_to_gcs(self):
        """
        Upload the entire run directory to GCS.
        Called only when cfg['gcp']['use_gcs'] = true.
        Requires: pip install google-cloud-storage
        """
        try:
            from google.cloud import storage  # type: ignore
        except ImportError:
            logger.warning(
                "google-cloud-storage not installed — skipping GCS upload."
            )
            return

        gcp = self.cfg["gcp"]
        client = storage.Client(project=gcp["project_id"])
        bucket = client.bucket(gcp["bucket_name"])

        for dirpath, _, filenames in os.walk(self.run_dir):
            for fname in filenames:
                local_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(local_path, self.run_dir)
                blob_name = (
                    gcp.get("reports_prefix", "reports/")
                    + f"run_{self.run_id}/"
                    + rel_path.replace(os.sep, "/")
                )
                bucket.blob(blob_name).upload_from_filename(local_path)
                logger.info(
                    f"Uploaded → gs://{gcp['bucket_name']}/{blob_name}"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Execute all pipeline steps in order with per-step timing and error reporting."""
        # Steps marked non_fatal will log an error but not abort the pipeline
        NON_FATAL = {"Save Config"}
        steps = [
            ("ETL",             self._step_etl),
            ("Validation",      self._step_validate),
            ("Features",        self._step_features),
            ("Split",           self._step_split),
            ("Training",        self._step_train),
            ("Forecasts",       self._step_forecasts),
            ("Analysis",        self._step_analysis),
            ("Actuals Compare", self._step_compare_actuals),
            ("Save Config",     self._step_save_config),
        ]
        pipeline_start = time.time()
        for step_name, step_fn in steps:
            t0 = time.time()
            try:
                step_fn()
                logger.info(
                    f"  [{step_name}] completed in {time.time() - t0:.1f}s"
                )
            except Exception:
                elapsed = time.time() - t0
                logger.error(
                    f"PIPELINE FAILED at step '{step_name}' "
                    f"after {elapsed:.1f}s:\n{traceback.format_exc()}"
                )
                if step_name in NON_FATAL:
                    logger.warning(f"  [{step_name}] non-fatal — continuing.")
                    continue
                sys.exit(1)

        total = time.time() - pipeline_start
        logger.info("=" * 70)
        logger.info(f"  PIPELINE COMPLETE — run={self.run_id}  total={total:.1f}s")
        logger.info(f"  Results → {self.run_dir}")
        logger.info("=" * 70)
