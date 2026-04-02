"""
Parallel Training Worker
=========================
Contains the standalone module-level function `_train_series_worker` that is
submitted to `ProcessPoolExecutor`.  Being at module level (not a method or
closure) ensures clean pickling across processes on all platforms.

Also exposes `run_parallel_training()` — the main entry point called by
pipeline.py — which submits all series to the pool, collects results, persists
models, and performs periodic batch-flush to keep RAM bounded.
"""

import gc
import math
import os
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.constants import GROUP_COLS
from src.utils.logger import get_logger
from src.utils.metrics import (
    batch_predict,
    batch_predict_proba,
    series_key_str,
    wmape_safe,
    medape_safe,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker (must NOT be a method so ProcessPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _train_series_worker(args):
    """
    Train ONE series (all horizons) inside a worker process.

    Arguments are packed as a single tuple so ProcessPoolExecutor can pass
    them through the multiprocessing queue without overhead.

    Returns a result dict with keys:
        grp_key, sk, grp_dict,
        reg_models, clf_models,      ← {horizon: model}
        metrics_rows, registry_row,
        error (str | None)
    """
    (
        grp_key, sk, grp_dict,
        df_tr_records, df_te_records,
        feature_cols, group_scalers,
        H, nz, min_nz, cv_folds,
        enable_selection, enable_ensemble, top_k,
        occur_threshold, model_configs,
        forecast_batch_size,
    ) = args

    # These imports happen inside the worker process
    import math
    import traceback

    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    # Import from pipeline package — worker process has the package on sys.path
    from src.forecasting.method_selector import (
        build_ensemble_classifier,
        build_ensemble_regressor,
        select_best_classifier,
        select_best_regressor,
    )
    from src.forecasting.models.ensemble import EnsembleClassifier, EnsembleRegressor
    from src.utils.metrics import (
        batch_predict,
        batch_predict_proba,
        wmape_safe,
        medape_safe,
    )

    result = {
        "grp_key": grp_key,
        "sk": sk,
        "grp_dict": grp_dict,
        "reg_models": {},
        "clf_models": {},
        "metrics_rows": [],
        "registry_row": None,
        "error": None,
    }

    try:
        df_tr = pd.DataFrame(df_tr_records)
        df_te = pd.DataFrame(df_te_records)

        if len(df_tr) < 3:
            result["error"] = f"too few training rows ({len(df_tr)})"
            return result

        # Drop rows where >98% of lag features are zero (degenerate rows)
        lag_cols = [c for c in feature_cols if c.startswith("lag_")]
        if lag_cols:
            zero_frac = (df_tr[lag_cols] == 0).sum(axis=1) / len(lag_cols)
            df_tr = df_tr[zero_frac < 0.98].copy()

        if len(df_tr) < 3:
            result["error"] = f"too few rows after zero-lag filter ({len(df_tr)})"
            return result

        # ── Ensemble / model selection on 80% of the training period ──
        sorted_dates = sorted(df_tr["date"].unique())
        cutoff_idx = max(1, int(len(sorted_dates) * 0.8))
        cutoff_date = sorted_dates[cutoff_idx - 1]
        tr_sel = df_tr[df_tr["date"] <= cutoff_date].copy()

        tr_sel["target_norm"] = tr_sel["amount_norm"].shift(-1)
        tr_sel["target_occur"] = (
            tr_sel["amount_raw"].shift(-1).abs() > nz
        ).astype(int)
        tr_sel = tr_sel.dropna(subset=["target_norm"])

        def _clean_name(n):
            return n.replace("(fallback)", "").strip()

        if len(tr_sel) < 3:
            reg_names, clf_names = ["lightgbm"], ["lightgbm"]
            reg_weights, clf_weights = [1.0], [1.0]
            best_reg_name = best_clf_name = "lightgbm"
        elif enable_selection and enable_ensemble and len(model_configs) > 0:
            X_sel = tr_sel[feature_cols].values
            y_sel_n = tr_sel["target_norm"].values
            y_sel_oc = tr_sel["target_occur"].values
            _, reg_names, reg_weights = build_ensemble_regressor(
                X_sel, y_sel_n, model_configs, top_k=top_k, cv_folds=cv_folds
            )
            _, clf_names, clf_weights = build_ensemble_classifier(
                X_sel, y_sel_oc, model_configs, top_k=top_k, cv_folds=cv_folds
            )
            best_reg_name = f"ENSEMBLE({'+'.join(reg_names)})"
            best_clf_name = f"ENSEMBLE({'+'.join(clf_names)})"
        else:
            X_sel = tr_sel[feature_cols].values
            y_sel_n = tr_sel["target_norm"].values
            y_sel_oc = tr_sel["target_occur"].values
            best_reg_name, _, _ = select_best_regressor(
                X_sel, y_sel_n, model_configs, cv_folds
            )
            best_clf_name, _, _ = select_best_classifier(
                X_sel, y_sel_oc, model_configs, cv_folds
            )
            reg_names = [best_reg_name]
            clf_names = [best_clf_name]
            reg_weights = [1.0]
            clf_weights = [1.0]

        result["registry_row"] = {
            **grp_dict,
            "reg_ensemble": best_reg_name,
            "clf_ensemble": best_clf_name,
            "n_train_rows": len(df_tr),
            "n_test_rows": len(df_te),
        }

        mu, std = group_scalers.get(grp_key, (0.0, 1.0))

        series_reg_models = {}
        series_clf_models = {}
        h_metrics_rows = []

        for h in range(1, H + 1):
            tr_h = df_tr.copy()
            te_h = df_te.copy()

            tr_h["target_norm"] = tr_h["amount_norm"].shift(-h)
            tr_h["target_raw"] = tr_h["amount_raw"].shift(-h)
            tr_h["target_occur"] = (
                tr_h["amount_raw"].shift(-h).abs() > nz
            ).astype(int)
            tr_h = tr_h.dropna(subset=["target_norm"])

            te_h["target_norm"] = te_h["amount_norm"].shift(-h)
            te_h["target_raw"] = te_h["amount_raw"].shift(-h)
            te_h["target_occur"] = (
                te_h["amount_raw"].shift(-h).abs() > nz
            ).astype(int)
            te_h = te_h.dropna(subset=["target_norm"])

            X_tr = tr_h[feature_cols].values
            y_tr_n = tr_h["target_norm"].values
            y_tr_oc = tr_h["target_occur"].values
            nz_mask = tr_h["target_occur"].values == 1
            X_tr_nz = X_tr[nz_mask]
            y_tr_nz = y_tr_n[nz_mask]

            # Instantiate ensemble members for this horizon
            reg_mw = [
                (
                    model_configs[_clean_name(name)]["regressor"](
                        **model_configs[_clean_name(name)]["reg_params"].copy()
                    ),
                    w,
                )
                for name, w in zip(reg_names, reg_weights)
                if _clean_name(name) in model_configs
            ]
            clf_mw = [
                (
                    model_configs[_clean_name(name)]["classifier"](
                        **model_configs[_clean_name(name)]["clf_params"].copy()
                    ),
                    w,
                )
                for name, w in zip(clf_names, clf_weights)
                if _clean_name(name) in model_configs
            ]

            if not reg_mw:
                lgbm = model_configs.get(
                    "lightgbm", next(iter(model_configs.values()))
                )
                reg_mw = [
                    (lgbm["regressor"](**lgbm["reg_params"].copy()), 1.0)
                ]
            if not clf_mw:
                lgbm = model_configs.get(
                    "lightgbm", next(iter(model_configs.values()))
                )
                clf_mw = [
                    (lgbm["classifier"](**lgbm["clf_params"].copy()), 1.0)
                ]

            reg = (
                EnsembleRegressor(reg_mw) if len(reg_mw) > 1 else reg_mw[0][0]
            )
            clf = (
                EnsembleClassifier(clf_mw) if len(clf_mw) > 1 else clf_mw[0][0]
            )

            p_const = 1.0
            unique_oc = np.unique(y_tr_oc)
            if len(unique_oc) < 2:
                clf = None
                p_const = float(unique_oc[0]) if len(unique_oc) > 0 else 0.0
            elif y_tr_oc.sum() >= 5:
                try:
                    clf.fit(X_tr, y_tr_oc)
                except Exception:
                    clf = None
            else:
                clf = None
                p_const = 1.0

            if len(X_tr_nz) >= min_nz:
                reg.fit(X_tr_nz, y_tr_nz)
            else:
                reg = None

            series_reg_models[h] = reg
            series_clf_models[h] = clf

            # ── Per-horizon metrics ──
            if len(te_h) >= 3 and reg is not None:
                X_te = te_h[feature_cols].values
                y_te_r = te_h["target_raw"].values
                y_te_oc = te_h["target_occur"].values
                p_occur = (
                    batch_predict_proba(clf, X_te, forecast_batch_size)
                    if clf is not None
                    else np.full(len(X_te), p_const)
                )
                y_pred_norm = batch_predict(reg, X_te, forecast_batch_size)
                y_pred_raw = y_pred_norm * std + mu
                p_binary = (p_occur >= occur_threshold).astype(float)
                y_pred_gated = y_pred_raw * p_binary
                wm = wmape_safe(y_te_r, y_pred_gated)
                md = medape_safe(y_te_r, y_pred_gated)
                ma = float(mean_absolute_error(y_te_r, y_pred_gated))
                acc = (
                    float(
                        ((p_occur >= occur_threshold).astype(int) == y_te_oc).mean()
                    )
                    * 100
                )
                eval_src = "test"
            elif reg is not None and len(X_tr_nz) >= 3:
                y_is_norm = batch_predict(reg, X_tr_nz, forecast_batch_size)
                y_is_raw = y_is_norm * std + mu
                y_true_raw = tr_h.loc[
                    tr_h["target_occur"] == 1, "target_raw"
                ].values
                wm = wmape_safe(y_true_raw, y_is_raw)
                md = medape_safe(y_true_raw, y_is_raw)
                ma = float(mean_absolute_error(y_true_raw, y_is_raw))
                acc = np.nan
                eval_src = "in-sample"
            else:
                wm = md = ma = acc = np.nan
                eval_src = "no-data"

            h_metrics_rows.append(
                {
                    **grp_dict,
                    "horizon": h,
                    "WMAPE": wm,
                    "MedAPE": md,
                    "MAE": ma,
                    "OccAcc%": acc,
                    "train_rows": len(tr_h),
                    "train_nz_rows": int(nz_mask.sum()),
                    "test_rows": len(te_h),
                    "eval_source": eval_src,
                }
            )

        result["reg_models"] = series_reg_models
        result["clf_models"] = series_clf_models
        result["metrics_rows"] = h_metrics_rows

    except Exception as e:
        result["error"] = f"{e}\n{traceback.format_exc()}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main parallel training entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_parallel_training(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    group_scalers: Dict,
    cfg: dict,
    run_dir: str,
    flush_callback,          # callable(keys_to_flush) — provided by pipeline
) -> Tuple[Dict, List[dict], List[dict]]:
    """
    Submit all series to ProcessPoolExecutor, collect results, persist models,
    and trigger batch-flush every `batch_flush_threshold` series.

    Returns:
        series_models      : {grp_key: {"reg": {h: model}, "clf": {h: model}, "scalers": (mu,std)}}
        series_metrics_rows: flat list of per-horizon metric dicts
        series_registry_rows: flat list of per-series registry dicts
    """
    H = cfg["model"]["forecast_horizons"]
    nz = cfg["model"]["nonzero_threshold"]
    min_nz = cfg["model"]["min_nonzero_train_rows"]
    cv_folds = cfg["model"].get("cv_folds", 3)
    enable_selection = cfg["model"].get("enable_model_selection", True)
    enable_ensemble = cfg["model"].get("enable_ensemble", True)
    top_k = cfg["model"].get("ensemble_top_k", 3)
    occur_threshold = cfg["model"].get("occur_threshold", 0.45)
    n_workers = cfg["performance"].get("parallel_workers", 4)
    flush_threshold = cfg["performance"].get("batch_flush_threshold", 200)
    forecast_batch_size = cfg["performance"].get("forecast_batch_size", 512)

    # Import model configs here (not inside worker — only names/classes needed)
    from src.forecasting.models.registry import MODEL_CONFIGS
    model_configs = MODEL_CONFIGS

    all_series = df_train[GROUP_COLS].drop_duplicates().reset_index(drop=True)
    n_series = len(all_series)
    logger.info(f"Total series to train : {n_series}")
    logger.info(f"Parallel workers      : {n_workers}")
    logger.info(f"Batch flush threshold : {flush_threshold}")

    # Pre-group for O(1) per-series data lookup
    train_groups = {
        grp_key: grp_df.copy()
        for grp_key, grp_df in df_train.groupby(GROUP_COLS)
    }
    test_groups = {
        grp_key: grp_df.copy()
        for grp_key, grp_df in df_test.groupby(GROUP_COLS)
    }

    worker_args = []
    for _, row in all_series.iterrows():
        grp_key = tuple(row[c] for c in GROUP_COLS)
        sk = series_key_str(grp_key)
        grp_dict = dict(zip(GROUP_COLS, grp_key))
        df_tr = train_groups.get(grp_key, pd.DataFrame())
        df_te = test_groups.get(grp_key, pd.DataFrame())
        worker_args.append(
            (
                grp_key, sk, grp_dict,
                df_tr.to_dict("records"),
                df_te.to_dict("records"),
                feature_cols, group_scalers,
                H, nz, min_nz, cv_folds,
                enable_selection, enable_ensemble, top_k,
                occur_threshold, model_configs,
                forecast_batch_size,
            )
        )

    series_models: Dict = {}
    series_metrics_rows: List[dict] = []
    series_registry_rows: List[dict] = []
    trained_count = 0
    pending_keys: List = []

    def _save_series_model(result: dict):
        grp_key = result["grp_key"]
        sk = result["sk"]
        series_model_dir = os.path.join(run_dir, "models", sk)
        os.makedirs(series_model_dir, exist_ok=True)
        with open(os.path.join(series_model_dir, "reg_models.pkl"), "wb") as f:
            pickle.dump(result["reg_models"], f)
        with open(os.path.join(series_model_dir, "clf_models.pkl"), "wb") as f:
            pickle.dump(result["clf_models"], f)
        series_models[grp_key] = {
            "reg": result["reg_models"],
            "clf": result["clf_models"],
            "scalers": group_scalers.get(grp_key, (0.0, 1.0)),
        }

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_train_series_worker, args): args[0]
            for args in worker_args
        }

        for future in as_completed(futures):
            grp_key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                grp_dict = dict(zip(GROUP_COLS, grp_key))
                logger.error(
                    f"Worker crashed for {series_key_str(grp_key)}: {exc}"
                )
                series_registry_rows.append(
                    {
                        **grp_dict,
                        "reg_ensemble": "ERROR",
                        "clf_ensemble": "ERROR",
                        "n_train_rows": -1,
                        "n_test_rows": -1,
                    }
                )
                continue

            trained_count += 1
            sk = result["sk"]
            grp_dict = result["grp_dict"]

            if result["error"]:
                logger.warning(
                    f"[{trained_count}/{n_series}] {sk}  ERROR: {result['error']}"
                )
                series_registry_rows.append(
                    {
                        **grp_dict,
                        "reg_ensemble": "ERROR",
                        "clf_ensemble": "ERROR",
                        "n_train_rows": -1,
                        "n_test_rows": -1,
                    }
                )
                continue

            logger.info(f"[{trained_count}/{n_series}] ✓ {sk}")
            _save_series_model(result)
            pending_keys.append(grp_key)

            if result["registry_row"]:
                series_registry_rows.append(result["registry_row"])
            series_metrics_rows.extend(result["metrics_rows"])

            # Log per-horizon summary
            for hm in result["metrics_rows"]:
                wm_s = (
                    f"{hm['WMAPE']:.1f}%"
                    if not (isinstance(hm["WMAPE"], float) and math.isnan(hm["WMAPE"]))
                    else "N/A"
                )
                ma_s = (
                    f"{hm['MAE']:,.0f}"
                    if not (isinstance(hm["MAE"], float) and math.isnan(hm["MAE"]))
                    else "N/A"
                )
                logger.info(
                    f"   h={hm['horizon']} WMAPE={wm_s:>8s}  MAE={ma_s:>14s} "
                    f" tr={hm['train_rows']}/te={hm['test_rows']}"
                    f"/nz={hm['train_nz_rows']}  [{hm.get('eval_source', '?')}]"
                )

            # Periodic flush to keep RAM bounded
            if len(pending_keys) >= flush_threshold:
                logger.info(
                    f"Flushing batch of {len(pending_keys)} series "
                    "(generate forecasts + free memory)"
                )
                flush_callback(pending_keys)
                for k in pending_keys:
                    series_models.pop(k, None)
                gc.collect()
                pending_keys = []

    # Final flush
    if pending_keys:
        logger.info(
            f"Final flush: {len(pending_keys)} remaining series"
        )
        flush_callback(pending_keys)
        for k in pending_keys:
            series_models.pop(k, None)
        gc.collect()

    return series_models, series_metrics_rows, series_registry_rows
