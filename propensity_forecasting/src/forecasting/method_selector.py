"""
Method Selector
================
Selects the best regressor and/or classifier for a given series via
cross-validated model comparison, then builds weighted ensembles from the
top-K performers.

All public functions are module-level so they can be imported by the
parallel worker (which runs in a separate process).
"""

import math
import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score

from src.forecasting.models.ensemble import EnsembleClassifier, EnsembleRegressor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal CV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_model(model, X, y, scoring: str, cv: int) -> float:
    actual_cv = min(cv, len(y))
    if actual_cv < 2:
        try:
            model.fit(X, y)
            return float(model.score(X, y))
        except Exception as exc:
            logger.debug(f"Model fit/score failed (fallback CV): {exc}")
            return float("-inf")
    try:
        scores = cross_val_score(model, X, y, scoring=scoring, cv=actual_cv)
        return float(scores.mean())
    except Exception as exc:
        logger.debug(f"cross_val_score failed for scoring='{scoring}': {exc}")
        return float("-inf")


# ─────────────────────────────────────────────────────────────────────────────
# Best-single-model selectors
# ─────────────────────────────────────────────────────────────────────────────

def select_best_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_configs: Dict,
    cv_folds: int = 3,
) -> Tuple[str, object, Dict[str, float]]:
    """CV-compare all regressors and return (best_name, unfitted_model, scores)."""
    scores: Dict[str, float] = {}
    for name, config in model_configs.items():
        try:
            model = config["regressor"](**config["reg_params"].copy())
            score = _evaluate_model(
                model, X_train, y_train,
                scoring="neg_mean_absolute_error", cv=cv_folds,
            )
            scores[name] = score
        except Exception:
            scores[name] = float("-inf")

    best_name = max(scores, key=scores.get)
    best_config = model_configs[best_name]
    best_model = best_config["regressor"](**best_config["reg_params"].copy())
    logger.debug(f"Best regressor: {best_name}  scores={scores}")
    return best_name, best_model, scores


def select_best_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_configs: Dict,
    cv_folds: int = 3,
) -> Tuple[str, object, Dict[str, float]]:
    """CV-compare all classifiers and return (best_name, unfitted_model, scores)."""
    # Guard: if any class has fewer samples than folds, skip CV and use fallback
    unique, counts = np.unique(y_train, return_counts=True)
    if len(counts) > 0 and counts.min() < cv_folds:
        fallback_name = next(iter(model_configs))
        fallback_cfg = model_configs[fallback_name]
        return (
            fallback_name,
            fallback_cfg["classifier"](**fallback_cfg["clf_params"].copy()),
            {},
        )

    scores: Dict[str, float] = {}
    for name, config in model_configs.items():
        try:
            model = config["classifier"](**config["clf_params"].copy())
            score = _evaluate_model(
                model, X_train, y_train, scoring="f1", cv=cv_folds
            )
            scores[name] = score
        except Exception:
            scores[name] = float("-inf")

    best_name = max(scores, key=scores.get)
    best_config = model_configs[best_name]
    best_model = best_config["classifier"](**best_config["clf_params"].copy())
    logger.debug(f"Best classifier: {best_name}  scores={scores}")
    return best_name, best_model, scores


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble builders
# ─────────────────────────────────────────────────────────────────────────────

def _lgbm_fallback_reg(model_configs: Dict):
    cfg = model_configs.get("lightgbm") or next(iter(model_configs.values()))
    return cfg["regressor"](**cfg["reg_params"].copy()), 1.0


def _lgbm_fallback_clf(model_configs: Dict):
    cfg = model_configs.get("lightgbm") or next(iter(model_configs.values()))
    return cfg["classifier"](**cfg["clf_params"].copy()), 1.0


def build_ensemble_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_configs: Dict,
    top_k: int = 3,
    cv_folds: int = 3,
) -> Tuple[EnsembleRegressor, List[str], List[float]]:
    """
    Build a weighted EnsembleRegressor from the top-K CV performers.
    Weight = 1 / (MAE + eps)   so better models get higher weight.
    """
    if len(X_train) < max(cv_folds, 3):
        fb_model, fb_w = _lgbm_fallback_reg(model_configs)
        return EnsembleRegressor([(fb_model, fb_w)]), ["lightgbm"], [fb_w]

    _, _, scores = select_best_regressor(X_train, y_train, model_configs, cv_folds)
    valid_scores = {
        k: v
        for k, v in scores.items()
        if v != float("-inf") and not (isinstance(v, float) and math.isnan(v))
    }

    if not valid_scores:
        fb_model, fb_w = _lgbm_fallback_reg(model_configs)
        return (
            EnsembleRegressor([(fb_model, fb_w)]),
            ["lightgbm(fallback)"],
            [fb_w],
        )

    sorted_models = sorted(
        valid_scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    models_with_weights, weights, names = [], [], []
    for name, score in sorted_models:
        config = model_configs[name]
        model = config["regressor"](**config["reg_params"].copy())
        w = 1.0 / (abs(score) + 1e-6)
        if w > 0:
            models_with_weights.append((model, w))
            weights.append(w)
            names.append(name)

    if not models_with_weights:
        fb_model, fb_w = _lgbm_fallback_reg(model_configs)
        return (
            EnsembleRegressor([(fb_model, fb_w)]),
            ["lightgbm(fallback)"],
            [fb_w],
        )

    return EnsembleRegressor(models_with_weights), names, weights


def build_ensemble_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_configs: Dict,
    top_k: int = 3,
    cv_folds: int = 3,
) -> Tuple[EnsembleClassifier, List[str], List[float]]:
    """
    Build a weighted EnsembleClassifier from the top-K CV F1 performers.
    Weight = F1 score (floored at eps).
    """
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2 or len(X_train) < max(cv_folds, 3):
        fb_model, fb_w = _lgbm_fallback_clf(model_configs)
        return (
            EnsembleClassifier([(fb_model, fb_w)]),
            ["lightgbm(fallback)"],
            [fb_w],
        )

    _, _, scores = select_best_classifier(X_train, y_train, model_configs, cv_folds)
    valid_scores = {
        k: v
        for k, v in scores.items()
        if v != float("-inf") and not (isinstance(v, float) and math.isnan(v))
    }

    if not valid_scores:
        fb_model, fb_w = _lgbm_fallback_clf(model_configs)
        return (
            EnsembleClassifier([(fb_model, fb_w)]),
            ["lightgbm(fallback)"],
            [fb_w],
        )

    sorted_models = sorted(
        valid_scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    models_with_weights, weights, names = [], [], []
    for name, score in sorted_models:
        config = model_configs[name]
        model = config["classifier"](**config["clf_params"].copy())
        w = max(score, 1e-6)
        models_with_weights.append((model, w))
        weights.append(w)
        names.append(name)

    if not models_with_weights:
        fb_model, fb_w = _lgbm_fallback_clf(model_configs)
        return (
            EnsembleClassifier([(fb_model, fb_w)]),
            ["lightgbm(fallback)"],
            [fb_w],
        )

    return EnsembleClassifier(models_with_weights), names, weights


# ─────────────────────────────────────────────────────────────────────────────
# Auto method selector (series-length based)
# ─────────────────────────────────────────────────────────────────────────────

def auto_select_method(n_train_rows: int, cfg: dict) -> str:
    """
    Return the recommended modelling method for a series given its length.

    Rules (configurable via cfg['model']):
      < min_train_months  → 'wma'          (too short for ML)
      < min_nonzero_train_rows (nonzero) → 'wma'
      otherwise           → 'ensemble'

    This is used by the parallel worker to decide whether to fit models at all
    or to fall back to a simple WMA for very sparse series.
    """
    min_train = cfg["model"].get("min_train_months", 5)
    if n_train_rows < min_train:
        return "wma"
    return "ensemble"
