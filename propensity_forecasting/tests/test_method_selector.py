"""
Tests — method_selector
========================
Unit tests for the auto_select_method function and ensemble builder guards.
Run with:  pytest tests/
"""

import sys
import os

# Ensure package root is on path when running tests directly
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_TESTS_DIR)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

import numpy as np
import pytest

from src.forecasting.method_selector import (
    auto_select_method,
    build_ensemble_classifier,
    build_ensemble_regressor,
    select_best_classifier,
    select_best_regressor,
)
from src.forecasting.models.ensemble import EnsembleClassifier, EnsembleRegressor
from src.forecasting.models.registry import MODEL_CONFIGS


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_reg_data():
    """Tiny regression dataset (10 samples, 3 features)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 3))
    y = rng.standard_normal(10)
    return X, y


@pytest.fixture()
def small_clf_data():
    """Tiny binary classification dataset (10 samples, 3 features)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 3))
    y = (rng.random(10) > 0.5).astype(int)
    return X, y


@pytest.fixture()
def tiny_model_configs():
    """Only LightGBM to keep tests fast."""
    return {
        k: v
        for k, v in MODEL_CONFIGS.items()
        if k == "lightgbm"
    }


# ─── auto_select_method ───────────────────────────────────────────────────────

class TestAutoSelectMethod:
    def _cfg(self, min_train=5):
        return {"model": {"min_train_months": min_train}}

    def test_short_series_uses_wma(self):
        assert auto_select_method(3, self._cfg(5)) == "wma"

    def test_equal_to_min_uses_ensemble(self):
        # n_train_rows == min_train_months → ensemble
        assert auto_select_method(5, self._cfg(5)) == "ensemble"

    def test_long_series_uses_ensemble(self):
        assert auto_select_method(24, self._cfg(5)) == "ensemble"

    def test_zero_rows_uses_wma(self):
        assert auto_select_method(0, self._cfg(5)) == "wma"


# ─── select_best_regressor ────────────────────────────────────────────────────

class TestSelectBestRegressor:
    def test_returns_valid_tuple(self, small_reg_data, tiny_model_configs):
        X, y = small_reg_data
        name, model, scores = select_best_regressor(
            X, y, tiny_model_configs, cv_folds=2
        )
        assert isinstance(name, str)
        assert hasattr(model, "fit") and hasattr(model, "predict")
        assert isinstance(scores, dict)
        assert name in tiny_model_configs

    def test_best_is_max_score(self, small_reg_data, tiny_model_configs):
        X, y = small_reg_data
        name, _, scores = select_best_regressor(
            X, y, tiny_model_configs, cv_folds=2
        )
        assert scores[name] == max(scores.values())


# ─── select_best_classifier ──────────────────────────────────────────────────

class TestSelectBestClassifier:
    def test_returns_valid_tuple(self, small_clf_data, tiny_model_configs):
        X, y = small_clf_data
        name, model, scores = select_best_classifier(
            X, y, tiny_model_configs, cv_folds=2
        )
        assert isinstance(name, str)
        assert hasattr(model, "fit") and hasattr(model, "predict_proba")

    def test_single_class_falls_back(self, tiny_model_configs):
        """If all labels are identical, classifier should fall back without error."""
        X = np.random.default_rng(0).standard_normal((8, 3))
        y = np.ones(8, dtype=int)
        name, model, scores = select_best_classifier(
            X, y, tiny_model_configs, cv_folds=3
        )
        assert name in tiny_model_configs


# ─── EnsembleRegressor ───────────────────────────────────────────────────────

class TestEnsembleRegressor:
    def test_predict_shape(self, small_reg_data, tiny_model_configs):
        X, y = small_reg_data
        ens, names, weights = build_ensemble_regressor(
            X, y, tiny_model_configs, top_k=1, cv_folds=2
        )
        assert isinstance(ens, EnsembleRegressor)
        ens.fit(X, y)
        preds = ens.predict(X)
        assert preds.shape == (len(X),)

    def test_weights_positive(self, small_reg_data, tiny_model_configs):
        X, y = small_reg_data
        _, _, weights = build_ensemble_regressor(
            X, y, tiny_model_configs, top_k=1, cv_folds=2
        )
        assert all(w > 0 for w in weights)


# ─── EnsembleClassifier ──────────────────────────────────────────────────────

class TestEnsembleClassifier:
    def test_predict_proba_shape(self, small_clf_data, tiny_model_configs):
        X, y = small_clf_data
        ens, _, _ = build_ensemble_classifier(
            X, y, tiny_model_configs, top_k=1, cv_folds=2
        )
        assert isinstance(ens, EnsembleClassifier)
        ens.fit(X, y)
        proba = ens.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_binary(self, small_clf_data, tiny_model_configs):
        X, y = small_clf_data
        ens, _, _ = build_ensemble_classifier(
            X, y, tiny_model_configs, top_k=1, cv_folds=2
        )
        ens.fit(X, y)
        preds = ens.predict(X)
        assert set(preds).issubset({0, 1})


# ─── build_ensemble_* with too few samples ───────────────────────────────────

class TestEnsembleGuards:
    def test_regressor_fallback_on_tiny_data(self, tiny_model_configs):
        X = np.random.default_rng(1).standard_normal((2, 3))
        y = np.array([1.0, 2.0])
        ens, names, weights = build_ensemble_regressor(
            X, y, tiny_model_configs, top_k=1, cv_folds=3
        )
        # Should return a valid ensemble without error
        assert isinstance(ens, EnsembleRegressor)

    def test_classifier_fallback_on_single_class(self, tiny_model_configs):
        X = np.random.default_rng(2).standard_normal((5, 3))
        y = np.zeros(5, dtype=int)
        ens, names, weights = build_ensemble_classifier(
            X, y, tiny_model_configs, top_k=1, cv_folds=3
        )
        assert isinstance(ens, EnsembleClassifier)
