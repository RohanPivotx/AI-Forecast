"""
Ensemble Model Wrappers
========================
EnsembleRegressor and EnsembleClassifier are weighted aggregators that hold
multiple base models and combine their predictions proportionally.

They satisfy the scikit-learn predict / predict_proba interface so they can be
swapped in anywhere a single model object is expected.
"""

import math
from typing import List, Tuple

import numpy as np


class EnsembleRegressor:
    """Weighted mean of multiple regressors.  Trained in .fit(), predicted in .predict()."""

    def __init__(self, models_with_weights: List[Tuple]):
        self.models_with_weights = models_with_weights

    def fit(self, X, y):
        fitted = []
        for model, w in self.models_with_weights:
            try:
                model.fit(X, y)
                fitted.append((model, w))
            except Exception:
                pass
        if not fitted:
            # At least try the first model — surface the error if it also fails
            self.models_with_weights[0][0].fit(X, y)
            fitted = [self.models_with_weights[0]]
        self.models_with_weights = fitted
        return self

    def predict(self, X) -> np.ndarray:
        total_weight = sum(w for _, w in self.models_with_weights)
        if total_weight <= 0 or not self.models_with_weights:
            preds = (
                np.mean([m.predict(X) for m, _ in self.models_with_weights], axis=0)
                if self.models_with_weights
                else np.zeros(len(X))
            )
            return preds
        preds = np.zeros(len(X))
        for model, weight in self.models_with_weights:
            preds += (weight / total_weight) * model.predict(X)
        return preds

    @property
    def feature_importances_(self):
        total_weight = sum(w for _, w in self.models_with_weights)
        if total_weight <= 0:
            return None
        fi = None
        for model, weight in self.models_with_weights:
            if hasattr(model, "feature_importances_"):
                contrib = (weight / total_weight) * model.feature_importances_
                fi = contrib if fi is None else fi + contrib
        return fi


class EnsembleClassifier:
    """Weighted soft-voting classifier (P(class=1) averaged by weight)."""

    def __init__(self, models_with_weights: List[Tuple]):
        self.models_with_weights = models_with_weights
        self.classes_ = None

    def fit(self, X, y):
        fitted = []
        for model, w in self.models_with_weights:
            try:
                model.fit(X, y)
                fitted.append((model, w))
            except Exception:
                pass
        if not fitted:
            self.models_with_weights[0][0].fit(X, y)
            fitted = [self.models_with_weights[0]]
        self.models_with_weights = fitted
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        total_weight = sum(w for _, w in self.models_with_weights)
        if total_weight <= 0 or not self.models_with_weights:
            return np.full((len(X), 2), 0.5)
        proba = np.zeros((len(X), 2))
        for model, weight in self.models_with_weights:
            model_proba = model.predict_proba(X)
            aligned = np.zeros((len(X), 2))
            model_classes = getattr(model, "classes_", np.array([1]))
            if model_proba.ndim == 1:
                aligned[:, 1] = model_proba
                aligned[:, 0] = 1.0 - model_proba
            elif model_proba.shape[1] == 1:
                cls = model_classes[0] if len(model_classes) > 0 else 1
                if cls == 1:
                    aligned[:, 1] = model_proba[:, 0]
                    aligned[:, 0] = 1.0 - model_proba[:, 0]
                else:
                    aligned[:, 0] = model_proba[:, 0]
                    aligned[:, 1] = 1.0 - model_proba[:, 0]
            else:
                for col_idx, cls in enumerate(model_classes):
                    if cls in (0, 1):
                        aligned[:, int(cls)] = model_proba[:, col_idx]
            proba += (weight / total_weight) * aligned
        return proba

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
