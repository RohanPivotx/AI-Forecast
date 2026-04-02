"""
Model Registry — all supported base model configurations in one place.
========================================================================
Each entry maps a short name → dict with keys:
    regressor   : class reference
    classifier  : class reference
    reg_params  : kwargs for the regressor
    clf_params  : kwargs for the classifier

Import this dict wherever model instantiation is needed.
Adding a new model means adding one entry here — no other file changes required.
"""

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

MODEL_CONFIGS: dict = {
    "lightgbm": {
        "regressor": LGBMRegressor,
        "classifier": LGBMClassifier,
        "reg_params": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        },
        "clf_params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 24,
            "min_child_samples": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        },
    },
    "xgboost": {
        "regressor": XGBRegressor,
        "classifier": XGBClassifier,
        "reg_params": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "gamma": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        },
        "clf_params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "eval_metric": "logloss",
        },
    },
    "random_forest": {
        "regressor": RandomForestRegressor,
        "classifier": RandomForestClassifier,
        "reg_params": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "max_features": 0.5,
            "random_state": 42,
            "n_jobs": -1,
        },
        "clf_params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "max_features": 0.5,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "gradient_boosting": {
        "regressor": GradientBoostingRegressor,
        "classifier": GradientBoostingClassifier,
        "reg_params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "subsample": 0.8,
            "random_state": 42,
        },
        "clf_params": {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "subsample": 0.8,
            "random_state": 42,
        },
    },
    "extra_trees": {
        "regressor": ExtraTreesRegressor,
        "classifier": ExtraTreesClassifier,
        "reg_params": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "max_features": 0.5,
            "random_state": 42,
            "n_jobs": -1,
        },
        "clf_params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "max_features": 0.5,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "hist_gradient_boosting": {
        "regressor": HistGradientBoostingRegressor,
        "classifier": HistGradientBoostingClassifier,
        "reg_params": {
            "max_iter": 500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_samples_leaf": 10,
            "l2_regularization": 0.1,
            "random_state": 42,
        },
        "clf_params": {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_samples_leaf": 10,
            "l2_regularization": 0.1,
            "random_state": 42,
        },
    },
}

if _CATBOOST_AVAILABLE:
    MODEL_CONFIGS["catboost"] = {
        "regressor": CatBoostRegressor,
        "classifier": CatBoostClassifier,
        "reg_params": {
            "iterations": 500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 5,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        },
        "clf_params": {
            "iterations": 300,
            "learning_rate": 0.03,
            "depth": 5,
            "l2_leaf_reg": 5,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        },
    }
