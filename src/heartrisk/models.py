"""Candidate model pipelines. Every pipeline accepts the raw 13-feature table with NaNs."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import CATEGORICAL, NUMERIC

SEED = 42


def _linear_prep() -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def _tree_prep() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
        ("cat", SimpleImputer(strategy="constant", fill_value=-1), CATEGORICAL),
    ])


def candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", _linear_prep()),
            ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", _tree_prep()),
            ("clf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        ]),
        # XGBoost learns a default direction for missing values, so no imputation.
        "xgboost": Pipeline([
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
                eval_metric="logloss", random_state=SEED, n_jobs=-1)),
        ]),
    }
