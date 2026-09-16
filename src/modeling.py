"""Candidate pipelines and conformal prediction intervals."""

import importlib.util
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.config import (
    CATEGORICAL_FEATURES,
    INTERVAL_COVERAGE,
    LAG_COLUMN,
    MIN_CONDITIONAL_CALIBRATION_ROWS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def conformal_quantile(
    scores: np.ndarray, coverage: float = INTERVAL_COVERAGE
) -> float:
    """Finite-sample split-conformal quantile for absolute residual scores."""
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("No hay residuos válidos para calibrar el intervalo.")
    rank = min(math.ceil((clean.size + 1) * coverage), clean.size)
    return float(np.partition(clean, rank - 1)[rank - 1])


def calibrate_prediction_intervals(
    actual: pd.Series,
    predicted: np.ndarray,
    lag: pd.Series,
    coverage: float = INTERVAL_COVERAGE,
) -> dict[str, Any]:
    """Calibrate global and production-regime split-conformal intervals."""
    calibration = pd.DataFrame(
        {"score": np.abs(actual.to_numpy() - predicted), "lag": lag.to_numpy()}
    ).dropna()
    global_radius = conformal_quantile(calibration["score"].to_numpy(), coverage)
    edges = np.unique(
        calibration["lag"].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy(dtype=float)
    )
    groups: list[dict[str, Any]] = []
    if len(edges) >= 2:
        edges[0], edges[-1] = -np.inf, np.inf
        bins = pd.cut(calibration["lag"], edges, include_lowest=True)
        for interval, part in calibration.groupby(bins, observed=True):
            if len(part) < MIN_CONDITIONAL_CALIBRATION_ROWS:
                continue
            groups.append(
                {
                    "lower_lag": None
                    if np.isneginf(interval.left)
                    else float(interval.left),
                    "upper_lag": None
                    if np.isposinf(interval.right)
                    else float(interval.right),
                    "radius": conformal_quantile(part["score"].to_numpy(), coverage),
                    "records": int(len(part)),
                }
            )
    return {
        "method": "mondrian_split_conformal_absolute_residual",
        "coverage_target": coverage,
        "global_radius": global_radius,
        "conditional_feature": LAG_COLUMN,
        "groups": groups,
        "calibration_records": int(len(calibration)),
    }


def interval_radii(lag: pd.Series, interval: dict[str, Any]) -> np.ndarray:
    """Return the matching conditional radius, falling back to the global one."""
    radii = np.full(len(lag), float(interval["global_radius"]))
    values = lag.to_numpy(dtype=float)
    for group in interval.get("groups", []):
        lower = -np.inf if group["lower_lag"] is None else group["lower_lag"]
        upper = np.inf if group["upper_lag"] is None else group["upper_lag"]
        radii[(values > lower) & (values <= upper)] = group["radius"]
    return radii


def _one_hot_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def _ordinal_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_candidates(include_xgboost: bool = False) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "random_forest": Pipeline(
            [
                ("preprocessor", _one_hot_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=15,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", _ordinal_preprocessor()),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        max_iter=200, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
    }
    if include_xgboost:
        if importlib.util.find_spec("xgboost") is None:
            raise RuntimeError("Instalá xgboost para usar --include-xgboost.")
        from xgboost import XGBRegressor

        candidates["xgboost"] = Pipeline(
            [
                ("preprocessor", _one_hot_preprocessor()),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return candidates
