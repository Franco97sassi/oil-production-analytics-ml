"""Offline training and evaluation for the oil-production model.

The API never trains a model.  Run this module after placing the public CSV at
``data/produccion.csv``; it writes the deployable bundle and auditable reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "produccion.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "modelo_produccion_petroleo.joblib"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

TARGET = "prod_pet"
ID_COLUMN = "idpozo"
DATE_COLUMN = "fecha"
LAG_COLUMN = "prod_pet_lag1"
OPERATIONAL_COLUMNS = ["iny_agua", "iny_gas", "tef"]
OPERATIONAL_LAG_COLUMNS = [f"{column}_lag1" for column in OPERATIONAL_COLUMNS]
NUMERIC_FEATURES = ["mes", *OPERATIONAL_LAG_COLUMNS, LAG_COLUMN]
CATEGORICAL_FEATURES = [
    "tipoextraccion",
    "tipoestado",
    "tipopozo",
    "provincia",
    "cuenca",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [
    "anio", "mes", ID_COLUMN, TARGET, *OPERATIONAL_COLUMNS,
    *CATEGORICAL_FEATURES,
]
RANDOM_STATE = 42
INTERVAL_COVERAGE = 0.90
MIN_CONDITIONAL_CALIBRATION_ROWS = 100


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load data and construct a strictly consecutive, well-level monthly lag."""
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")

    frame = frame.copy()
    frame[DATE_COLUMN] = pd.to_datetime(
        {"year": frame["anio"], "month": frame["mes"], "day": 1},
        errors="coerce",
    )
    frame = frame.dropna(subset=[DATE_COLUMN, ID_COLUMN, TARGET])
    duplicates = frame.duplicated([ID_COLUMN, DATE_COLUMN], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, [ID_COLUMN, DATE_COLUMN]].head(5)
        raise ValueError(
            "Hay registros duplicados para el mismo pozo y mes; resolvé las "
            f"revisiones antes de entrenar. Ejemplos: {examples.to_dict('records')}"
        )
    frame = frame.sort_values([ID_COLUMN, DATE_COLUMN])

    previous_date = frame.groupby(ID_COLUMN, sort=False)[DATE_COLUMN].shift(1)
    previous_production = frame.groupby(ID_COLUMN, sort=False)[TARGET].shift(1)
    consecutive = frame[DATE_COLUMN].eq(previous_date + pd.offsets.MonthBegin(1))
    frame[LAG_COLUMN] = previous_production.where(consecutive)
    for source, lagged in zip(OPERATIONAL_COLUMNS, OPERATIONAL_LAG_COLUMNS):
        previous_value = frame.groupby(ID_COLUMN, sort=False)[source].shift(1)
        frame[lagged] = previous_value.where(consecutive)

    frame = frame[(frame[TARGET] >= 0) & frame[LAG_COLUMN].notna()].copy()
    if frame.empty:
        raise ValueError("No quedaron observaciones con un mes anterior consecutivo.")
    return frame


def temporal_split(
    frame: pd.DataFrame, test_months: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Reserve the final N observed calendar months as a true future holdout."""
    months = pd.Index(frame[DATE_COLUMN].dropna().sort_values().unique())
    if test_months < 1 or len(months) <= test_months:
        raise ValueError("Se necesitan más meses observados que meses de test.")
    cutoff = pd.Timestamp(months[-test_months])
    train = frame[frame[DATE_COLUMN] < cutoff].copy()
    test = frame[frame[DATE_COLUMN] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("El corte temporal produjo un conjunto vacío.")
    return train, test, cutoff


def temporal_validation_split(
    frame: pd.DataFrame,
    validation_months: int = 3,
    calibration_months: int = 3,
    test_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]]:
    """Create disjoint train, model-selection, calibration and final-test periods."""
    sizes = {
        "validation_months": validation_months,
        "calibration_months": calibration_months,
        "test_months": test_months,
    }
    if any(value < 1 for value in sizes.values()):
        raise ValueError("Cada período temporal debe contener al menos un mes.")
    months = pd.Index(frame[DATE_COLUMN].dropna().sort_values().unique())
    reserved = sum(sizes.values())
    if len(months) <= reserved:
        raise ValueError(
            "Se necesitan más meses observados que la suma de validation, "
            "calibration y test."
        )
    validation_cutoff = pd.Timestamp(months[-reserved])
    calibration_cutoff = pd.Timestamp(months[-(calibration_months + test_months)])
    test_cutoff = pd.Timestamp(months[-test_months])
    train_frame = frame[frame[DATE_COLUMN] < validation_cutoff].copy()
    validation_frame = frame[
        (frame[DATE_COLUMN] >= validation_cutoff)
        & (frame[DATE_COLUMN] < calibration_cutoff)
    ].copy()
    calibration_frame = frame[
        (frame[DATE_COLUMN] >= calibration_cutoff)
        & (frame[DATE_COLUMN] < test_cutoff)
    ].copy()
    test_frame = frame[frame[DATE_COLUMN] >= test_cutoff].copy()
    return train_frame, validation_frame, calibration_frame, test_frame, {
        "validation": validation_cutoff,
        "calibration": calibration_cutoff,
        "test": test_cutoff,
    }


def conformal_quantile(scores: np.ndarray, coverage: float = INTERVAL_COVERAGE) -> float:
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
                    "lower_lag": None if np.isneginf(interval.left) else float(interval.left),
                    "upper_lag": None if np.isposinf(interval.right) else float(interval.right),
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
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def _ordinal_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", categorical, CATEGORICAL_FEATURES)]
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


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "registros": int(len(actual)),
    }


def grouped_metrics(
    evaluation: pd.DataFrame, group: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, part in evaluation.groupby(group, observed=True, dropna=False):
        metrics = regression_metrics(part["real"], part["predicha"].to_numpy())
        rows.append({group: str(value), **metrics})
    return rows


def population_stability_index(train: pd.Series, test: pd.Series) -> float:
    """Compute numeric PSI with bins learned only from training data."""
    train = pd.to_numeric(train, errors="coerce").dropna()
    test = pd.to_numeric(test, errors="coerce").dropna()
    if train.empty or test.empty:
        return float("nan")
    edges = np.unique(train.quantile(np.linspace(0, 1, 11)).to_numpy())
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    train_share = pd.cut(train, edges, include_lowest=True).value_counts(normalize=True)
    test_share = pd.cut(test, edges, include_lowest=True).value_counts(normalize=True)
    train_share, test_share = train_share.align(test_share, fill_value=0)
    train_share = train_share.clip(lower=1e-6)
    test_share = test_share.clip(lower=1e-6)
    return float(((test_share - train_share) * np.log(test_share / train_share)).sum())


def drift_report(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    numeric = {
        column: population_stability_index(train[column], test[column])
        for column in NUMERIC_FEATURES
    }
    categorical: dict[str, float] = {}
    for column in CATEGORICAL_FEATURES:
        left = train[column].fillna("<NA>").value_counts(normalize=True)
        right = test[column].fillna("<NA>").value_counts(normalize=True)
        left, right = left.align(right, fill_value=0)
        categorical[column] = float(0.5 * (left - right).abs().sum())
    return {"numeric_psi": numeric, "categorical_total_variation": categorical}


def save_plots(evaluation: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    sample = evaluation.sample(min(5000, len(evaluation)), random_state=RANDOM_STATE)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(sample["real"], sample["predicha"], alpha=0.25, s=10)
    limit = float(max(sample["real"].max(), sample["predicha"].max()))
    axis.plot([0, limit], [0, limit], "--", color="black")
    axis.set(xlabel="Producción real (m³)", ylabel="Producción predicha (m³)",
             title="Producción real vs. predicha — holdout temporal")
    figure.tight_layout()
    figure.savefig(report_dir / "model_real_vs_predicted.png", dpi=160)
    plt.close(figure)

    monthly = evaluation.groupby("fecha", observed=True)[["real", "predicha"]].sum()
    figure, axis = plt.subplots(figsize=(9, 5))
    monthly.plot(ax=axis, marker="o")
    axis.set(xlabel="Mes", ylabel="Producción (m³)",
             title="Producción mensual en el holdout")
    figure.tight_layout()
    figure.savefig(report_dir / "model_monthly_holdout.png", dpi=160)
    plt.close(figure)


def save_shap_summary(model: Pipeline, sample: pd.DataFrame, report_dir: Path) -> str:
    if importlib.util.find_spec("shap") is None:
        return "omitido: instalá shap y ejecutá con --with-shap"
    import shap

    transformed = model.named_steps["preprocessor"].transform(sample)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    explainer = shap.TreeExplainer(model.named_steps["regressor"])
    values = explainer.shap_values(transformed)
    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(report_dir / "shap_importance.csv", index=False)
    return "reports/shap_importance.csv"


def train(args: argparse.Namespace) -> dict[str, Any]:
    frame = load_and_prepare(args.data)
    train_frame, validation_frame, calibration_frame, test_frame, cutoffs = (
        temporal_validation_split(
            frame, args.validation_months, args.calibration_months, args.test_months
        )
    )
    x_train, y_train = train_frame[FEATURES], train_frame[TARGET]
    x_validation = validation_frame[FEATURES]
    y_validation = validation_frame[TARGET]
    x_calibration = calibration_frame[FEATURES]
    y_calibration = calibration_frame[TARGET]
    x_test, y_test = test_frame[FEATURES], test_frame[TARGET]

    validation_predictions: dict[str, np.ndarray] = {
        "persistencia": x_validation[LAG_COLUMN].to_numpy(),
        "media": np.full(len(x_validation), float(y_train.mean())),
    }
    test_predictions: dict[str, np.ndarray] = {
        "persistencia": x_test[LAG_COLUMN].to_numpy(),
        "media": np.full(len(x_test), float(y_train.mean())),
    }
    fitted: dict[str, Any] = {}
    for name, candidate in build_candidates(args.include_xgboost).items():
        candidate.fit(x_train, y_train)
        fitted[name] = candidate
        validation_predictions[name] = candidate.predict(x_validation)

    validation_metrics = {
        name: regression_metrics(y_validation, prediction)
        for name, prediction in validation_predictions.items()
    }
    selected_name = min(fitted, key=lambda name: validation_metrics[name]["mae"])
    development_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    x_development = development_frame[FEATURES]
    y_development = development_frame[TARGET]
    for candidate in fitted.values():
        candidate.fit(x_development, y_development)
    selected_model = fitted[selected_name]
    calibration_prediction = selected_model.predict(x_calibration)
    prediction_interval = calibrate_prediction_intervals(
        y_calibration, calibration_prediction, x_calibration[LAG_COLUMN]
    )
    for name in validation_predictions:
        if name == "persistencia":
            test_predictions[name] = x_test[LAG_COLUMN].to_numpy()
        elif name == "media":
            test_predictions[name] = np.full(len(x_test), float(y_development.mean()))
        else:
            test_predictions[name] = fitted[name].predict(x_test)
    test_metrics = {
        name: regression_metrics(y_test, prediction)
        for name, prediction in test_predictions.items()
    }
    selected_prediction = test_predictions[selected_name]
    radii = interval_radii(x_test[LAG_COLUMN], prediction_interval)
    prediction_interval["test_coverage"] = float(
        np.mean((y_test.to_numpy() >= np.maximum(0, selected_prediction - radii))
                & (y_test.to_numpy() <= selected_prediction + radii))
    )
    known_wells = set(development_frame[ID_COLUMN])
    evaluation = test_frame[
        [DATE_COLUMN, ID_COLUMN, "provincia", "cuenca", TARGET]
    ].rename(columns={TARGET: "real"})
    evaluation["predicha"] = selected_prediction
    evaluation["estado_pozo"] = np.where(
        evaluation[ID_COLUMN].isin(known_wells), "conocido", "nuevo"
    )
    evaluation["rango_produccion"] = pd.cut(
        evaluation["real"],
        bins=[-np.inf, 10, 100, 500, np.inf],
        labels=["0–10", "10–100", "100–500", ">500"],
    )

    report = {
        "selected_model": selected_name,
        "selection_metric": "validation_mae",
        "cutoffs": {name: value.strftime("%Y-%m-%d") for name, value in cutoffs.items()},
        "train_period": [str(train_frame[DATE_COLUMN].min().date()),
                         str(train_frame[DATE_COLUMN].max().date())],
        "validation_period": [str(validation_frame[DATE_COLUMN].min().date()),
                              str(validation_frame[DATE_COLUMN].max().date())],
        "calibration_period": [str(calibration_frame[DATE_COLUMN].min().date()),
                               str(calibration_frame[DATE_COLUMN].max().date())],
        "test_period": [str(test_frame[DATE_COLUMN].min().date()),
                        str(test_frame[DATE_COLUMN].max().date())],
        "validation_metrics_by_model": validation_metrics,
        "metrics_by_model": test_metrics,
        "metrics_by_province": grouped_metrics(evaluation, "provincia"),
        "metrics_by_basin": grouped_metrics(evaluation, "cuenca"),
        "metrics_by_production_range": grouped_metrics(evaluation, "rango_produccion"),
        "metrics_new_vs_known_wells": grouped_metrics(evaluation, "estado_pozo"),
        "prediction_interval": prediction_interval,
        "drift": drift_report(development_frame, test_frame),
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    save_plots(evaluation, args.report_dir)
    if args.with_shap:
        report["shap"] = save_shap_summary(
            selected_model, x_test.sample(min(1000, len(x_test)), random_state=RANDOM_STATE),
            args.report_dir,
        )
    (args.report_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evaluation.to_csv(args.report_dir / "holdout_predictions.csv", index=False)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": selected_model,
            "prediction_interval": prediction_interval,
            "features": FEATURES,
            "metadata": {
                "selected_model": selected_name,
                "trained_until": str(validation_frame[DATE_COLUMN].max().date()),
                "validation_from": str(cutoffs["validation"].date()),
                "calibration_from": str(cutoffs["calibration"].date()),
                "test_from": str(cutoffs["test"].date()),
                "prediction_semantics": (
                    "one_step_ahead_monthly_forecast; every input must be known "
                    "before the forecast month starts"
                ),
            },
        },
        args.model,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--calibration-months", type=int, default=3)
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--with-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = train(parse_args())
    print(json.dumps(report["metrics_by_model"], ensure_ascii=False, indent=2))
    print(f"Modelo seleccionado: {report['selected_model']}")


if __name__ == "__main__":
    main()
