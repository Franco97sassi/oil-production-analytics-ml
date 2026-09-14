import argparse

import joblib
import numpy as np
import pandas as pd
import pytest

from src.train import (
    calibrate_prediction_intervals,
    interval_radii,
    load_and_prepare,
    temporal_split,
    temporal_validation_split,
    train,
)


def test_lag_is_only_created_for_consecutive_months(tmp_path):
    rows = []
    for year, month, production in [(2024, 12, 10), (2025, 1, 12), (2025, 3, 20)]:
        rows.append(
            {
                "anio": year, "mes": month, "idpozo": "A",
                "prod_pet": production, "iny_agua": 0, "iny_gas": 0,
                "tef": 30, "tipoextraccion": "BM", "tipoestado": "activo",
                "tipopozo": "petrolifero", "provincia": "Neuquén",
                "cuenca": "Neuquina",
            }
        )
    path = tmp_path / "production.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    prepared = load_and_prepare(path)

    assert len(prepared) == 1
    assert prepared.iloc[0]["fecha"] == pd.Timestamp("2025-01-01")
    assert prepared.iloc[0]["prod_pet_lag1"] == 10


def test_duplicate_well_month_is_rejected_instead_of_creating_a_false_lag(tmp_path):
    row = {
        "anio": 2024, "mes": 1, "idpozo": "A", "prod_pet": 10,
        "iny_agua": 0, "iny_gas": 0, "tef": 30,
        "tipoextraccion": "BM", "tipoestado": "activo",
        "tipopozo": "petrolifero", "provincia": "Neuquén",
        "cuenca": "Neuquina",
    }
    path = tmp_path / "duplicates.csv"
    pd.DataFrame([row, row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicados"):
        load_and_prepare(path)


def test_temporal_split_uses_complete_dates():
    frame = pd.DataFrame(
        {"fecha": pd.date_range("2024-11-01", periods=5, freq="MS"), "value": range(5)}
    )

    train, test, cutoff = temporal_split(frame, test_months=2)

    assert cutoff == pd.Timestamp("2025-02-01")
    assert train["fecha"].max() == pd.Timestamp("2025-01-01")
    assert test["fecha"].min() == cutoff


def test_four_way_temporal_split_has_disjoint_ordered_periods():
    frame = pd.DataFrame(
        {"fecha": pd.date_range("2023-01-01", periods=13, freq="MS")}
    )

    training, validation, calibration, test, cutoffs = temporal_validation_split(
        frame, validation_months=2, calibration_months=2, test_months=2
    )

    assert training["fecha"].max() < validation["fecha"].min()
    assert validation["fecha"].max() < calibration["fecha"].min()
    assert calibration["fecha"].max() < test["fecha"].min()
    assert cutoffs == {
        "validation": pd.Timestamp("2023-08-01"),
        "calibration": pd.Timestamp("2023-10-01"),
        "test": pd.Timestamp("2023-12-01"),
    }


def test_conditional_intervals_use_production_regime_when_enough_rows():
    lag = pd.Series(np.arange(400, dtype=float))
    actual = pd.Series(np.r_[np.ones(200), np.full(200, 20.0)])
    interval = calibrate_prediction_intervals(actual, np.zeros(400), lag)

    radii = interval_radii(pd.Series([10.0, 390.0]), interval)

    assert interval["method"] == "mondrian_split_conformal_absolute_residual"
    assert len(interval["groups"]) == 4
    assert radii[0] == 1
    assert radii[1] == 20


def test_training_writes_auditable_bundle_from_disjoint_periods(tmp_path):
    rows = []
    for well_index in range(4):
        for date in pd.date_range("2023-01-01", periods=13, freq="MS"):
            rows.append(
                {
                    "anio": date.year,
                    "mes": date.month,
                    "idpozo": f"P-{well_index}",
                    "prod_pet": 20 + well_index + date.month,
                    "iny_agua": 1,
                    "iny_gas": 0,
                    "tef": 28,
                    "tipoextraccion": "BM",
                    "tipoestado": "activo",
                    "tipopozo": "petrolifero",
                    "provincia": "Neuquén",
                    "cuenca": "Neuquina",
                }
            )
    data_path = tmp_path / "production.csv"
    model_path = tmp_path / "model.joblib"
    report_dir = tmp_path / "reports"
    published_metrics_path = tmp_path / "published" / "strict_metrics.json"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    args = argparse.Namespace(
        data=data_path,
        model=model_path,
        report_dir=report_dir,
        validation_months=2,
        calibration_months=2,
        test_months=2,
        include_xgboost=False,
        with_shap=False,
        publish_metrics=published_metrics_path,
    )

    report = train(args)
    bundle = joblib.load(model_path)

    assert report["selection_metric"] == "validation_mae"
    assert report["test_period"] == ["2023-12-01", "2024-01-01"]
    assert bundle["prediction_interval"]["calibration_records"] == 8
    assert bundle["metadata"]["prediction_semantics"].startswith("one_step_ahead")
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "holdout_predictions.csv").exists()
    assert published_metrics_path.exists()
    assert report["dataset"]["sha256"]
    assert report["records_by_split"] == {
        "train": 24, "validation": 8, "calibration": 8, "test": 8,
    }
