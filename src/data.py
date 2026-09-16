"""Dataset validation, feature preparation, and temporal splitting."""

from pathlib import Path

import pandas as pd

from src.config import (
    DATE_COLUMN,
    ID_COLUMN,
    LAG_COLUMN,
    OPERATIONAL_COLUMNS,
    OPERATIONAL_LAG_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET,
)


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
    for source, lagged in zip(
        OPERATIONAL_COLUMNS, OPERATIONAL_LAG_COLUMNS, strict=True
    ):
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
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp]
]:
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
        (frame[DATE_COLUMN] >= calibration_cutoff) & (frame[DATE_COLUMN] < test_cutoff)
    ].copy()
    test_frame = frame[frame[DATE_COLUMN] >= test_cutoff].copy()
    return (
        train_frame,
        validation_frame,
        calibration_frame,
        test_frame,
        {
            "validation": validation_cutoff,
            "calibration": calibration_cutoff,
            "test": test_cutoff,
        },
    )
