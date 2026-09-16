"""Shared paths, feature definitions, and training constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "produccion.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "modelo_produccion_petroleo.joblib"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_PUBLISHED_METRICS_PATH = (
    PROJECT_ROOT / "docs" / "metrics" / "strict_temporal_metrics.json"
)

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
    "anio",
    "mes",
    ID_COLUMN,
    TARGET,
    *OPERATIONAL_COLUMNS,
    *CATEGORICAL_FEATURES,
]
RANDOM_STATE = 42
INTERVAL_COVERAGE = 0.90
MIN_CONDITIONAL_CALIBRATION_ROWS = 100
