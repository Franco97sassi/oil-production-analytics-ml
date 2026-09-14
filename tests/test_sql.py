import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def build_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript((ROOT / "sql/schema.sql").read_text())
    connection.executescript((ROOT / "sql/views.sql").read_text())
    return connection


def test_schema_has_surrogate_key_constraints_and_analytical_indexes():
    connection = build_database()
    indexes = {
        row[1] for row in connection.execute("PRAGMA index_list('produccion')")
    }

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("INSERT INTO produccion (anio, mes) VALUES (2024, 13)")

    assert "idx_produccion_pozo_periodo" in indexes
    assert "idx_produccion_periodo" in indexes


def test_data_quality_view_reports_duplicate_well_months():
    connection = build_database()
    connection.executemany(
        "INSERT INTO produccion (anio, mes, idpozo, prod_pet, tef) VALUES (?, ?, ?, ?, ?)",
        [(2024, 1, "A", 10, 31), (2024, 1, "A", 11, 31)],
    )

    row = connection.execute(
        "SELECT cantidad_registros, cantidad_pozos, posibles_duplicados_pozo_mes "
        "FROM vw_calidad_datos_mensual"
    ).fetchone()

    assert row == (2, 1, 1)
