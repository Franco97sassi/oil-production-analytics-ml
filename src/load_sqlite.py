import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "produccion.csv"
DB_PATH = BASE_DIR / "data" / "produccion.db"
SCHEMA_PATH = BASE_DIR / "sql" / "schema.sql"
VIEWS_PATH = BASE_DIR / "sql" / "views.sql"


def main():
    print("Leyendo CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print(f"Filas cargadas: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        print("Creando schema...")
        cursor.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        print("Insertando datos...")
        df.to_sql(
            "produccion",
            connection,
            if_exists="append",
            index=False,
        )

        print("Creando views...")
        cursor.executescript(VIEWS_PATH.read_text(encoding="utf-8"))

        connection.commit()

        total = cursor.execute(
            "SELECT COUNT(*) FROM produccion"
        ).fetchone()[0]

        print(f"Registros en SQLite: {total:,}")
        print(f"Base creada correctamente en: {DB_PATH}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()