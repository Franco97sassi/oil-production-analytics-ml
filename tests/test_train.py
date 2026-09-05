import pandas as pd

from src.train import load_and_prepare, temporal_split


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


def test_temporal_split_uses_complete_dates():
    frame = pd.DataFrame(
        {"fecha": pd.date_range("2024-11-01", periods=5, freq="MS"), "value": range(5)}
    )

    train, test, cutoff = temporal_split(frame, test_months=2)

    assert cutoff == pd.Timestamp("2025-02-01")
    assert train["fecha"].max() == pd.Timestamp("2025-01-01")
    assert test["fecha"].min() == cutoff
