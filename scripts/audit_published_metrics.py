"""Verify that committed portfolio metrics match stored notebook outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "docs" / "metrics" / "notebook_metrics.json"


def output_text(cell: dict[str, Any]) -> str:
    """Return plain-text representations saved in a notebook code cell."""
    parts: list[str] = []
    for output in cell.get("outputs", []):
        text = output.get("text")
        if text:
            parts.extend(text if isinstance(text, list) else [text])
        plain = output.get("data", {}).get("text/plain")
        if plain:
            parts.extend(plain if isinstance(plain, list) else [plain])
    return "".join(parts)


def parse_three_metrics(text: str) -> dict[str, float]:
    """Extract the final MAE/RMSE/R2 triplet printed by a notebook cell."""
    matches = re.findall(
        r"MAE:\s+([0-9.]+).*?RMSE:\s+([0-9.]+).*?R²:\s+([0-9.]+)",
        text,
        flags=re.DOTALL,
    )
    if not matches:
        raise ValueError("No se encontró una salida MAE/RMSE/R² auditable.")
    mae, rmse, r2 = matches[-1]
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def parse_monthly_metrics(text: str) -> list[dict[str, float | int]]:
    """Extract the three month rows from the saved plain-text DataFrame."""
    rows = re.findall(
        r"^\s*(10|11|12)\s+([0-9]+)\.0\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if len(rows) != 3:
        raise ValueError("Se esperaban exactamente tres filas mensuales auditables.")
    return [
        {"month": int(month), "records": int(records), "mae": float(mae),
         "rmse": float(rmse), "r2": float(r2)}
        for month, records, mae, rmse, r2 in rows
    ]


def audit() -> None:
    published = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    provenance = published["provenance"]
    notebook_path = PROJECT_ROOT / provenance["source"]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    indexes = provenance["source_cells"]
    observed = {
        "without_history": parse_three_metrics(
            output_text(cells[indexes["without_history"]])
        ),
        "with_lag1": parse_three_metrics(output_text(cells[indexes["with_lag1"]])),
        "by_month": parse_monthly_metrics(output_text(cells[indexes["by_month"]])),
    }
    expected = {key: published[key] for key in observed}
    if observed != expected:
        raise AssertionError(
            "Las métricas publicadas no coinciden con las salidas del notebook:\n"
            f"esperadas={expected}\nobservadas={observed}"
        )


if __name__ == "__main__":
    audit()
    print(f"OK: métricas verificadas contra {METRICS_PATH.relative_to(PROJECT_ROOT)}")
