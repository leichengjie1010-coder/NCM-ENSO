"""Read the plotting interface shared by training and the figure tables."""
from __future__ import annotations

import csv
from pathlib import Path

TABLES = {
    "acc": {"model", "lead", "ACC"},
    "heatmap": {"model", "initial_month", "lead", "ACC", "n"},
    "events": {"event", "model", "target_month", "prediction_C"},
    "event_mae": {"event", "model", "MAE_C", "rank", "n"},
    "forecast2026": {"model", "target_month", "prediction_C"},
}


def load_tables(results: Path, reference: Path):
    if results.exists() and not results.is_dir():
        raise ValueError(f"Result path is not a directory: {results}")
    populated = results.is_dir() and any(results.iterdir())
    folder = results if populated else reference
    missing = [f"{name}.csv" for name in TABLES
               if not (folder / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete result directory: {folder}. Missing: {', '.join(missing)}. "
            "Complete the training run or use the supplied figure_data directory."
        )
    tables = {}
    for name, columns in TABLES.items():
        with (folder / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not columns.issubset(reader.fieldnames or ()):
                required = ", ".join(sorted(columns))
                raise ValueError(f"{name}.csv needs columns: {required}")
            tables[name] = list(reader)
        if not tables[name]:
            raise ValueError(f"{name}.csv contains no results")
    return tables
