"""Skill tables and monthly event results shared by training and plotting."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .neural import predict, score_predictions

REFERENCE = Path(__file__).resolve().parents[1] / "figure_data"
LOCAL_MODELS = ("NCM", "MMarkov-4", "MMarkov-3")


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_bank(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def dates_of(bank):
    return bank["dates"].astype("datetime64[M]").astype(str)


def forecast_arrays(bank, net):
    positions = np.searchsorted(bank["all_origins"], bank["val_origins"])
    np.testing.assert_array_equal(
        bank["all_origins"][positions], bank["val_origins"]
    )
    return {
        "NCM": predict(net, bank["val_x"]),
        "MMarkov-4": bank["val_markov"],
        "MMarkov-3": bank["all_markov3"][positions],
    }


def smooth_trajectories(values):
    result = values.copy()
    result[:, 1:-1] = (values[:, :-2] + values[:, 1:-1] + values[:, 2:]) / 3
    return result


def skill_tables(bank, forecasts):
    dates = dates_of(bank)
    start = int(np.flatnonzero(dates == "2013-01")[0])
    end = int(np.flatnonzero(dates == "2024-12")[0]) + 1
    cfg = {"validation_start": start, "validation_end_exclusive": end}
    acc = [
        row for row in read_rows(REFERENCE / "acc.csv")
        if row["model"] not in LOCAL_MODELS
    ]
    scores, heatmap = [], []
    observation = bank["obs"][start:end].copy()
    observation[1:-1] = (
        bank["obs"][start:end-2]
        + bank["obs"][start+1:end-1]
        + bank["obs"][start+2:end]
    ) / 3
    origins = bank["val_origins"]
    initial_month = np.array([int(date[5:7]) for date in dates[origins]])
    for model, prediction in forecasts.items():
        metric_rows = score_predictions(prediction, origins, bank["obs"], cfg)
        for row in metric_rows:
            if row["metric"] != "three_month":
                continue
            scores.append({"model": model, **row})
            if row["lead"] <= 8:
                acc.append({
                    "model": model, "lead": row["lead"], "ACC": row["ACC"]
                })
        smoothed = smooth_trajectories(prediction)
        for lead in range(1, 9):
            targets = origins + lead
            valid = (targets >= start) & (targets < end)
            for month in range(1, 13):
                pick = valid & (initial_month == month)
                p = smoothed[pick, lead-1]
                y = observation[targets[pick]-start]
                value = float(np.corrcoef(p, y)[0, 1]) if len(p) > 2 else np.nan
                heatmap.append({
                    "model": model, "initial_month": month, "lead": lead,
                    "ACC": value, "n": len(p),
                })
    return acc, heatmap, scores


def event_tables(bank, forecasts):
    cohort = {
        (int(row["event"]), row["model"])
        for row in read_rows(REFERENCE / "event_mae.csv")
    }
    rows = [
        row for row in read_rows(REFERENCE / "events.csv")
        if row["model"] == "ORAS5"
        or (int(row["event"]), row["model"]) in cohort
    ]
    dates, origins = dates_of(bank), bank["val_origins"]
    offset = bank["common_offset_C"]
    for event in sorted({int(row["event"]) for row in rows}):
        where = np.flatnonzero(dates[origins] == f"{event}-08")
        if len(where) != 1:
            raise ValueError(f"The evaluation inputs lack August {event}")
        index = int(where[0])
        origin = int(origins[index])
        for row in rows:
            if int(row["event"]) != event:
                continue
            target = np.datetime64(row["target_month"], "M")
            lead = int((target - np.datetime64(f"{event}-08")).astype(int))
            month = int(row["target_month"][5:7])
            model = row["model"]
            if model in LOCAL_MODELS:
                row["prediction_C"] = float(
                    forecasts[model][index, lead-1] + offset[month-1]
                )
            elif model == "ORAS5":
                row["prediction_C"] = float(
                    bank["obs"][origin+lead] + offset[month-1]
                )
    rankings = []
    for event in sorted({int(row["event"]) for row in rows}):
        group = [row for row in rows if int(row["event"]) == event]
        models = list(dict.fromkeys(row["model"] for row in group))
        months = sorted({row["target_month"] for row in group})
        arrays = {
            model: np.array([
                float(next(row["prediction_C"] for row in group
                           if row["model"] == model
                           and row["target_month"] == month) or "nan")
                for month in months
            ])
            for model in models
        }
        valid = np.logical_and.reduce([
            np.isfinite(values) for values in arrays.values()
        ])
        ranked = [{
            "event": event, "model": model,
            "MAE_C": float(np.abs(values[valid]-arrays["ORAS5"][valid]).mean()),
            "rank": 0, "n": int(valid.sum()), "model_count": len(models)-1,
        } for model, values in arrays.items() if model != "ORAS5"]
        ranked.sort(key=lambda row: (row["MAE_C"], row["model"]))
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        rankings.extend(ranked)
    return rows, rankings


def key_metrics(bank, net):
    forecasts = forecast_arrays(bank, net)
    _, _, scores = skill_tables(bank, forecasts)
    _, events = event_tables(bank, forecasts)
    curve = [row for row in scores if row["model"] == "NCM"]
    metrics = {
        "mean_ACC_1_6": float(np.mean([row["ACC"] for row in curve[:6]])),
        "ACC6": curve[5]["ACC"],
    }
    for row in events:
        if row["model"] == "NCM":
            year = row["event"]
            metrics[f"MAE{year}"] = row["MAE_C"]
            metrics[f"rank{year}"] = row["rank"]
    return metrics


def export_results(bank, net, destination, future=None):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    forecasts = forecast_arrays(bank, net)
    acc, heatmap, scores = skill_tables(bank, forecasts)
    events, rankings = event_tables(bank, forecasts)
    future_rows = read_rows(REFERENCE / "forecast2026.csv")
    if future is not None:
        required = {
            (row["model"], row["target_month"])
            for row in future_rows if row["model"] in LOCAL_MODELS
        }
        if not required.issubset(future):
            raise ValueError(
                "The forecast comparison needs the six target months "
                "September 2026 through February 2027 for NCM, MM4 and MM3"
            )
        for row in future_rows:
            if row["model"] in LOCAL_MODELS:
                row["prediction_C"] = future[
                    row["model"], row["target_month"]
                ]
    for name, table in (
        ("acc.csv", acc), ("heatmap.csv", heatmap),
        ("events.csv", events), ("event_mae.csv", rankings),
        ("forecast2026.csv", future_rows), ("metrics.csv", scores),
    ):
        write_rows(out / name, table)
    np.savez_compressed(
        out / "predictions.npz", origins=bank["val_origins"],
        NCM=forecasts["NCM"], MMarkov4=forecasts["MMarkov-4"],
        MMarkov3=forecasts["MMarkov-3"],
    )
    return key_metrics(bank, net)
