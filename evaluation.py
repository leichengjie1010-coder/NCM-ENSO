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


def output_offsets(bank):
    rows = sorted(read_rows(REFERENCE / "reference_climatology.csv"),
                  key=lambda row: int(row["calendar_month"]))
    reference = np.array([float(row["mean_SST_C"]) for row in rows])
    native_value = bank.get("nino_climatology_1958_2012")
    if native_value is None:
        native_value = bank["nino_climatology"]
    native = np.asarray(native_value)
    if native.shape != (12,) or reference.shape != (12,):
        raise ValueError("Expected twelve calendar-month climatological means")
    return native - reference


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
    origins = bank["val_origins"]
    offset = output_offsets(bank)
    observed = bank["obs"] + offset[np.arange(len(bank["obs"])) % 12]
    observation = observed[start:end].copy()
    observation[1:-1] = (
        observed[start:end-2]
        + observed[start+1:end-1]
        + observed[start+2:end]
    ) / 3
    initial_month = np.array([int(date[5:7]) % 12 + 1 for date in dates[origins]])
    targets_all = origins[:, None] + np.arange(1, 21)[None, :]
    for model, prediction in forecasts.items():
        display_prediction = prediction + offset[targets_all % 12]
        metric_rows = score_predictions(display_prediction, origins, observed, cfg)
        for row in metric_rows:
            if row["metric"] != "three_month":
                continue
            scores.append({"model": model, **row})
            if row["lead"] <= 8:
                acc.append({
                    "model": model, "lead": row["lead"], "ACC": row["ACC"]
                })
        smoothed = smooth_trajectories(display_prediction)
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


def event_tables(bank, forecasts, period="winter"):
    if period not in ("winter", "february"):
        raise ValueError("Expected winter or february event period")
    suffix = "" if period == "winter" else "_february"
    cohort = {
        (int(row["event"]), row["model"])
        for row in read_rows(REFERENCE / f"event_mae{suffix}.csv")
    }
    rows = [
        row for row in read_rows(REFERENCE / f"events{suffix}.csv")
        if row["model"] == "ORAS5"
        or (int(row["event"]), row["model"]) in cohort
    ]
    dates, origins = dates_of(bank), bank["val_origins"]
    offset = output_offsets(bank)
    for event in sorted({int(row["event"]) for row in rows}):
        input_month = f"{event}-08" if period == "winter" else f"{event}-01"
        where = np.flatnonzero(dates[origins] == input_month)
        if len(where) != 1:
            raise ValueError(f"The evaluation inputs lack {input_month}")
        index = int(where[0])
        origin = int(origins[index])
        for row in rows:
            if int(row["event"]) != event:
                continue
            target = np.datetime64(row["target_month"], "M")
            lead = int((target - np.datetime64(input_month, "M")).astype(int))
            if not 1 <= lead <= 6:
                raise ValueError(f"Unexpected {period} target month {target}")
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
        ranked = []
        if len(months) != 6 or not np.isfinite(arrays["ORAS5"]).all():
            raise ValueError(f"Incomplete {period} observation window for {event}")
        for model, values in arrays.items():
            if model == "ORAS5":
                continue
            valid = np.isfinite(values) & np.isfinite(arrays["ORAS5"])
            if period == "winter" and event in (2015, 2017):
                valid[0] = False
            expected_n = 5 if event in (2015, 2017) and (period == "winter" or model not in LOCAL_MODELS) else 6
            expected_mask = np.ones(6, dtype=bool)
            if expected_n == 5:
                expected_mask[0] = False
            if not np.array_equal(valid, expected_mask):
                raise ValueError(f"Incorrect {period} coverage for {event} {model}")
            errors = values[valid] - arrays["ORAS5"][valid]
            ranked.append({
                "event": event, "model": model,
                "MAE_C": float(np.abs(errors).mean()), "mean_bias_C": float(errors.mean()),
                "rank": 0, "n": int(valid.sum()), "model_count": len(models)-1,
            })
        ranked.sort(key=lambda row: (row["MAE_C"], row["model"]))
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        rankings.extend(ranked)
    return rows, rankings


def djf_scatter_table(bank, forecasts):
    dates = dates_of(bank)
    offsets = output_offsets(bank)
    rows = []
    for index, origin in enumerate(bank["val_origins"]):
        target = int(origin) + 6
        if not 0 <= target < len(bank["obs"]):
            continue
        target_month = dates[target]
        month = int(target_month[5:7])
        if not "2013-01" <= target_month <= "2024-12" or month not in (12, 1, 2):
            continue
        input_month = dates[int(origin)]
        rows.append({
            "target_month": target_month,
            "last_complete_input_month": input_month,
            "named_initialization_month": str(np.datetime64(input_month, "M") + 1),
            "prediction_C": float(forecasts["NCM"][index, 5] + offsets[month-1]),
            "observation_C": float(bank["obs"][target] + offsets[month-1]),
        })
    if not rows:
        raise ValueError("The evaluation period contains no DJF lead-6 pairs")
    return rows


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
    february_events, february_rankings = event_tables(bank, forecasts, period="february")
    djf_scatter = djf_scatter_table(bank, forecasts)
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
        ("events_february.csv", february_events), ("event_mae_february.csv", february_rankings),
        ("forecast2026.csv", future_rows), ("metrics.csv", scores),
        ("djf_scatter.csv", djf_scatter),
    ):
        write_rows(out / name, table)
    np.savez_compressed(
        out / "predictions.npz", origins=bank["val_origins"],
        NCM=forecasts["NCM"], MMarkov4=forecasts["MMarkov-4"],
        MMarkov3=forecasts["MMarkov-3"],
    )
    return key_metrics(bank, net)
