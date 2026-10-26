"""Fit the EOF–Markov backbone and neural correction from prepared fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import config

DATA_FILE = config.DATA_FILE
FORECAST_FILE = config.FORECAST_FILE
OUTPUT_DIR = config.OUTPUT_DIR


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--forecast-data", type=Path, default=FORECAST_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mode", choices=("fit", "predict"), default="predict")
    parser.add_argument("--reuse-linear", type=Path, default=None)
    parser.add_argument("--forecast-origin", default=config.FORECAST_ORIGIN)
    args = parser.parse_args()
    if args.forecast_origin != "2026-08":
        parser.error(
            "The paper forecast comparison uses the August 2026 window. "
            "Other origins need matching external comparison tables."
        )
    if args.reuse_linear is None and not args.data.is_file():
        parser.error(
            "Set DATA_FILE at the top of train.py or pass "
            "--data path/to/processed_fields.nc"
        )
    if args.reuse_linear is not None:
        bank = args.reuse_linear / "network_inputs_366.npz"
        if not bank.is_file():
            parser.error(
                "--reuse-linear must name a completed linear output folder "
                "containing network_inputs_366.npz"
            )
    if args.forecast_data is not None and not args.forecast_data.is_file():
        parser.error("--forecast-data must name an existing prepared-field file")
    return args


def main():
    args = arguments()
    from model import linear, neural
    from model.evaluation import (
        dates_of, export_results, read_bank,
    )
    import numpy as np
    import torch

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    linear_dir = args.reuse_linear or args.output / "linear"
    if args.reuse_linear is None:
        linear.fit_pipeline(args.data, linear_dir)
    bank = read_bank(linear_dir / "network_inputs_366.npz")

    if args.mode == "fit":
        net, losses = neural.fit(bank)
        (args.output / "training_loss.json").write_text(
            json.dumps(losses, indent=2) + "\n"
        )
    else:
        net = neural.load_model()
    neural.save_checkpoint(args.output / "model.npz", net)
    future_values = None
    forecast_mode = "packaged reference forecast"
    if args.forecast_data is not None:
        forecast_dir = args.output / "forecast_inputs"
        linear.project_inputs(args.forecast_data, linear_dir, forecast_dir)
        event = read_bank(forecast_dir / "network_inputs_366.npz")
        dates = dates_of(event)
        where = np.flatnonzero(
            dates[event["all_origins"]] == args.forecast_origin
        )
        if len(where) != 1:
            raise ValueError("The forecast input window lacks the chosen origin")
        index = int(where[0])
        raw = event["all_x"][[index]]
        prediction = neural.predict(net, raw)[0]
        offset = event["common_offset_C"]
        future_values = {}
        for lead in range(1, 7):
            target = str(
                np.datetime64(args.forecast_origin, "M")
                + np.timedelta64(lead, "M")
            )
            month = int(target[5:7])
            for name, values in (
                ("NCM", prediction),
                ("MMarkov-4", event["all_markov"][index]),
                ("MMarkov-3", event["all_markov3"][index]),
            ):
                future_values[name, target] = float(
                    values[lead-1] + offset[month-1]
                )
        forecast_mode = "computed from the supplied forecast fields"
    metrics = export_results(bank, net, args.output, future=future_values)
    run = {
        "mode": args.mode,
        "architecture": {
            "stored_inputs": 366, "active_inputs": 324,
            "summary_inputs": 29, "hidden_units": 4,
            "forecast_months": 20, "parameters": 17843,
        },
        "evaluation": "2013-2024 development period",
        "forecast": forecast_mode,
        "metrics": metrics,
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
