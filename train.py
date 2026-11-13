"""Run final-weight replay, constrained development fitting, or NCM inference."""
from __future__ import annotations

import argparse
import json
import hashlib
import os
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
    parser.add_argument(
        "--mode", choices=("predict", "fit-final", "fit", "continue-fit", "replay-final", "verify-recipe"),
        default="predict",
        help="fit/fit-final optimize the selected development recipe; continue-fit is extra historical training; replay-final exactly reconstructs selected weights",
    )
    parser.add_argument("--reuse-linear", type=Path, default=None)
    parser.add_argument("--forecast-origin", default=config.FORECAST_ORIGIN)
    args = parser.parse_args()
    if args.mode == "fit":
        args.mode = "fit-final"
    if args.mode in ("replay-final", "verify-recipe"):
        return args
    if args.mode == "fit-final" and args.reuse_linear is None:
        parser.error("fit-final requires --reuse-linear with the original matching feature bank; use replay-final for exact reconstruction without data")
    if args.forecast_origin != "2026-09":
        parser.error(
            "The paper forecast comparison uses September 2026 initialization with complete August inputs. "
            "Other origins need matching external comparison tables."
        )
    if args.reuse_linear is None and not args.data.is_file():
        parser.error(
            "Set DATA_FILE in config.py or pass "
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


def verify_data_repository():
    """Require the mounted canonical data volume before data-dependent work."""
    catalogue = config.DATA_ROOT / "DATA_INDEX.json"
    if not os.path.ismount(config.DATA_ROOT.parent) or not catalogue.is_file():
        raise RuntimeError(
            "The canonical Data drive is not mounted. Data-dependent work stopped; "
            "replay-final and verify-recipe do not require climate data."
        )
    document = json.loads(catalogue.read_text())

    if str(document.get("volume_uuid", "")).upper() != config.DATA_VOLUME_UUID.upper():
        raise RuntimeError("DATA_INDEX.json does not report the expected Data volume UUID")
    return {"catalogue": str(catalogue), "volume_uuid": config.DATA_VOLUME_UUID,
            "catalogue_sha256": hashlib.sha256(catalogue.read_bytes()).hexdigest()}


def main():
    args = arguments()
    if args.mode in ("replay-final", "verify-recipe"):
        from model import final_recipe
        result = (final_recipe.verify_recipe(config.RECIPE_DIR) if args.mode == "verify-recipe"
                  else final_recipe.replay(args.output, config.RECIPE_DIR))
        print(json.dumps(result, indent=2))
        return
    data_guard = verify_data_repository()
    from model import linear, neural
    from model.evaluation import (
        dates_of, export_results, read_bank, output_offsets,
    )
    import numpy as np
    import torch

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    linear_dir = args.reuse_linear or args.output / "linear"
    if args.reuse_linear is None:
        linear.fit_pipeline(args.data, linear_dir)
    bank = read_bank(linear_dir / "network_inputs_366.npz")

    training_run = None
    if args.mode == "fit-final":
        from model import final_recipe
        net, training_run = final_recipe.fit(
            bank, linear_dir / "network_inputs_366.npz", args.output, config.RECIPE_DIR,
        )
    elif args.mode == "continue-fit":
        net, losses = neural.continue_fit(bank)
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
        input_month = str(np.datetime64(args.forecast_origin, "M") - np.timedelta64(1, "M"))
        where = np.flatnonzero(
            dates[event["all_origins"]] == input_month
        )
        if len(where) != 1:
            raise ValueError("The forecast input window lacks the chosen origin")
        index = int(where[0])
        raw = event["all_x"][[index]]
        prediction = neural.predict(net, raw)[0]
        offset = output_offsets(event)
        future_values = {}
        for lead in range(1, 7):
            target = str(
                np.datetime64(input_month, "M")
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
        "selected_final_model_sha256": config.FINAL_MODEL_SHA256,
        "output_checkpoint_sha256": hashlib.sha256((args.output / "model.npz").read_bytes()).hexdigest(),
        "data_repository_guard": data_guard,
        "training_recipe": training_run,
        "architecture": {
            "stored_inputs": 366, "active_inputs": 324,
            "summary_inputs": 29, "hidden_units": 4,
            "forecast_months": 20, "parameters": 17843,
        },
        "evaluation": "2013-2024 development period",
        "independent_test_claim": False,
        "forecast": forecast_mode,
        "named_initialization_month": args.forecast_origin,
        "last_complete_input_month": str(np.datetime64(args.forecast_origin, "M") - np.timedelta64(1, "M")),
        "metrics": metrics,
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
