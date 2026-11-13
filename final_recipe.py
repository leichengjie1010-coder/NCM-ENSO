"""Reconstruct or optimize the selected NCM development recipe.

Parameter replay is exact and needs only NumPy. Optimizer fitting uses the
archived feature directions, supplied matching feature bank, and known event
observations. Neither operation is training from random weights.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np

RECIPE_DIR = Path(__file__).resolve().parents[1] / "training_recipe"


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _state(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files
                if key != "configuration"}


def _assert_state(state, expected):
    if set(state) != set(expected):
        raise ValueError("Checkpoint state members differ from the recipe")
    for name, value in state.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint member: {name}")
        actual = sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if actual != expected[name]:
            raise ValueError(f"Recipe numeric verification failed: {name}")


def load_recipe(directory=RECIPE_DIR):
    directory = Path(directory)
    recipe = json.loads((directory / "recipe.json").read_text())
    for name, expected in ((recipe["start_checkpoint"], recipe["start_checkpoint_sha256"]),
                           (recipe["updates_file"], recipe["updates_sha256"])):
        if _digest(directory / name) != expected:
            raise ValueError(f"Recipe file SHA-256 mismatch: {name}")
    with np.load(directory / recipe["updates_file"], allow_pickle=False) as archive:
        updates = {key: archive[key].copy() for key in archive.files}
    return recipe, updates


def reconstructed_state(directory=RECIPE_DIR):
    """Apply archived selected updates; never read predictors or observations."""
    recipe, updates = load_recipe(directory)
    state = _state(Path(directory) / recipe["start_checkpoint"])
    stages = []
    for stage in recipe["stages"]:
        prefix = stage["id"]
        direction = updates[prefix + "_direction"]
        coordinates = updates[prefix + "_coordinates"]
        if direction.shape != (366,) or coordinates.shape != (1, 6):
            raise ValueError(f"Invalid parameter replay shape: {prefix}")
        if np.any(direction[306:348] != 0):
            raise ValueError("Inactive-input direction must be zero")
        state["L0"][:, :6] += direction[:, None] @ coordinates
        _assert_state(state, stage["numeric_state_sha256"])
        stages.append({"stage": prefix, "selected_optimizer_step": stage["selected_step"],
                       "all_numeric_members_match": True})
    transfer = recipe["amplitude_transfer"]
    direction = updates["inherited_raw_hidden_direction"]
    if direction.shape != (4, 366) or np.any(direction[:, 306:348] != 0):
        raise ValueError("Invalid inherited amplitude direction")
    delta = transfer["lambda"] * direction
    if np.max(np.abs(delta)) > transfer["maximum_single_weight_change"] + 1e-15:
        raise ValueError("Inherited amplitude update exceeds its declared bound")
    state["raw_hidden.weight"] += delta
    _assert_state(state, recipe["final_numeric_state_sha256"])
    return state, recipe, stages


def _archive_bytes(state, configuration):
    buffer = BytesIO()
    np.savez_compressed(buffer, configuration=configuration, **state)
    return buffer.getvalue()


def verify_recipe(directory=RECIPE_DIR):
    state, recipe, stages = reconstructed_state(directory)
    raw = _archive_bytes(state, recipe["final_checkpoint_configuration_json"])
    actual = sha256(raw).hexdigest()
    return {
        "operation": "selected_parameter_replay",
        "optimization_performed": False,
        "raw_climate_data_read": False,
        "training_from_random_weights": False,
        "independent_test_claim": False,
        "final_model_sha256": recipe["final_model_sha256"],
        "reconstructed_archive_sha256": actual,
        "archive_bytes_match": actual == recipe["final_model_sha256"],
        "all_numeric_members_match": True,
        "numeric_state_members": len(state),
        "stages": stages,
    }


def replay(output, directory=RECIPE_DIR):
    """Write the selected checkpoint and a verification receipt, without fitting."""
    state, recipe, _ = reconstructed_state(directory)
    receipt = verify_recipe(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "model.npz"
    if target.exists() or (output / "run.json").exists():
        raise FileExistsError("Use a new output directory for parameter replay")
    target.write_bytes(_archive_bytes(state, recipe["final_checkpoint_configuration_json"]))
    # ZIP/container metadata can vary with NumPy; numeric members must be exact.
    _assert_state(_state(target), recipe["final_numeric_state_sha256"])
    receipt["output_checkpoint"] = str(target.resolve())
    (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def fit(bank, bank_path, output, directory=RECIPE_DIR):
    """Optimize the archived constrained development recipe from the R13 start.

    This uses known 2015/2020 event observations and development ACC in the
    objective. Directions are archived, not recomputed from new data. Numerical
    replay, rather than a fresh optimizer run, is the exact reconstruction path.
    """
    import torch
    from torch.func import functional_call
    from . import neural

    recipe, updates = load_recipe(directory)
    if _digest(bank_path) != recipe["training_bank_sha256"]:
        raise ValueError("The fixed development directions require the original matching feature bank")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory for development fitting")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(recipe["torch_threads"])
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(recipe["seed"])
    net = neural.load_checkpoint(Path(directory) / recipe["start_checkpoint"])
    baseline = neural.load_checkpoint(Path(directory) / recipe["start_checkpoint"])
    net.requires_grad_(False)
    x = torch.as_tensor(bank["val_x"], dtype=torch.float64)
    train_x = torch.as_tensor(bank["train_x"], dtype=torch.float64)
    origins = torch.as_tensor(bank["val_origins"], dtype=torch.int64)
    truth = torch.as_tensor(bank["obs"].copy(), dtype=torch.float64)
    dates = bank["dates"].astype("datetime64[M]")
    shift = torch.tensor(recipe["native_to_reference_offsets_C"])
    offsets = shift[(origins[:, None] + torch.arange(1, 21)[None, :]) % 12]
    reference_truth = truth.clone()
    reference_truth[660:804] += shift[torch.arange(660, 804) % 12]

    def acc(prediction, observation):
        p = torch.cat([prediction[:, :1],
                       (prediction[:, :-2] + prediction[:, 1:-1] + prediction[:, 2:]) / 3,
                       prediction[:, -1:]], dim=1)
        y0 = observation[660:804]
        y = torch.cat([y0[:1], (y0[:-2] + y0[1:-1] + y0[2:]) / 3, y0[-1:]])
        scores = []
        for lead in range(1, 7):
            targets = origins + lead
            keep = targets < 804
            a, b = p[keep, lead - 1], y[targets[keep] - 660]
            a, b = a - a.mean(), b - b.mean()
            scores.append((a * b).sum() / torch.sqrt(a.square().sum() * b.square().sum()))
        return torch.stack(scores)

    with torch.no_grad():
        frozen = baseline(x)
        teacher = baseline(train_x)
        reference = acc(frozen + offsets, reference_truth)
        native_reference = acc(frozen, truth)
    history_y = torch.as_tensor(bank["train_y"][:, :6])
    mask = torch.as_tensor(bank["mask"][:, :6])
    results = []
    for stage in recipe["stages"]:
        settings = stage["optimizer"]
        event_row = stage["event_row"]
        if str(dates[bank["val_origins"][event_row]]) != stage["event_input_month"]:
            raise ValueError("Development event row does not match the recipe calendar")
        protected = stage["protected_validation_rows"]
        basis = torch.as_tensor(updates[stage["id"] + "_direction"][:, None])
        base_L0 = net.L0.detach().clone()
        theta = torch.nn.Parameter(torch.zeros(1, 6))
        with torch.no_grad():
            initial = net(x)
            event_u = net.forward_with_diagnostics(x[event_row:event_row + 1])[1]["decoder"][0, :6]
            event_truth = truth[origins[event_row] + torch.arange(1, 7)]
            if stage["initialize_decoder_to_event"]:
                theta.copy_((event_truth - event_u)[None, :])
        optimizer = torch.optim.Adam([theta], lr=settings["lr"])
        best = None
        trace = []

        def changed_L0():
            return torch.cat([base_L0[:, :6] + basis @ theta, base_L0[:, 6:]], dim=1)

        def values(inputs):
            return functional_call(net, {"L0": changed_L0()}, (inputs,), strict=False)

        def inspect(step):
            nonlocal best
            with torch.no_grad():
                prediction = values(x)
                score, native = acc(prediction + offsets, reference_truth), acc(prediction, truth)
                mae = float(abs(prediction[event_row, :6] - event_truth).mean())
                drift = float(abs(prediction[protected] - initial[protected]).max())
                gates = bool(score.mean() >= reference.mean() - 1e-12
                             and score[5] >= reference[5] - 1e-12
                             and native.mean() >= native_reference.mean() - 1e-12
                             and native[5] >= native_reference[5] - 1e-12
                             and drift < 1e-9)
                row = {"step": step, "event_MAE_C": mae, "mean_ACC": float(score.mean()),
                       "ACC6": float(score[5]), "native_mean_ACC": float(native.mean()),
                       "native_ACC6": float(native[5]), "maximum_protected_prediction_change_C": drift,
                       "gates_pass": gates, "coordinates": theta.detach().numpy().tolist()}
                trace.append(row)
                if gates and (best is None or mae < best[0]):
                    best = mae, theta.detach().clone(), row
                return gates and mae <= settings["target_mae"]

        inspect(0)
        for step in range(1, settings["steps"] + 1):
            optimizer.zero_grad()
            prediction, historical = values(x), values(train_x)
            score, native = acc(prediction + offsets, reference_truth), acc(prediction, truth)
            deficits = torch.stack([
                torch.relu(reference.mean() + settings["acc_margin"] - score.mean()),
                torch.relu(reference[5] + settings["acc_margin"] - score[5]),
                torch.relu(native_reference.mean() + settings["acc_margin"] - native.mean()),
                torch.relu(native_reference[5] + settings["acc_margin"] - native[5]),
            ])
            loss = ((prediction[event_row, :6] - event_truth).square().mean()
                    + settings["teacher_weight"] * (historical - teacher).square().mean()
                    + settings["history_weight"] * (((historical[:, :6] - history_y).square() * mask).sum() / mask.sum())
                    + settings["acc_weight"] * deficits.square().sum()
                    + settings["l2"] * theta.square().mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite constrained-development loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 5.0)
            optimizer.step()
            if step % settings["eval_every"] == 0 or step == settings["steps"]:
                if inspect(step):
                    break
        if best is None:
            raise RuntimeError(f"No candidate met the development gates: {stage['id']}")
        with torch.no_grad():
            net.L0[:, :6].copy_(base_L0[:, :6] + basis @ best[1])
        stage_dir = output / stage["id"]
        stage_dir.mkdir()
        neural.save_checkpoint(stage_dir / "model.npz", net,
                               {"training_mode": "constrained_development_event_fitting",
                                "independent_test_claim": False, "stage": stage["id"],
                                "formal_primary_model": False, "explicit_user_adoption": False,
                                "post2012_truth_used_in_optimizer": True,
                                "post2012_labels_used_in_optimizer": True,
                                "direct_development_event_fitting": True,
                                "development_ACC_used_in_optimizer": True,
                                "current_role": "New constrained-development candidate"})
        (stage_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        results.append({"stage": stage["id"], **best[2]})
    with torch.no_grad():
        net.raw_hidden.weight.add_(recipe["amplitude_transfer"]["lambda"]
                                   * torch.as_tensor(updates["inherited_raw_hidden_direction"]))
    neural.validate_network(net)
    actual = {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}
    numeric_match = all(sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                        == recipe["final_numeric_state_sha256"][key]
                        for key, value in actual.items())
    metadata = {"training_mode": "constrained_development_recipe_refit",
                "weights_are_warm_started": True, "independent_test_claim": False,
                "formal_primary_model": False, "explicit_user_adoption": False,
                "post2012_truth_used_in_optimizer": True,
                "post2012_labels_used_in_optimizer": True,
                "direct_development_event_fitting": True,
                "development_ACC_used_in_optimizer": True,
                "optimizer_observation_windows": recipe["optimizer_observation_windows"],
                "exact_final_numeric_match": numeric_match}
    neural.save_checkpoint(output / "model.npz", net, metadata)
    (output / "training_recipe_run.json").write_text(json.dumps({**metadata, "stages": results}, indent=2) + "\n")
    return net, {**metadata, "stages": results}
