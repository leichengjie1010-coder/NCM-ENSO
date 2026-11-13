"""EOF reduction, seasonal Markov prediction and causal predictor construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr


FIELD_NAMES = ("sst", "taux", "tauy", "ssh", "ohc300", "z20")
BLOCKS4 = ("sst", "tau", "ssh", "z20")
BLOCKS3 = ("sst", "tau", "ssh")
BLOCK_FIELDS = {
    "sst": ("sst",),
    "tau": ("taux", "tauy"),
    "ssh": ("ssh",),
    "z20": ("z20",),
}
LAGS_PC = (0, 1, 2, 3, 6, 12)
LAGS_STATE = (0, 1, 3, 6, 12)
LAGS_NINO = (1, 3, 6, 12)
N_PC, K, H = 10, 3, 20
CUTOFF = 0.99
INACTIVE_SLICE = slice(306, 348)
Q18_NAMES = (
    "z20_east",
    "ohc300_east",
    "taux_times_angular_sst_gradient_lon_eq",
    "d1_z20_east",
    "d1_ohc300_east",
    "d1_taux_times_angular_sst_gradient_lon_eq",
    "z20_eq",
    "z20_west_minus_east",
    "ohc300_eq",
    "d1_z20_eq",
    "d1_z20_west_minus_east",
    "d1_ohc300_eq",
    "angular_stress_curl_eq",
    "tauy_times_angular_sst_gradient_lat_eq",
    "d1_angular_stress_curl_eq",
    "d1_tauy_times_angular_sst_gradient_lat_eq",
    "ohc300_west_minus_east",
    "d1_ohc300_west_minus_east",
)
REGIONAL_NAMES = (
    "z20_eq",
    "z20_west_minus_east",
    "z20_east",
    "ohc300_eq",
    "ohc300_west_minus_east",
    "ohc300_east",
    "taux_eq",
    "tauy_eq",
    "angular_stress_curl_eq",
    "angular_stress_curl_offeq",
    "taux_times_angular_sst_gradient_lon_eq",
    "tauy_times_angular_sst_gradient_lat_eq",
)
EQ7_NAMES = (
    "z20",
    "ohc300",
    "taux",
    "tauy",
    "angular_stress_curl",
    "taux_times_angular_sst_gradient_lon",
    "tauy_times_angular_sst_gradient_lat",
)


def _unit_token(unit: str) -> str:
    return (
        str(unit)
        .lower()
        .replace(" ", "")
        .replace("**", "^")
        .replace("²", "2")
        .replace("−", "-")
        .replace("_", "")
    )


def validate_physical_input(
    raw: xr.Dataset, *, require_training: bool = True
) -> pd.DatetimeIndex:
    """Validate coordinate/product contract without choosing masks using future data."""
    for name in ("time", "lat", "lon"):
        if name not in raw.coords or raw[name].ndim != 1:
            raise ValueError(f"Require one-dimensional coordinate {name}")
    dates = pd.DatetimeIndex(pd.to_datetime(raw.time.values))
    periods = dates.to_period("M")
    if len(dates) == 0 or dates.isna().any() or periods.has_duplicates:
        raise ValueError("Empty, invalid or duplicate months")
    if require_training and periods[0] != pd.Period("1958-01", freq="M"):
        raise ValueError("Input index zero must be January 1958")
    if not periods.equals(pd.period_range(periods[0], periods[-1], freq="M")):
        raise ValueError(
            (
                "Input months must be sorted and continuous; missing months "
                "cannot be inferred"
            )
        )
    for coord, expected in (
        ("lat", np.arange(-20.0, 20.0001, 2.5)),
        ("lon", np.arange(120.0, 280.0001, 2.5)),
    ):
        values = np.asarray(raw[coord], float)
        if values.shape != expected.shape or not np.allclose(
            values, expected, rtol=0, atol=1e-10
        ):
            raise ValueError(
                f"Unexpected {coord} grid; "
                "provide the stated 2.5-degree analysis grid"
            )
    units = {
        "sst": {"degc", "c", "celsius", "degreecelsius", "degreescelsius"},
        "taux": {"nm-2", "n/m2", "nm^-2", "n/m^2", "pa"},
        "tauy": {"nm-2", "n/m2", "nm^-2", "n/m^2", "pa"},
        "ssh": {"m", "meter", "metre", "meters", "metres"},
        "z20": {"m", "meter", "metre", "meters", "metres"},
        "ohc300": {"jm-2", "j/m2", "jm^-2", "j/m^2"},
    }
    if require_training and (
        len(dates) < 660
        or not periods[:660].equals(pd.period_range("1958-01", "2012-12", freq="M"))
    ):
        raise ValueError(
            "All 660 fitting months from 1958-01 through 2012-12 are required"
        )
    if not require_training and len(dates) < 13:
        raise ValueError(
            "Event projection needs at least 13 continuous monthly inputs"
        )
    for name in FIELD_NAMES:
        if name not in raw or raw[name].dims != ("time", "lat", "lon"):
            raise ValueError(f"Missing or incompatible {name}(time,lat,lon)")
        if _unit_token(raw[name].attrs.get("units", "")) not in units[name]:
            raise ValueError(
                f"Unsupported or absent units for {name}"
            )
        values = np.asarray(raw[name].values)
        if np.isinf(values).any():
            raise ValueError(f"Infinite values in {name}")
        if require_training and not np.isfinite(values[:660]).any(axis=(1, 2)).all():
            raise ValueError(f"Wholly missing fitting month in {name}")
    # Future missing fields are retained as missing, not imputed. Invalid origins
    # will be explicitly excluded from the downstream input bank.
    return dates


def fit_climatology(
    raw: xr.Dataset, train_n: int = 660
) -> tuple[xr.Dataset, xr.Dataset]:
    """Fit training-only monthly climatology, preserving storage precision."""
    training = raw[list(FIELD_NAMES)].isel(time=slice(0, train_n))
    climate = training.groupby("time.month").mean("time", skipna=True, dtype=np.float64)
    climate = climate.astype({name: raw[name].dtype for name in FIELD_NAMES})
    climate.attrs.update(
        climatology_period="1958-01 to2012-12",
        fit_months=train_n,
        fitted_on_training_only="true",
    )
    anomaly = raw[list(FIELD_NAMES)].groupby("time.month") - climate
    for name in FIELD_NAMES:
        anomaly[name].attrs = dict(raw[name].attrs)
        anomaly[name].attrs[
            "anomaly_reference"
        ] = "1958-2012 calendar-month climatology"
    return climate, anomaly


@dataclass
class FirstEOF:
    block: str
    mean: np.ndarray
    eof: np.ndarray
    singular_values_all: np.ndarray
    variance_fraction_all: np.ndarray
    cumulative_fraction_all: np.ndarray
    j99: int
    mask: np.ndarray
    spatial_weights: np.ndarray
    pc_all: np.ndarray
    projection_valid: np.ndarray


@dataclass
class MarkovModel:
    name: str
    blocks: tuple[str, ...]
    modes: np.ndarray
    sigma: dict[str, float]
    singular_values_all: np.ndarray
    variance_fraction_all: np.ndarray
    state_all: np.ndarray
    patterns: dict[str, np.ndarray]
    patterns_grid: dict[str, np.ndarray]
    decoder_coeff: np.ndarray
    phase: np.ndarray
    nonseasonal: np.ndarray
    phase_sample_counts: np.ndarray


def fit_first_eofs(anomaly: xr.Dataset, train_n: int = 660) -> dict[str, FirstEOF]:
    """Unweighted covariance EOFs; finite spatial support comes from training only."""
    n, ny, nx = anomaly.sizes["time"], anomaly.sizes["lat"], anomaly.sizes["lon"]
    fits = {}
    for block in BLOCKS4:
        variables = BLOCK_FIELDS[block]
        mask2d = np.ones((ny, nx), dtype=bool)
        for name in variables:
            mask2d &= np.isfinite(np.asarray(anomaly[name].values[:train_n])).all(
                axis=0
            )
        mask = mask2d.reshape(-1)
        if not mask.any():
            raise ValueError(f"No common finite training cells in {block}")
        weights = np.ones(
            int(mask.sum()), dtype=np.float64
        )
        x = np.concatenate(
            [
                np.asarray(anomaly[v].values).reshape(n, -1)[:, mask] * weights[None, :]
                for v in variables
            ],
            axis=1,
        )
        mean = x[:train_n].mean(axis=0)
        _, singular, vt = np.linalg.svd(
            x[:train_n] - mean[None, :], full_matrices=False
        )
        variance = singular * singular
        total = float(variance.sum())
        if total <= np.finfo(float).eps:
            raise ValueError(f"Degenerate first EOF block {block}")
        fraction, cumulative = variance / total, np.cumsum(variance / total)
        j99 = min(int(np.searchsorted(cumulative, CUTOFF) + 1), len(singular))
        if j99 < N_PC:
            raise ValueError(
                f"{block} retains only {j99} PCs; "
                "the fixed 10-PC input schema is unavailable"
            )
        eof = vt[:j99].copy()
        valid = np.isfinite(x).all(axis=1)
        pc_all = np.full((n, j99), np.nan)
        pc_all[valid] = (x[valid] - mean[None, :]) @ eof.T
        fits[block] = FirstEOF(
            block,
            mean,
            eof,
            singular,
            fraction,
            cumulative,
            j99,
            mask,
            weights,
            pc_all,
            valid,
        )
    return fits


def regional_mean(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_mask: np.ndarray,
    lon_mask: np.ndarray,
) -> np.ndarray:
    """Cosine-latitude-weighted mean over finite cells."""
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError("Regional index is outside the supplied grid")
    values = np.asarray(field, float)[:, lat_mask, :][:, :, lon_mask]
    weights = np.cos(np.deg2rad(lat[lat_mask]))[None, :, None]
    denom = np.nansum(np.where(np.isfinite(values), weights, np.nan), axis=(1, 2))
    numerator = np.nansum(values * weights, axis=(1, 2))
    return np.divide(
        numerator, denom, out=np.full_like(numerator, np.nan), where=denom > 0
    )


def nino34(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return regional_mean(
        field, lat, lon, (lat >= -5) & (lat <= 5), (lon >= 190) & (lon <= 240)
    )


def fit_markov(
    state: np.ndarray, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source-month m uses transitions pooled over m-1,m,m+1; no ridge penalty."""
    phase = np.empty((12, K, K), dtype=np.float64)
    counts = np.empty(12, dtype=int)
    for month in range(1, 13):
        allowed = (((month - 2) % 12) + 1, month, (month % 12) + 1)
        index = np.flatnonzero(np.isin(dates.month, allowed))
        index = index[index < len(state) - 1]
        if len(index) < K + 2:
            raise ValueError(f"Too few transitions in source-month phase {month}")
        x, y = state[index], state[index + 1]
        denom = max(len(index) - 1, 1)
        phase[month - 1] = ((y.T @ x) / denom) @ np.linalg.pinv(
            (x.T @ x) / denom, rcond=1e-10
        )
        counts[month - 1] = len(index)
    x, y = state[:-1], state[1:]
    denom = max(len(x) - 1, 1)
    nonseasonal = ((y.T @ x) / denom) @ np.linalg.pinv((x.T @ x) / denom, rcond=1e-10)
    return phase, nonseasonal, counts


def fit_meof_markov(
    fits: dict[str, FirstEOF],
    blocks: Sequence[str],
    dates: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    train_n: int = 660,
) -> MarkovModel:
    """Fit a separate joint basis and operator for each block set."""
    blocks = tuple(blocks)
    sigma = {b: float(np.sqrt(np.sum(fits[b].pc_all[:train_n] ** 2))) for b in blocks}
    if any(value <= 0 or not np.isfinite(value) for value in sigma.values()):
        raise ValueError("Nonpositive MEOF block Frobenius norm")
    combined = np.concatenate([fits[b].pc_all / sigma[b] for b in blocks], axis=1)
    _, singular, vt = np.linalg.svd(combined[:train_n], full_matrices=False)
    if len(singular) < K:
        raise ValueError("Fewer than three MEOF modes")
    fraction = singular**2 / np.sum(singular**2)
    modes = vt[:K].copy()
    valid = np.isfinite(combined).all(axis=1)
    state = np.full((len(dates), K), np.nan)
    state[valid] = combined[valid] @ modes.T
    phase, nonseasonal, counts = fit_markov(state[:train_n], dates[:train_n])
    patterns, grids = {}, {}
    offset = 0
    for block in blocks:
        fit = fits[block]
        patterns[block] = (modes[:, offset : offset + fit.j99] * sigma[block]) @ fit.eof
        offset += fit.j99
        nvalid = int(fit.mask.sum())
        for component, variable in enumerate(BLOCK_FIELDS[block]):
            flat = np.full((K, len(lat) * len(lon)), np.nan)
            flat[:, fit.mask] = (
                patterns[block][:, component * nvalid : (component + 1) * nvalid]
                / fit.spatial_weights[None, :]
            )
            grids[variable] = flat.reshape(K, len(lat), len(lon))
    coeff = nino34(grids["sst"], lat, lon)
    if not np.isfinite(coeff).all():
        raise ValueError("MEOF SST decoder lacks Niño3.4 support")
    return MarkovModel(
        f"MMarkov-{len(blocks)}",
        blocks,
        modes,
        sigma,
        singular,
        fraction,
        state,
        patterns,
        grids,
        coeff,
        phase,
        nonseasonal,
        counts,
    )


def predict_markov(
    model: MarkovModel, origins: np.ndarray, dates: pd.DatetimeIndex, horizon: int = H
) -> np.ndarray:
    origins = np.asarray(origins, dtype=int)
    if origins.ndim != 1 or np.any(origins < 0) or np.any(origins >= len(dates)):
        raise ValueError("Forecast origins must be one-dimensional valid month indices")
    z = model.state_all[origins].copy()
    months = dates[origins].month.to_numpy()
    out = np.empty((len(origins), horizon), dtype=np.float64)
    for lead in range(horizon):
        z = np.einsum("nij,nj->ni", model.phase[months - 1], z)
        out[:, lead] = z @ model.decoder_coeff
        months = months % 12 + 1
    return out


def angular_gradient_proxies(
    anomaly: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Angular-gradient proxies used as statistical predictors.

    Curl is a statistical stress-curl proxy in Pa/radian. Stress times SST
    gradient is not a heat budget or an oceanic advective tendency.
    """
    lat, lon = np.asarray(anomaly.lat, float), np.asarray(anomaly.lon, float)
    sst = np.asarray(anomaly.sst, float)
    taux, tauy = np.asarray(anomaly.taux, float), np.asarray(anomaly.tauy, float)
    dlat, dlon = np.deg2rad(np.gradient(lat)), np.deg2rad(np.gradient(lon))
    gx = np.gradient(sst, axis=2) / dlon[None, None, :]
    gy = np.gradient(sst, axis=1) / dlat[None, :, None]
    curl = (
        np.gradient(tauy, axis=2) / dlon[None, None, :]
        - np.gradient(taux, axis=1) / dlat[None, :, None]
    )
    return curl, taux * gx, tauy * gy


def causal_first_difference(values: np.ndarray) -> np.ndarray:
    """First row has no observed predecessor and remains NaN, never a real zero."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    out[1:] = np.diff(values, axis=0)
    return out


def physical_indices(anomaly: xr.Dataset) -> dict[str, np.ndarray]:
    """Compute regional state predictors and their causal first differences."""
    lat, lon = np.asarray(anomaly.lat, float), np.asarray(anomaly.lon, float)
    curl, heatx, heaty = angular_gradient_proxies(anomaly)
    z20, ohc = np.asarray(anomaly.z20, float), np.asarray(anomaly.ohc300, float)
    taux, tauy = np.asarray(anomaly.taux, float), np.asarray(anomaly.tauy, float)
    eq = (lat >= -5) & (lat <= 5)
    offeq = ((lat >= -15) & (lat <= -5)) | ((lat >= 5) & (lat <= 15))
    whole, west, east = (
        (lon >= 120) & (lon <= 280),
        (lon >= 120) & (lon <= 180),
        (lon >= 220) & (lon <= 280),
    )
    fields = (
        (z20, eq, whole),
        (z20, eq, west),
        (z20, eq, east),
        (ohc, eq, whole),
        (ohc, eq, west),
        (ohc, eq, east),
        (taux, eq, whole),
        (tauy, eq, whole),
        (curl, eq, whole),
        (curl, offeq, whole),
        (heatx, eq, whole),
        (heaty, eq, whole),
    )
    raw = np.column_stack([regional_mean(a, lat, lon, lm, om) for a, lm, om in fields])
    raw[:, 1] -= raw[:, 2]
    raw[:, 4] -= raw[:, 5]
    difference = causal_first_difference(raw)
    q18 = np.column_stack(
        (
            raw[:, 2],
            raw[:, 5],
            raw[:, 10],
            difference[:, 2],
            difference[:, 5],
            difference[:, 10],
            raw[:, 0],
            raw[:, 1],
            raw[:, 3],
            difference[:, 0],
            difference[:, 1],
            difference[:, 3],
            raw[:, 8],
            raw[:, 11],
            difference[:, 8],
            difference[:, 11],
            raw[:, 4],
            difference[:, 4],
        )
    )
    eq10 = (lat >= -10) & (lat <= 10)
    eq7 = np.column_stack(
        [
            regional_mean(a, lat, lon, eq10, whole)
            for a in (z20, ohc, taux, tauy, curl, heatx, heaty)
        ]
    )
    return {
        "regional12": raw,
        "regional12_d1": difference,
        "q18": q18,
        "eq7": eq7,
        "eq7_d1": causal_first_difference(eq7),
    }


def feature_names() -> list[str]:
    names = [f"markov4_lead{h:02d}" for h in range(1, H + 1)]
    names += [
        f"{b}_PC{j:02d}_lag{lag:02d}"
        for lag in LAGS_PC
        for b in BLOCKS4
        for j in range(1, N_PC + 1)
    ]
    for lag in LAGS_STATE:
        names += [f"MEOF4_PC{k}_lag{lag:02d}" for k in range(1, K + 1)]
        names += [f"nino34_lag{lag:02d}"]
    names += [
        "calendar_sin",
        "calendar_cos",
        "nino34_current",
        "nino34_squared",
        "nino34_cubed",
        "nino34_signed_sqrt",
    ]
    for lag in LAGS_NINO:
        names += [
            f"nino34_change_lag{lag:02d}",
            f"nino34_lag{lag:02d}_nonlinear_branch",
        ]
    names += [f"calendar_month_{m:02d}" for m in range(1, 13)]
    names += [
        f"INACTIVE_d1_eq10_{name}_lag{lag:02d}" for lag in LAGS_PC for name in EQ7_NAMES
    ]
    names += list(Q18_NAMES)
    if len(names) != 366:
        raise AssertionError("Expected exactly 366 predictor names")
    return names


def build_network_features(
    fits: dict[str, FirstEOF],
    model4: MarkovModel,
    markov4_all: np.ndarray,
    obs: np.ndarray,
    physical: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    """Construct active predictors only from information at/before each origin.

    Active lag-12 memory needs 13 observed months (t-12 through t). Inactive lag-12
    differences would need 14 (t-13 through t). Their missing values are tagged
    and set to zero exclusively because every network route masks them.
    """
    origins = np.arange(12, len(dates), dtype=int)
    pieces = [markov4_all[origins]]
    pieces += [fits[b].pc_all[origins - lag, :N_PC] for lag in LAGS_PC for b in BLOCKS4]
    for lag in LAGS_STATE:
        pieces += [model4.state_all[origins - lag], obs[origins - lag, None]]
    months = dates[origins].month.to_numpy()
    pieces += [
        np.sin(2 * np.pi * months / 12)[:, None],
        np.cos(2 * np.pi * months / 12)[:, None],
    ]
    nino = obs[origins]
    pieces += [
        nino[:, None],
        (nino**2)[:, None],
        (nino**3)[:, None],
        (np.sign(nino) * np.sqrt(np.abs(nino)))[:, None],
    ]
    for lag in LAGS_NINO:
        lagged = obs[origins - lag]
        pieces += [(nino - lagged)[:, None], lagged[:, None]]
    onehot = np.zeros((len(origins), 12))
    onehot[np.arange(len(origins)), months - 1] = 1.0
    pieces += [onehot]
    inactive = np.concatenate(
        [physical["eq7_d1"][origins - lag] for lag in LAGS_PC], axis=1
    )
    inactive_valid = np.isfinite(inactive)
    pieces += [np.where(inactive_valid, inactive, 0.0), physical["q18"][origins]]
    x = np.concatenate(pieces, axis=1).astype(np.float64, copy=False)
    if x.shape[1] != 366:
        raise AssertionError("Expected exactly 366 predictor columns")
    active = np.ones(366, dtype=bool)
    active[INACTIVE_SLICE] = False
    valid = np.isfinite(x[:, active]).all(axis=1)
    return {
        "origins": origins,
        "x": x,
        "active_valid": valid,
        "inactive42_valid": inactive_valid,
        "active_mask": active,
    }


def centered_three_month(values: np.ndarray) -> np.ndarray:
    """Centered three-month means with unaltered endpoints."""
    raw = np.asarray(values, dtype=np.float64)
    out = raw.copy()
    if len(raw) >= 3:
        out[1:-1] = (raw[:-2] + raw[1:-1] + raw[2:]) / 3.0
    return out


def training_labels(
    origins: np.ndarray, obs: np.ndarray, cutoff: int = 660, horizon: int = H
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = origins[:, None] + np.arange(1, horizon + 1)[None, :]
    inbounds = target < min(cutoff, len(obs))
    y = np.zeros(target.shape, dtype=np.float64)
    y[inbounds] = obs[target[inbounds]]
    mask = inbounds & np.isfinite(y)
    y[~mask] = 0.0
    # A centered label at 2012-12 requires 2013-01 and is excluded even if loaded.
    smooth = centered_three_month(obs)
    inbounds3 = (target > 0) & (target + 1 < min(cutoff, len(obs)))
    y3 = np.zeros(target.shape, dtype=np.float64)
    y3[inbounds3] = smooth[target[inbounds3]]
    mask3 = inbounds3 & np.isfinite(y3)
    y3[~mask3] = 0.0
    return y, y3, mask, mask3


def _save_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def save_first_sources(
    out: Path,
    fits: dict[str, FirstEOF],
    dates: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    train_n: int = 660,
) -> None:
    for block, fit in fits.items():
        sd = fit.pc_all[:train_n].std(axis=0, ddof=1)
        _save_npz(
            out / f"first_eof_{block}.npz",
            mean=fit.mean,
            eof=fit.eof,
            singular_values_all=fit.singular_values_all,
            variance_fraction_all=fit.variance_fraction_all,
            cumulative_fraction_all=fit.cumulative_fraction_all,
            retained_singular_values=fit.singular_values_all[: fit.j99],
            j99=np.asarray(fit.j99),
            mask=fit.mask,
            spatial_weights=fit.spatial_weights,
            pc_all=fit.pc_all,
            pc_train=fit.pc_all[:train_n],
            pc_training_sample_std=sd,
            pc1_train=fit.pc_all[:train_n, 0],
            pc1_train_standardized=fit.pc_all[:train_n, 0] / sd[0],
            projection_valid=fit.projection_valid,
            dates=dates.to_numpy(dtype="datetime64[ns]"),
            lat=lat,
            lon=lon,
            block=np.asarray(block),
            train_n=np.asarray(train_n),
            sign_convention=np.asarray(
                "native SVD; plotting may pair-flip loading and PC only"
            ),
        )


def save_meof_sources(
    out: Path,
    model: MarkovModel,
    dates: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    train_n: int = 660,
) -> None:
    sd = model.state_all[:train_n].std(axis=0, ddof=1)
    payload = {
        "modes": model.modes,
        "blocks": np.asarray(model.blocks),
        "sigma": np.asarray([model.sigma[b] for b in model.blocks]),
        "singular_values_all": model.singular_values_all,
        "variance_fraction_all": model.variance_fraction_all,
        "state_all": model.state_all,
        "state_train": model.state_all[:train_n],
        "state_training_sample_std": sd,
        "state_train_standardized": model.state_all[:train_n] / sd[None, :],
        "decoder_coeff": model.decoder_coeff,
        "phase": model.phase,
        "nonseasonal": model.nonseasonal,
        "phase_sample_counts": model.phase_sample_counts,
        "lat": lat,
        "lon": lon,
        "dates": dates.to_numpy(dtype="datetime64[ns]"),
        "train_n": np.asarray(train_n),
        "name": np.asarray(model.name),
        "display_sign": np.asarray([-1.0, 1.0, 1.0]),
    }
    for block, pattern in model.patterns.items():
        payload[f"pattern_weighted_{block}"] = pattern
    for name, grid in model.patterns_grid.items():
        payload[f"patterns_grid_{name}"] = grid
        payload[f"patterns_grid_{name}_per_training_sd"] = grid * sd[:, None, None]
    _save_npz(out / f"meof_markov{len(model.blocks)}.npz", **payload)
    fields = {
        name: (("mode", "lat", "lon"), grid * sd[:, None, None])
        for name, grid in model.patterns_grid.items()
    }
    ds = xr.Dataset(
        {
            **fields,
            "pc_raw": (("time", "mode"), model.state_all[:train_n]),
            "pc_standardized": (
                ("time", "mode"),
                model.state_all[:train_n] / sd[None, :],
            ),
            "pc_sample_std": (("mode",), sd),
            "variance_fraction": (("mode",), model.variance_fraction_all[:K]),
        },
        coords={
            "mode": np.arange(1, K + 1),
            "time": dates[:train_n],
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "model": model.name,
            "basis_fit_period": "1958-01 to2012-12",
            "spatial_pattern_scale": "physical anomaly per training sample SD (ddof=1)",
            ("mode_convention"): (
                "native fitted signs; paired display sign is applied only in "
                "plotting"
            ),
            "area_weighted_EOF": 0,
            "linear_detrending": 0,
            "full_fit_training_predictions_are_OOF": 0,
        },
    )
    for name in fields:
        ds[name].attrs["units"] = {
            "sst": "degC",
            "taux": "N m-2",
            "tauy": "N m-2",
            "ssh": "m",
            "z20": "m",
        }[name]
    ds.to_netcdf(out / f"meof_markov{len(model.blocks)}_plot_sources.nc")


def _load_projected_linear_stage(
    linear_dir: Path, anomaly: xr.Dataset, dates: pd.DatetimeIndex
) -> tuple[dict[str, FirstEOF], MarkovModel, MarkovModel]:
    """Load a fitted basis and project a new window."""
    fits = {}
    for block in BLOCKS4:
        with np.load(
            linear_dir / f"first_eof_{block}.npz", allow_pickle=False
        ) as saved:
            mask, weights = saved["mask"].astype(bool), saved["spatial_weights"].copy()
            mean, eof = saved["mean"].copy(), saved["eof"].copy()
            x = np.concatenate(
                [
                    np.asarray(anomaly[name]).reshape(len(dates), -1)[:, mask]
                    * weights[None, :]
                    for name in BLOCK_FIELDS[block]
                ],
                axis=1,
            )
            valid = np.isfinite(x).all(axis=1)
            pc = np.full((len(dates), eof.shape[0]), np.nan)
            pc[valid] = (x[valid] - mean[None, :]) @ eof.T
            fits[block] = FirstEOF(
                block,
                mean,
                eof,
                saved["singular_values_all"].copy(),
                saved["variance_fraction_all"].copy(),
                saved["cumulative_fraction_all"].copy(),
                int(saved["j99"]),
                mask,
                weights,
                pc,
                valid,
            )
    models = []
    for number in (4, 3):
        with np.load(
            linear_dir / f"meof_markov{number}.npz", allow_pickle=False
        ) as saved:
            blocks = tuple(str(v) for v in saved["blocks"])
            sigma = dict(zip(blocks, (float(v) for v in saved["sigma"])))
            modes = saved["modes"].copy()
            combined = np.concatenate(
                [fits[b].pc_all / sigma[b] for b in blocks], axis=1
            )
            state = combined @ modes.T
            models.append(
                MarkovModel(
                    str(saved["name"]),
                    blocks,
                    modes,
                    sigma,
                    saved["singular_values_all"].copy(),
                    saved["variance_fraction_all"].copy(),
                    state,
                    {},
                    {},
                    saved["decoder_coeff"].copy(),
                    saved["phase"].copy(),
                    saved["nonseasonal"].copy(),
                    saved["phase_sample_counts"].copy(),
                )
            )
    return fits, models[0], models[1]


def _write_inputs(
    out: Path,
    fits: dict[str, FirstEOF],
    model4: MarkovModel,
    model3: MarkovModel,
    anomaly: xr.Dataset,
    dates: pd.DatetimeIndex,
    climate_native: np.ndarray,
    climate_common: np.ndarray,
    *,
    training: bool,
) -> dict[str, Any]:
    lat, lon = np.asarray(anomaly.lat, float), np.asarray(anomaly.lon, float)
    all_months = np.arange(len(dates), dtype=int)
    pred4 = predict_markov(model4, all_months, dates)
    pred3 = predict_markov(model3, all_months, dates)
    obs = nino34(np.asarray(anomaly.sst), lat, lon)
    indices = physical_indices(anomaly)
    features = build_network_features(fits, model4, pred4, obs, indices, dates)
    valid = features["active_valid"]
    origins, inputs = features["origins"][valid], features["x"][valid]
    if len(origins) == 0:
        raise ValueError("No origin has all 324 required predictors")
    common = {
        "obs": obs,
        "dates": dates.to_numpy(dtype="datetime64[ns]"),
        "all_x": inputs,
        "all_origins": origins,
        "all_inactive42_valid": features["inactive42_valid"][valid],
        "active_input_mask": features["active_mask"],
        "all_markov": pred4[origins],
        "all_markov3": pred3[origins],
        "nino_climatology": climate_native,
        "nino_climatology_1958_2012": climate_native,
        "nino_climatology_1982_2010": climate_common,
        "common_offset_C": climate_native-climate_common,
        "feature_names": np.asarray(feature_names()),
    }
    metadata = {
        "stored_features": 366,
        "active_features": 324,
        "masked_slice": [306, 348],
        "summary_features": 29,
        "horizon": H,
        "basis_fit_period": ["1958-01", "2012-12"],
        "training_predictions": "full-fit, in-sample",
    }
    if training:
        train_origins = origins[origins < 660]
        train_x = inputs[origins < 660]
        y, y3, mask, mask3 = training_labels(train_origins, obs)
        usable = mask.any(1)
        train_origins, train_x = train_origins[usable], train_x[usable]
        periods = dates.to_period("M")
        evaluation = ((periods[origins] >= pd.Period("2013-01", freq="M"))
                      & (periods[origins] <= pd.Period("2024-11", freq="M")))
        val_origins, val_x = origins[evaluation], inputs[evaluation]
        common.update(
            train_x=train_x, train_origins=train_origins, origins=train_origins,
            train_y=y[usable], train_y3=y3[usable],
            mask=mask[usable], mask3=mask3[usable],
            val_x=val_x, val_origins=val_origins,
            train_markov=pred4[train_origins], val_markov=pred4[val_origins],
        )
        metadata.update(
            train_origins=int(len(train_origins)),
            evaluation_origins=int(len(val_origins)),
            evaluation_period=["2013-01", "2024-12"],
            evaluation_kind="development evaluation",
        )
    _save_npz(out / "network_inputs_366.npz", **common)
    _save_npz(
        out / "all_origin_baselines_20.npz",
        origins=all_months,
        dates=dates.to_numpy(dtype="datetime64[ns]"),
        obs=obs,
        MMarkov4=pred4,
        MMarkov3=pred3,
        nino_climatology=climate_native,
        nino_climatology_1958_2012=climate_native,
        nino_climatology_1982_2010=climate_common,
        common_offset_C=climate_native-climate_common,
    )
    np.savez_compressed(
        out / "physical_regional_indices.npz", **indices,
        q18_names=np.asarray(Q18_NAMES), regional12_names=np.asarray(REGIONAL_NAMES),
        eq7_names=np.asarray(EQ7_NAMES), dates=dates.to_numpy(dtype="datetime64[ns]"),
    )
    return metadata


def fit_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit the study's 1958–2012 backbone from prepared monthly physical fields.

    The input contains unit-converted fields on the analysis grid, before
    calendar-month climatology removal. No raw-product processing is performed.
    """
    input_path, out = Path(input_path), Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Prepared monthly fields were not found: {input_path}")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}")
    with xr.open_dataset(input_path) as handle:
        raw = handle.load()
    dates = validate_physical_input(raw)
    lat, lon = np.asarray(raw.lat, float), np.asarray(raw.lon, float)
    climate, anomaly = fit_climatology(raw)
    fits = fit_first_eofs(anomaly)
    model4 = fit_meof_markov(fits, BLOCKS4, dates, lat, lon)
    model3 = fit_meof_markov(fits, BLOCKS3, dates, lat, lon)
    native = nino34(np.asarray(climate.sst.transpose("month", "lat", "lon")), lat, lon)
    common_field = (
        raw.sst.sel(time=slice("1982-01", "2010-12"))
        .groupby("time.month").mean("time", skipna=True, dtype=np.float64)
        .astype(raw.sst.dtype)
    )
    common = nino34(np.asarray(common_field.transpose("month", "lat", "lon")), lat, lon)
    out.mkdir(parents=True, exist_ok=True)
    metadata = _write_inputs(out, fits, model4, model3, anomaly, dates,
                             native, common, training=True)
    climate.to_netcdf(out / "monthly_climatology_1958_2012.nc")
    _save_npz(out / "climatology_nino34.npz", month=np.arange(1, 13),
              nino_climatology_1958_2012=native,
              nino_climatology_1982_2010=common, common_offset_C=native-common)
    save_first_sources(out, fits, dates, lat, lon)
    save_meof_sources(out, model4, dates, lat, lon)
    save_meof_sources(out, model3, dates, lat, lon)
    return metadata


def project_inputs(
    input_path: str | Path,
    fitted_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Project a continuous 13+-month window through the frozen fitted basis."""
    out, fitted = Path(output_dir), Path(fitted_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}")
    with xr.open_dataset(input_path) as handle:
        raw = handle.load()
    dates = validate_physical_input(raw, require_training=False)
    with xr.open_dataset(fitted / "monthly_climatology_1958_2012.nc") as handle:
        climate = handle.load()
    with np.load(fitted / "climatology_nino34.npz", allow_pickle=False) as saved:
        native = saved["nino_climatology_1958_2012"].copy()
        common = saved["nino_climatology_1982_2010"].copy()
    anomaly = raw[list(FIELD_NAMES)].groupby("time.month")-climate
    fits, model4, model3 = _load_projected_linear_stage(fitted, anomaly, dates)
    out.mkdir(parents=True, exist_ok=True)
    metadata = _write_inputs(out, fits, model4, model3, anomaly, dates,
                             native, common, training=False)
    metadata["projection_only"] = True
    return metadata
