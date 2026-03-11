from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def _ensure_scgeo_uns(adata):
    if "scgeo" not in adata.uns or not isinstance(adata.uns["scgeo"], dict):
        adata.uns["scgeo"] = {}


def _get_basis_xy(adata, basis: str) -> np.ndarray:
    key = basis if basis.startswith("X_") else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found in adata.obsm.")
    xy = np.asarray(adata.obsm[key])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Embedding '{key}' must have shape (n_cells, >=2).")
    return xy[:, :2]


def _as_str_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(s):
        return s.astype(str)
    return s.astype("category").astype(str)


def _compute_shift_summary(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str,
    min_cells: int = 10,
) -> pd.DataFrame:
    xy = _get_basis_xy(adata, basis)

    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")
    if condition_key not in adata.obs:
        raise KeyError(f"'{condition_key}' not found in adata.obs.")

    g0 = str(group0)
    g1 = str(group1)

    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            condition_key: adata.obs[condition_key].astype(str).values,
            "x": xy[:, 0],
            "y": xy[:, 1],
        },
        index=adata.obs_names,
    )
    df = df[df[condition_key].isin([g0, g1])].copy()
    if df.empty:
        raise ValueError(f"No cells found for groups {g0!r} and {g1!r}.")

    cent = (
        df.groupby([node_key, condition_key], observed=False)[["x", "y"]]
        .mean()
        .reset_index()
    )
    counts = (
        df.groupby([node_key, condition_key], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cent = cent.merge(counts, on=[node_key, condition_key], how="left")

    c0 = cent[cent[condition_key] == g0].rename(
        columns={"x": "x0", "y": "y0", "n_cells": "n0"}
    )[[node_key, "x0", "y0", "n0"]]
    c1 = cent[cent[condition_key] == g1].rename(
        columns={"x": "x1", "y": "y1", "n_cells": "n1"}
    )[[node_key, "x1", "y1", "n1"]]

    out = c0.merge(c1, on=node_key, how="outer")
    out["n0"] = out["n0"].fillna(0).astype(int)
    out["n1"] = out["n1"].fillna(0).astype(int)
    out["present0"] = out["n0"] >= int(min_cells)
    out["present1"] = out["n1"] >= int(min_cells)
    out["dx"] = out["x1"] - out["x0"]
    out["dy"] = out["y1"] - out["y0"]
    out["shift_norm"] = np.sqrt(np.square(out["dx"]) + np.square(out["dy"]))
    return out


def _compute_velocity_alignment(
    adata,
    *,
    shift_df: pd.DataFrame,
    node_key: str,
    basis: str,
    velocity_basis: Optional[str] = None,
    min_cells: int = 10,
) -> pd.DataFrame:
    if velocity_basis is None:
        vel_key = f"velocity_{basis}" if not basis.startswith("X_") else f"velocity_{basis[2:]}"
    else:
        vel_key = velocity_basis if velocity_basis.startswith("velocity_") else f"velocity_{velocity_basis}"

    if vel_key not in adata.obsm:
        raise KeyError(f"Velocity embedding '{vel_key}' not found in adata.obsm.")
    if node_key not in adata.obs:
        raise KeyError(f"'{node_key}' not found in adata.obs.")

    vel = np.asarray(adata.obsm[vel_key])
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"Velocity embedding '{vel_key}' must have shape (n_cells, >=2).")

    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            "vx": vel[:, 0],
            "vy": vel[:, 1],
        },
        index=adata.obs_names,
    )

    out = (
        df.groupby(node_key, observed=False)[["vx", "vy"]]
        .mean()
        .reset_index()
    )
    counts = df.groupby(node_key, observed=False).size().rename("n_vel").reset_index()
    out = out.merge(counts, on=node_key, how="left")
    out["velocity_present"] = out["n_vel"].fillna(0).astype(int) >= int(min_cells)
    out["vel_norm"] = np.sqrt(np.square(out["vx"]) + np.square(out["vy"]))

    merged = shift_df.merge(out, on=node_key, how="left")

    dx = merged["dx"].to_numpy(dtype=float)
    dy = merged["dy"].to_numpy(dtype=float)
    vx = merged["vx"].to_numpy(dtype=float)
    vy = merged["vy"].to_numpy(dtype=float)

    dot = dx * vx + dy * vy
    dn = np.sqrt(dx * dx + dy * dy)
    vn = np.sqrt(vx * vx + vy * vy)
    denom = dn * vn

    cosine = np.full_like(dot, np.nan, dtype=float)
    mask = np.isfinite(denom) & (denom > 0)
    cosine[mask] = dot[mask] / denom[mask]
    merged["alignment_cosine"] = cosine
    return merged


def _compute_composition(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
) -> pd.DataFrame:
    g0 = str(group0)
    g1 = str(group1)

    df = pd.DataFrame(
        {
            node_key: _as_str_series(adata.obs[node_key]),
            condition_key: adata.obs[condition_key].astype(str).values,
        },
        index=adata.obs_names,
    )
    df = df[df[condition_key].isin([g0, g1])].copy()
    if df.empty:
        raise ValueError(f"No cells found for groups {g0!r} and {g1!r}.")

    counts = (
        df.groupby([node_key, condition_key], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = (
        counts.groupby(condition_key, observed=False)["n_cells"]
        .sum()
        .rename("total_cells")
        .reset_index()
    )
    counts = counts.merge(totals, on=condition_key, how="left")
    counts["fraction"] = counts["n_cells"] / counts["total_cells"]

    c0 = counts[counts[condition_key] == g0].rename(
        columns={"n_cells": "n0", "fraction": "frac0"}
    )[[node_key, "n0", "frac0"]]
    c1 = counts[counts[condition_key] == g1].rename(
        columns={"n_cells": "n1", "fraction": "frac1"}
    )[[node_key, "n1", "frac1"]]

    out = c0.merge(c1, on=node_key, how="outer")
    out["n0"] = out["n0"].fillna(0).astype(int)
    out["n1"] = out["n1"].fillna(0).astype(int)
    out["frac0"] = out["frac0"].fillna(0.0)
    out["frac1"] = out["frac1"].fillna(0.0)
    out["delta_frac"] = out["frac1"] - out["frac0"]
    out["log2_fc"] = np.log2((out["frac1"] + 1e-9) / (out["frac0"] + 1e-9))
    return out


def _compute_ood_summary(
    adata,
    *,
    ood_key: str,
    groupby: str,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    if ood_key not in adata.obs:
        raise KeyError(f"'{ood_key}' not found in adata.obs.")
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not found in adata.obs.")

    scores = pd.to_numeric(adata.obs[ood_key], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise ValueError(f"No finite scores in '{ood_key}'.")

    thr = float(np.nanquantile(scores[finite], 0.95)) if threshold is None else float(threshold)
    flagged = finite & (scores >= thr)

    df = pd.DataFrame(
        {
            "group": adata.obs[groupby].astype(str).values,
            "score": scores,
            "flagged": flagged,
        },
        index=adata.obs_names,
    )
    out = (
        df.groupby("group", observed=False)
        .agg(
            n=("group", "size"),
            flagged_n=("flagged", "sum"),
            flagged_frac=("flagged", "mean"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
        )
        .reset_index()
        .sort_values(["flagged_frac", "mean_score"], ascending=[False, False])
    )
    out["threshold_used"] = thr
    return out


def analyze_shift(
    adata,
    *,
    node_key: str,
    condition_key: str,
    group0: Any,
    group1: Any,
    basis: str = "umap",
    velocity_basis: Optional[str] = None,
    ood_key: Optional[str] = None,
    ood_groupby: Optional[str] = None,
    robustness: Optional[pd.DataFrame] = None,
    min_cells: int = 10,
    store_key: str = "shift",
    overwrite: bool = True,
):
    """
    High-level ScGeo analysis orchestrator.

    Stores a compact analysis bundle at:
        adata.uns["scgeo"][store_key]

    Stored keys
    -----------
    params
    shift_summary
    velocity_alignment   (if velocity embedding exists / requested)
    composition
    ood_summary          (if ood_key is provided)
    robustness           (if robustness DataFrame is provided)
    """
    _ensure_scgeo_uns(adata)

    if (store_key in adata.uns["scgeo"]) and (not overwrite):
        raise ValueError(
            f"adata.uns['scgeo']['{store_key}'] already exists. "
            "Use overwrite=True to replace it."
        )

    shift_summary = _compute_shift_summary(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
        basis=basis,
        min_cells=min_cells,
    )

    composition = _compute_composition(
        adata,
        node_key=node_key,
        condition_key=condition_key,
        group0=group0,
        group1=group1,
    )

    out = {
        "params": {
            "node_key": node_key,
            "condition_key": condition_key,
            "group0": group0,
            "group1": group1,
            "basis": basis,
            "velocity_basis": velocity_basis,
            "ood_key": ood_key,
            "ood_groupby": ood_groupby,
            "min_cells": min_cells,
        },
        "shift_summary": shift_summary,
        "composition": composition,
    }

    try:
        out["velocity_alignment"] = _compute_velocity_alignment(
            adata,
            shift_df=shift_summary,
            node_key=node_key,
            basis=basis,
            velocity_basis=velocity_basis,
            min_cells=min_cells,
        )
    except KeyError:
        pass

    if ood_key is not None:
        out["ood_summary"] = _compute_ood_summary(
            adata,
            ood_key=ood_key,
            groupby=ood_groupby or node_key,
        )

    if robustness is not None:
        if not isinstance(robustness, pd.DataFrame):
            raise TypeError("robustness must be a pandas DataFrame.")
        out["robustness"] = robustness.copy()

    adata.uns["scgeo"][store_key] = out
    return adata