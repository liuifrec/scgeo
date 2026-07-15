from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .._utils import _as_2d_array, _mask_from_obs


_CENTER_ESTIMATORS = {"mean", "median", "trimmed_mean", "geometric_median"}
_BOOTSTRAP_UNITS = {"auto", "cell", "sample"}
_NORMALIZERS = {"pooled_robust_scale", "none", None}
_EPS = 1e-12


def _get_representation(adata, rep: str) -> np.ndarray:
    if rep == "X":
        return _as_2d_array(adata.X).astype(np.float64, copy=False)
    if rep not in adata.obsm:
        raise KeyError(f"obsm['{rep}'] not found")
    return _as_2d_array(adata.obsm[rep]).astype(np.float64, copy=False)


def _infer_groups(adata, condition_key: str, group0: Any, group1: Any) -> tuple[Any, Any]:
    if condition_key not in adata.obs:
        raise KeyError(f"obs key '{condition_key}' not found")

    vals = list(adata.obs[condition_key].unique())
    if group0 is None and group1 is None:
        if len(vals) != 2:
            raise ValueError(
                f"Need group0/group1 or exactly 2 unique values in obs['{condition_key}'], got {vals}"
            )
        return vals[0], vals[1]

    if group0 is None or group1 is None:
        if len(vals) != 2:
            raise ValueError(
                f"Need both group0 and group1 when obs['{condition_key}'] has {len(vals)} unique values"
            )
        if group0 is None:
            group0 = vals[0] if vals[1] == group1 else vals[1]
        if group1 is None:
            group1 = vals[0] if vals[1] == group0 else vals[1]

    return group0, group1


def _validate_params(
    center: str,
    trim_fraction: float,
    n_boot: int,
    bootstrap_unit: str,
    normalize_by: Optional[str],
    sample_key: Optional[str],
) -> str:
    if center not in _CENTER_ESTIMATORS:
        allowed = ", ".join(sorted(_CENTER_ESTIMATORS))
        raise ValueError(f"center must be one of {{{allowed}}}, got {center!r}")
    if not (0.0 <= float(trim_fraction) < 0.5):
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if int(n_boot) < 0:
        raise ValueError("n_boot must be non-negative")
    if bootstrap_unit not in _BOOTSTRAP_UNITS:
        allowed = ", ".join(sorted(_BOOTSTRAP_UNITS))
        raise ValueError(f"bootstrap_unit must be one of {{{allowed}}}, got {bootstrap_unit!r}")
    if normalize_by not in _NORMALIZERS:
        raise ValueError("normalize_by must be 'pooled_robust_scale', 'none', or None")

    if bootstrap_unit == "auto":
        return "sample" if sample_key is not None else "cell"
    if sample_key is not None and bootstrap_unit == "cell":
        raise ValueError("bootstrap_unit='cell' is not allowed when sample_key is provided")
    if sample_key is None and bootstrap_unit == "sample":
        raise ValueError("bootstrap_unit='sample' requires sample_key")
    return bootstrap_unit


def _check_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("Cannot estimate a center from zero observations")
    if not np.isfinite(X).all():
        raise ValueError("Representation contains non-finite values in the selected cells")
    return X


def _trimmed_mean(X: np.ndarray, trim_fraction: float) -> np.ndarray:
    X = _check_matrix(X)
    n = X.shape[0]
    k = int(np.floor(n * float(trim_fraction)))
    if k == 0:
        return X.mean(axis=0)
    if 2 * k >= n:
        return np.median(X, axis=0)
    return np.sort(X, axis=0)[k : n - k].mean(axis=0)


def _geometric_median(
    X: np.ndarray,
    *,
    tol: float = 1e-7,
    max_iter: int = 512,
    eps: float = _EPS,
) -> np.ndarray:
    X = _check_matrix(X)
    if X.shape[0] == 1:
        return X[0].copy()

    y = np.median(X, axis=0)
    for _ in range(int(max_iter)):
        dist = np.linalg.norm(X - y, axis=1)
        if np.all(dist <= eps):
            return y

        weights = 1.0 / np.maximum(dist, eps)
        y_next = (X * weights[:, None]).sum(axis=0) / weights.sum()
        step = np.linalg.norm(y_next - y)
        if step <= tol * max(1.0, np.linalg.norm(y)):
            return y_next
        y = y_next

    return y


def _estimate_center(X: np.ndarray, center: str, trim_fraction: float) -> np.ndarray:
    X = _check_matrix(X)
    if center == "mean":
        return X.mean(axis=0)
    if center == "median":
        return np.median(X, axis=0)
    if center == "trimmed_mean":
        return _trimmed_mean(X, trim_fraction)
    if center == "geometric_median":
        return _geometric_median(X)
    raise ValueError(f"Unsupported center estimator: {center!r}")


def _unique_in_order(values: np.ndarray) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _sample_level_centers(
    X: np.ndarray,
    mask: np.ndarray,
    sample_values: np.ndarray,
    *,
    center: str,
    trim_fraction: float,
) -> tuple[np.ndarray, list[str], list[int]]:
    sample_ids = _unique_in_order(sample_values[mask])
    centers = []
    counts = []
    for sample_id in sample_ids:
        sample_mask = mask & (sample_values == sample_id)
        centers.append(_estimate_center(X[sample_mask], center, trim_fraction))
        counts.append(int(sample_mask.sum()))
    if not centers:
        return np.empty((0, X.shape[1]), dtype=np.float64), [], []
    return np.vstack(centers), sample_ids, counts


def _robust_scale(
    X0: np.ndarray,
    X1: np.ndarray,
    center0: np.ndarray,
    center1: np.ndarray,
    normalize_by: Optional[str],
) -> tuple[float, float]:
    if normalize_by in {None, "none"}:
        return 1.0, np.nan

    d0 = np.linalg.norm(X0 - center0, axis=1)
    d1 = np.linalg.norm(X1 - center1, axis=1)
    r0 = float(np.median(d0)) if d0.size else np.nan
    r1 = float(np.median(d1)) if d1.size else np.nan
    finite = np.asarray([r0, r1], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    scale = float(np.sqrt(np.mean(np.square(finite))))
    return scale, scale


def _safe_normalized_norm(delta_norm: float, scale: float) -> float:
    if not np.isfinite(delta_norm):
        return np.nan
    if not np.isfinite(scale):
        return np.nan
    if scale <= _EPS:
        return 0.0 if delta_norm <= _EPS else np.nan
    return float(delta_norm / scale)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= _EPS or nb <= _EPS:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def _bootstrap_deltas(
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    center: str,
    trim_fraction: float,
    n_boot: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    n0, n1 = X0.shape[0], X1.shape[0]
    d = X0.shape[1]
    if n_boot == 0:
        return np.empty((0, d), dtype=np.float64)

    deltas = np.empty((int(n_boot), d), dtype=np.float64)
    for i in range(int(n_boot)):
        idx0 = rng.choice(n0, size=n0, replace=True)
        idx1 = rng.choice(n1, size=n1, replace=True)
        c0 = _estimate_center(X0[idx0], center, trim_fraction)
        c1 = _estimate_center(X1[idx1], center, trim_fraction)
        deltas[i] = c1 - c0
    return deltas


def _bootstrap_summary(boot_deltas: np.ndarray, delta: np.ndarray) -> Dict[str, float | list[float]]:
    if boot_deltas.shape[0] == 0:
        return {
            "bootstrap_magnitude_ci95": [np.nan, np.nan],
            "bootstrap_directional_resultant_length": np.nan,
            "direction_stability": np.nan,
            "sign_stability": np.nan,
        }

    norms = np.linalg.norm(boot_deltas, axis=1)
    valid_norms = norms[np.isfinite(norms)]
    if valid_norms.size == 0:
        ci = [np.nan, np.nan]
    else:
        ci = [float(x) for x in np.nanpercentile(valid_norms, [2.5, 97.5])]

    nonzero = np.isfinite(norms) & (norms > _EPS)
    if np.any(nonzero):
        directions = boot_deltas[nonzero] / norms[nonzero, None]
        resultant = float(np.linalg.norm(directions.mean(axis=0)))
    else:
        resultant = np.nan

    delta_norm = float(np.linalg.norm(delta))
    if delta_norm <= _EPS or not np.isfinite(delta_norm) or not np.any(nonzero):
        stability = np.nan
    else:
        dots = boot_deltas[nonzero] @ delta
        stability = float(np.mean(dots >= 0.0))

    return {
        "bootstrap_magnitude_ci95": ci,
        "bootstrap_directional_resultant_length": resultant,
        "direction_stability": stability,
        "sign_stability": stability,
    }


def _empty_result(
    *,
    n0: int,
    n1: int,
    n_samples0: Optional[int],
    n_samples1: Optional[int],
    estimator_params: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "center0": None,
        "center1": None,
        "delta": None,
        "delta_norm": np.nan,
        "normalized_delta_norm": np.nan,
        "bootstrap_magnitude_ci95": [np.nan, np.nan],
        "bootstrap_directional_resultant_length": np.nan,
        "direction_stability": np.nan,
        "sign_stability": np.nan,
        "n0": int(n0),
        "n1": int(n1),
        "n_cells0": int(n0),
        "n_cells1": int(n1),
        "n_samples0": n_samples0,
        "n_samples1": n_samples1,
        "estimator_params": estimator_params,
        "outlier_sensitivity": None,
    }


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _calculate_shift(
    X: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    *,
    sample_values: Optional[np.ndarray],
    center: str,
    trim_fraction: float,
    n_boot: int,
    bootstrap_unit: str,
    normalize_by: Optional[str],
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    n0 = int(mask0.sum())
    n1 = int(mask1.sum())
    estimator_params: Dict[str, Any] = {
        "center": center,
        "trim_fraction": float(trim_fraction),
        "n_boot": int(n_boot),
        "bootstrap_unit": bootstrap_unit,
        "normalize_by": normalize_by,
    }
    if n0 == 0 or n1 == 0:
        if sample_values is None:
            n_samples0 = None
            n_samples1 = None
        else:
            n_samples0 = len(_unique_in_order(sample_values[mask0])) if n0 else 0
            n_samples1 = len(_unique_in_order(sample_values[mask1])) if n1 else 0
        return _empty_result(
            n0=n0,
            n1=n1,
            n_samples0=n_samples0,
            n_samples1=n_samples1,
            estimator_params=estimator_params,
        )

    if sample_values is None:
        X0 = _check_matrix(X[mask0])
        X1 = _check_matrix(X[mask1])
        sample_ids0: list[str] = []
        sample_ids1: list[str] = []
        sample_cell_counts0: list[int] = []
        sample_cell_counts1: list[int] = []
        mean_X0 = X0
        mean_X1 = X1
        n_samples0 = None
        n_samples1 = None
    else:
        X0, sample_ids0, sample_cell_counts0 = _sample_level_centers(
            X,
            mask0,
            sample_values,
            center=center,
            trim_fraction=trim_fraction,
        )
        X1, sample_ids1, sample_cell_counts1 = _sample_level_centers(
            X,
            mask1,
            sample_values,
            center=center,
            trim_fraction=trim_fraction,
        )
        mean_X0, _, _ = _sample_level_centers(
            X,
            mask0,
            sample_values,
            center="mean",
            trim_fraction=trim_fraction,
        )
        mean_X1, _, _ = _sample_level_centers(
            X,
            mask1,
            sample_values,
            center="mean",
            trim_fraction=trim_fraction,
        )
        n_samples0 = int(X0.shape[0])
        n_samples1 = int(X1.shape[0])

    center0 = _estimate_center(X0, center, trim_fraction)
    center1 = _estimate_center(X1, center, trim_fraction)
    delta = center1 - center0
    delta_norm = float(np.linalg.norm(delta))

    scale, scale_value = _robust_scale(X0, X1, center0, center1, normalize_by)
    normalized_delta_norm = _safe_normalized_norm(delta_norm, scale)

    mean_center0 = _estimate_center(mean_X0, "mean", trim_fraction)
    mean_center1 = _estimate_center(mean_X1, "mean", trim_fraction)
    mean_delta = mean_center1 - mean_center0
    mean_delta_norm = float(np.linalg.norm(mean_delta))
    delta_difference = mean_delta - delta

    boot_deltas = _bootstrap_deltas(
        X0,
        X1,
        center=center,
        trim_fraction=trim_fraction,
        n_boot=int(n_boot),
        rng=rng,
    )
    boot = _bootstrap_summary(boot_deltas, delta)

    out: Dict[str, Any] = {
        "center0": _as_float32(center0),
        "center1": _as_float32(center1),
        "delta": _as_float32(delta),
        "delta_norm": delta_norm,
        "normalized_delta_norm": normalized_delta_norm,
        "bootstrap_magnitude_ci95": boot["bootstrap_magnitude_ci95"],
        "bootstrap_directional_resultant_length": boot["bootstrap_directional_resultant_length"],
        "direction_stability": boot["direction_stability"],
        "sign_stability": boot["sign_stability"],
        "n0": n0,
        "n1": n1,
        "n_cells0": n0,
        "n_cells1": n1,
        "n_samples0": n_samples0,
        "n_samples1": n_samples1,
        "estimator_params": estimator_params,
        "normalization": {
            "method": normalize_by,
            "scale": scale_value,
        },
        "outlier_sensitivity": {
            "mean_center0": _as_float32(mean_center0),
            "mean_center1": _as_float32(mean_center1),
            "mean_delta": _as_float32(mean_delta),
            "mean_delta_norm": mean_delta_norm,
            "robust_delta": _as_float32(delta),
            "robust_delta_norm": delta_norm,
            "delta_difference": _as_float32(delta_difference),
            "delta_difference_norm": float(np.linalg.norm(delta_difference)),
            "relative_norm_change": (
                float((delta_norm - mean_delta_norm) / mean_delta_norm)
                if mean_delta_norm > _EPS
                else np.nan
            ),
            "cosine_to_mean": _cosine(delta, mean_delta),
        },
    }
    if sample_values is not None:
        out["sample_ids0"] = sample_ids0
        out["sample_ids1"] = sample_ids1
        out["sample_cell_counts0"] = sample_cell_counts0
        out["sample_cell_counts1"] = sample_cell_counts1
    return out


def robust_shift(
    adata,
    rep: str = "X_pca",
    condition_key: str = "condition",
    group0: Any = None,
    group1: Any = None,
    by: Optional[str] = None,
    sample_key: Optional[str] = None,
    center: str = "geometric_median",
    trim_fraction: float = 0.1,
    n_boot: int = 500,
    bootstrap_unit: str = "auto",
    normalize_by: Optional[str] = "pooled_robust_scale",
    seed: int = 0,
    store_key: str = "robust_shift",
) -> Dict[str, Any]:
    """
    Robust condition displacement with sample-aware bootstrap uncertainty.

    When ``sample_key`` is provided, per-condition centers are estimated from
    sample-level centers and bootstrap resampling uses samples by default.

    Minimal example
    ---------------
    >>> out = sg.tl.robust_shift(
    ...     adata,
    ...     rep="X_pca",
    ...     condition_key="condition",
    ...     group0="control",
    ...     group1="treated",
    ...     sample_key="donor",
    ... )
    >>> out["global"]["delta_norm"]

    Stores results at ``adata.uns["scgeo"][store_key]`` and returns the same
    dictionary.
    """
    center = str(center)
    bootstrap_unit = str(bootstrap_unit)
    group0, group1 = _infer_groups(adata, condition_key, group0, group1)
    resolved_bootstrap_unit = _validate_params(
        center,
        float(trim_fraction),
        int(n_boot),
        bootstrap_unit,
        normalize_by,
        sample_key,
    )

    X = _get_representation(adata, rep)
    m0 = _mask_from_obs(adata, condition_key, group0)
    m1 = _mask_from_obs(adata, condition_key, group1)
    if int(m0.sum()) == 0 or int(m1.sum()) == 0:
        raise ValueError(
            f"Groups must be non-empty: group0={group0!r} (n={int(m0.sum())}), "
            f"group1={group1!r} (n={int(m1.sum())})"
        )

    if sample_key is not None:
        if sample_key not in adata.obs:
            raise KeyError(f"obs key '{sample_key}' not found")
        sample_values = adata.obs[sample_key].astype(str).to_numpy()
    else:
        sample_values = None

    rng = np.random.RandomState(seed)
    out: Dict[str, Any] = {
        "params": {
            "rep": rep,
            "condition_key": condition_key,
            "group0": group0,
            "group1": group1,
            "by": by,
            "sample_key": sample_key,
            "center": center,
            "trim_fraction": float(trim_fraction),
            "n_boot": int(n_boot),
            "bootstrap_unit": bootstrap_unit,
            "resolved_bootstrap_unit": resolved_bootstrap_unit,
            "normalize_by": normalize_by,
            "seed": int(seed),
            "store_key": store_key,
        },
        "global": _calculate_shift(
            X,
            m0,
            m1,
            sample_values=sample_values,
            center=center,
            trim_fraction=float(trim_fraction),
            n_boot=int(n_boot),
            bootstrap_unit=resolved_bootstrap_unit,
            normalize_by=normalize_by,
            rng=rng,
        ),
    }

    if by is not None:
        if by not in adata.obs:
            raise KeyError(f"obs key '{by}' not found")
        by_values = adata.obs[by].astype(str).to_numpy()
        out_by: Dict[str, Any] = {}
        for level in _unique_in_order(by_values):
            level_mask = by_values == level
            out_by[level] = _calculate_shift(
                X,
                m0 & level_mask,
                m1 & level_mask,
                sample_values=sample_values,
                center=center,
                trim_fraction=float(trim_fraction),
                n_boot=int(n_boot),
                bootstrap_unit=resolved_bootstrap_unit,
                normalize_by=normalize_by,
                rng=rng,
            )
        out["by"] = out_by

    adata.uns.setdefault("scgeo", {})
    adata.uns["scgeo"][store_key] = out
    return out
