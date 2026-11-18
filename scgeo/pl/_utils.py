from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_embedding(adata, basis="umap"):
    """Resolve a named embedding once, accepting 'umap' and 'X_umap'."""
    key = str(basis) if str(basis) in adata.obsm or str(basis).startswith("X_") else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"{key} not found in adata.obsm")
    coords = np.asarray(adata.obsm[key])
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != adata.n_obs:
        raise ValueError(f"{key} must have n_obs rows and at least two columns; got {coords.shape}.")
    return key, coords


def display_indices(labels, max_points=None, random_state=0, required_indices=None):
    """Select deterministic display rows while retaining each group/highlight.

    The budget is strict. Raise if it cannot retain one point from every group
    and all required rows. This function never changes analytical summaries.
    """
    import pandas as pd

    labels = np.asarray(labels, dtype=object)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    n = len(labels)
    if required_indices is None:
        required = np.empty(0, dtype=int)
    else:
        raw = np.asarray(required_indices)
        if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
            raise ValueError("required_indices must contain integer positions.")
        required = np.unique(raw.astype(int))
        if np.any(required < 0) or np.any(required >= n):
            raise ValueError("required_indices contains an out-of-range position.")
    if max_points is None:
        return np.arange(n)
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points < 1:
        raise ValueError("max_points must be a positive integer or None.")
    if max_points >= n:
        return np.arange(n)
    if len(required) > max_points:
        raise ValueError("max_points is smaller than the number of required highlights.")
    # Factorization keeps missing labels as their own group and avoids string
    # coercion merging a genuine category with a missing-value sentinel.
    codes, _ = pd.factorize(labels, sort=False)
    rng = np.random.default_rng(random_state)
    selected = set(required.tolist())
    represented = set(codes[required].tolist())
    for code in np.unique(codes):
        if code not in represented:
            selected.add(int(rng.choice(np.flatnonzero(codes == code))))
    if len(selected) > max_points:
        raise ValueError("max_points cannot preserve all groups and required highlights; increase it.")
    remaining = np.setdiff1d(np.arange(n), np.fromiter(selected, dtype=int), assume_unique=False)
    if len(selected) < max_points:
        selected.update(rng.choice(remaining, size=max_points-len(selected), replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def _get_embedding_xy(adata, basis: str = "umap"):
    _, X = resolve_embedding(adata, basis)
    return X[:, 0], X[:, 1]


def _get_ax(ax=None, figsize=(5.0, 4.5)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax
    return ax.figure, ax


def _topk_mask(values: np.ndarray, k: int):
    values = np.asarray(values)
    finite = np.isfinite(values)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return np.zeros(values.shape[0], dtype=bool)
    k = int(min(k, idx.size))
    top_idx = idx[np.argsort(values[idx])[-k:]]
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[top_idx] = True
    return mask


def observation_labels(series, name):
    """Keep true missing entries apart from literal labels such as 'NA'."""
    raw = pd.Series(series).reset_index(drop=True)
    present = pd.unique(raw.dropna().astype(object))
    names = [str(x) for x in present]
    if len(names) != len(set(names)):
        raise ValueError(f"{name} has distinct values with colliding text labels.")
    missing = "(missing)"
    while missing in names:
        missing = f"({missing})"
    mapped = raw.map(lambda x: missing if pd.isna(x) else str(x))
    if isinstance(series.dtype, pd.CategoricalDtype):
        ordered = [str(x) for x in series.cat.categories if str(x) in names]
        if len(ordered) != len(set(ordered)):
            raise ValueError(f"{name} has colliding categorical levels.")
    else:
        ordered = names
    if raw.isna().any():
        ordered.append(missing)
    return mapped, ordered, missing if raw.isna().any() else None
