"""Read biological sample centers retained by :func:`scgeo.tl.robust_shift`."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd


def sample_centers(adata_or_result, *, store_key="robust_shift", scope="auto"):
    """Return a tidy copy of the stored biological sample centers.

    Parameters
    ----------
    adata_or_result
        AnnData with a stored robust-shift result, or that result dictionary.
        Old results without sample-center matrices are not recomputed silently.
    store_key
        Key inside ``adata.uns['scgeo']``; ignored for a result dictionary.
    scope
        ``'state'`` selects stratified results, ``'global'`` selects the global
        result, and ``'auto'`` selects states when available, otherwise global.

    Returns
    -------
    pandas.DataFrame
        Columns ``scope``, ``state``, ``condition``, ``sample``, ``coordinate``,
        ``value``, ``cell_count``, ``representation``, and ``center``. Coordinates
        are zero-based. Each sample has one row per representation coordinate;
        ``cell_count`` is descriptive and must not be used as a replicate weight.
        The global state label is ``'__global__'``. Provenance is in ``attrs``.
    """
    if scope not in {"auto", "state", "global"}:
        raise ValueError("scope must be 'auto', 'state', or 'global'.")
    if isinstance(adata_or_result, Mapping):
        result = adata_or_result
    else:
        store = adata_or_result.uns.get("scgeo", {})
        if store_key not in store:
            raise KeyError(f"No robust-shift result at adata.uns['scgeo'][{store_key!r}].")
        result = store[store_key]
    if not isinstance(result, Mapping):
        raise TypeError("The robust-shift result must be a dictionary.")
    params = result.get("params", {})
    if not params.get("sample_key"):
        raise ValueError(
            "Biological sample centers require robust_shift(..., sample_key='...'). "
            "Cell-level results cannot be treated as biological replicates."
        )
    selected_scope = "state" if scope == "auto" and "by" in result else scope
    if selected_scope == "auto":
        selected_scope = "global"
    if selected_scope == "state":
        blocks = result.get("by")
        if not isinstance(blocks, Mapping) or not blocks:
            raise ValueError("No state-level results; use scope='global' or rerun robust_shift with by=.")
    else:
        blocks = {"__global__": result.get("global", {})}
    records = []
    n_dimensions = None
    for state, block in blocks.items():
        for suffix in ("0", "1"):
            key = f"sample_centers{suffix}"
            if key not in block:
                raise ValueError(
                    "This result predates stored sample centers. Rerun "
                    "sg.tl.robust_shift(..., sample_key='...') with the original "
                    "analysis parameters; no values have been reconstructed."
                )
            centers = np.asarray(block[key], dtype=float)
            ids = list(block.get(f"sample_ids{suffix}", []))
            counts = list(block.get(f"sample_cell_counts{suffix}", []))
            if centers.ndim != 2 or centers.shape[0] != len(ids) or len(ids) != len(counts):
                raise ValueError(f"Malformed stored sample centers for state {state!r}, group {suffix}.")
            if n_dimensions is not None and n_dimensions != centers.shape[1]:
                raise ValueError("Stored sample centers use inconsistent coordinate counts.")
            n_dimensions = centers.shape[1]
            if len(set(map(str, ids))) != len(ids):
                raise ValueError(f"Duplicate sample identifiers for state {state!r}, group {suffix}.")
            for sample, count, values in zip(ids, counts, centers):
                for coordinate, value in enumerate(values):
                    records.append({
                        "scope": selected_scope,
                        "state": str(state),
                        "condition": str(params[f"group{suffix}"]),
                        "sample": str(sample),
                        "coordinate": coordinate,
                        "value": float(value),
                        "cell_count": int(count),
                        "representation": str(params.get("rep", "unknown")),
                        "center": str(params.get("center", "unknown")),
                    })
    columns = ["scope", "state", "condition", "sample", "coordinate", "value",
               "cell_count", "representation", "center"]
    table = pd.DataFrame.from_records(records, columns=columns)
    table.attrs["provenance"] = deepcopy(params)
    table.attrs["sample_center_schema_version"] = int(result.get("sample_center_schema_version", 1))
    table.attrs["scope"] = selected_scope
    table.attrs["coordinate_indexing"] = "zero-based"
    table.attrs["n_dimensions"] = n_dimensions
    table.attrs["inference"] = "Descriptive biological sample centers; no additional inference was computed."
    return table
