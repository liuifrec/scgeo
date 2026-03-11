from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def _require_uns(adata, store_key: str = "scgeo") -> dict:
    if "scgeo" not in adata.uns:
        raise KeyError("adata.uns['scgeo'] not found.")
    if store_key not in adata.uns["scgeo"]:
        raise KeyError(f"adata.uns['scgeo']['{store_key}'] not found.")
    block = adata.uns["scgeo"][store_key]
    if not isinstance(block, dict):
        raise TypeError(f"adata.uns['scgeo']['{store_key}'] must be a dict.")
    return block


def _to_dataframe(obj, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    raise TypeError(f"{name} must be a pandas DataFrame, got {type(obj)!r}.")


def get_shift_summary(
    adata,
    *,
    store_key: str = "shift",
) -> pd.DataFrame:
    block = _require_uns(adata, store_key=store_key)
    if "shift_summary" not in block:
        raise KeyError(f"'shift_summary' not found in adata.uns['scgeo']['{store_key}'].")
    return _to_dataframe(block["shift_summary"], "shift_summary")


def get_velocity_alignment_summary(
    adata,
    *,
    store_key: str = "shift",
) -> pd.DataFrame:
    block = _require_uns(adata, store_key=store_key)
    if "velocity_alignment" not in block:
        raise KeyError(f"'velocity_alignment' not found in adata.uns['scgeo']['{store_key}'].")
    return _to_dataframe(block["velocity_alignment"], "velocity_alignment")


def get_composition_table(
    adata,
    *,
    store_key: str = "shift",
) -> pd.DataFrame:
    block = _require_uns(adata, store_key=store_key)
    if "composition" not in block:
        raise KeyError(f"'composition' not found in adata.uns['scgeo']['{store_key}'].")
    return _to_dataframe(block["composition"], "composition")


def get_robustness_table(
    adata,
    *,
    store_key: str = "shift",
) -> pd.DataFrame:
    block = _require_uns(adata, store_key=store_key)
    if "robustness" not in block:
        raise KeyError(f"'robustness' not found in adata.uns['scgeo']['{store_key}'].")
    return _to_dataframe(block["robustness"], "robustness")


def get_ood_summary(
    adata,
    *,
    store_key: str = "shift",
) -> pd.DataFrame:
    block = _require_uns(adata, store_key=store_key)
    if "ood_summary" not in block:
        raise KeyError(f"'ood_summary' not found in adata.uns['scgeo']['{store_key}'].")
    return _to_dataframe(block["ood_summary"], "ood_summary")


def get_available_tables(
    adata,
    *,
    store_key: str = "shift",
) -> list[str]:
    block = _require_uns(adata, store_key=store_key)
    out = []
    for k, v in block.items():
        if isinstance(v, pd.DataFrame):
            out.append(k)
    return sorted(out)
