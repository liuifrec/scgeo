from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def table(
    adata,
    store_key: str = "scgeo",
    kind: str = "shift",
    level: str = "global",  # "global" | "by" | "samples"
) -> pd.DataFrame:
    """
    Return tidy table from adata.uns[store_key][kind].
    """
    if store_key not in adata.uns:
        raise KeyError(f"adata.uns['{store_key}'] not found")
    if kind not in adata.uns[store_key]:
        raise KeyError(f"adata.uns['{store_key}']['{kind}'] not found")

    obj = adata.uns[store_key][kind]

    def rowify(name: str, d: Dict[str, Any]) -> Dict[str, Any]:
        return dict(name=name, n1=d.get("n1"), n0=d.get("n0"), delta_norm=d.get("delta_norm"))

    rows = []
    if level == "global":
        rows.append(rowify("global", obj["global"]))
    elif level == "by":
        for k, v in obj.get("by", {}).items():
            rows.append(rowify(k, v))
    elif level == "samples":
        for k, v in obj.get("samples", {}).items():
            rows.append(rowify(k, v))
    else:
        raise ValueError("level must be one of: global, by, samples")

    return pd.DataFrame(rows)
