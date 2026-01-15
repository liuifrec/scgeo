from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def table(
    adata,
    store_key: str = "scgeo",
    kind: str = "shift",
    level: str = "global",  # "global" | "by" | "samples" | (map_query_to_ref: "per_label")
) -> pd.DataFrame:
    """
    Return tidy table from adata.uns[store_key][kind].

    Conventions:
      - shift-like objects: expect keys: ["global"] and optionally ["by"], ["samples"].
      - paga_composition_stats: expects obj["table"] as a DataFrame-like.
      - map_query_to_ref: expects obj["global"] and either obj["per_label_dict"] or obj["per_label"].
    """
    if store_key not in adata.uns:
        raise KeyError(f"adata.uns['{store_key}'] not found")
    if kind not in adata.uns[store_key]:
        raise KeyError(f"adata.uns['{store_key}']['{kind}'] not found")

    obj = adata.uns[store_key][kind]

    # --- special case: paga_composition_stats stores a ready table
    if kind == "paga_composition_stats":
        df = obj.get("table", None)
        if df is None:
            raise KeyError(f"adata.uns['{store_key}']['{kind}']['table'] not found")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return df.copy()

    # --- special case: map_query_to_ref summary table
    if kind == "map_query_to_ref":
        rows = []

        if level == "global":
            g = obj.get("global", None)
            if g is None:
                raise KeyError(f"adata.uns['{store_key}']['{kind}']['global'] not found")
            rows.append({"name": "global", **dict(g)})
            return pd.DataFrame(rows)

        if level in ("by", "per_label"):
            d = obj.get("per_label_dict", None)
            if d is not None:
                for lab, stats in d.items():
                    rows.append({"name": str(lab), **dict(stats)})
                return pd.DataFrame(rows)

            df = obj.get("per_label", None)
            if df is None:
                raise KeyError(
                    f"adata.uns['{store_key}']['{kind}']['per_label_dict' or 'per_label'] not found"
                )
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if "label" in df.columns and "name" not in df.columns:
                df = df.rename(columns={"label": "name"})
            return df.copy()

        raise ValueError("level must be one of: global, by/per_label (for map_query_to_ref)")

    # --- default behavior: shift-like objects
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
