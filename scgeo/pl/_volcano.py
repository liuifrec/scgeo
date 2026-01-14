from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._utils import _get_ax


def _get_dt_table(adata, key: str):
    if "scgeo" not in adata.uns:
        raise KeyError("adata.uns['scgeo'] not found")
    obj = adata.uns["scgeo"].get(key)
    if obj is None:
        raise KeyError(f"adata.uns['scgeo']['{key}'] not found")

    # common patterns
    if isinstance(obj, dict) and "table" in obj:
        obj = obj["table"]

    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return pd.DataFrame(obj)


def distribution_test_volcano(
    adata,
    *,
    key: str = "distribution_test",
    effect_col: str = "effect",
    p_col: str = "p_adj",
    label_col: str = "group",
    top_n_labels: int = 10,
    ax=None,
    figsize=(6.0, 5.0),
    title: str | None = None,
    show: bool = True,
):
    """
    Volcano-like plot for distribution/comparison results.

    x = effect (distance / signed effect)
    y = -log10(p_adj)

    Requires a table with at least: effect_col, p_col.
    Optionally label the top hits by p-value.
    """
    df = _get_dt_table(adata, key=key)

    if effect_col not in df.columns:
        raise KeyError(f"effect_col '{effect_col}' not found in table columns: {list(df.columns)}")
    if p_col not in df.columns:
        raise KeyError(f"p_col '{p_col}' not found in table columns: {list(df.columns)}")

    x = pd.to_numeric(df[effect_col], errors="coerce").to_numpy()
    p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
    y = -np.log10(np.clip(p, 1e-300, 1.0))

    fig, ax = _get_ax(ax=ax, figsize=figsize)
    ax.scatter(x, y, s=12, alpha=0.8, linewidths=0)

    # label top hits by p (smallest p => largest y)
    if label_col in df.columns and top_n_labels > 0:
        order = np.argsort(p)
        for i in order[: min(top_n_labels, len(order))]:
            if np.isfinite(x[i]) and np.isfinite(y[i]):
                ax.text(x[i], y[i], str(df.iloc[i][label_col]), fontsize=8)

    ax.set_xlabel(effect_col)
    ax.set_ylabel(f"-log10({p_col})")
    ax.set_title(title or f"Volcano: {key}")
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)

    if show and ax is None:
        plt.show()

    return fig, ax, df
