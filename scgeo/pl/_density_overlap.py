from __future__ import annotations

from typing import Optional, Iterable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_density_overlap_uns(adata, store_key: str):
    if "scgeo" not in adata.uns or store_key not in adata.uns["scgeo"]:
        raise KeyError(f"adata.uns['scgeo']['{store_key}'] not found. Run sg.tl.density_overlap first.")
    return adata.uns["scgeo"][store_key]


def _by_dict_to_df(by: dict) -> pd.DataFrame:
    # by: {group: {"bc":..., "hellinger":..., "n0":..., "n1":...}, ...}
    rows = []
    for k, v in by.items():
        if v is None:
            continue
        row = {"group": str(k)}
        if isinstance(v, dict):
            row.update(v)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["group"])
    df = pd.DataFrame(rows).set_index("group")
    return df


def density_overlap(
    adata,
    *,
    store_key: str = "density_overlap",
    level: str = "by",              # "global" | "by"
    metrics: Tuple[str, ...] = ("bc", "hellinger"),
    top_k: Optional[int] = 10,
    highlight: str = "worst",       # "worst" | "best" | "none"
    sort_by: Optional[str] = None,  # default depends on metric
    figsize=None,
    show: bool = True,
):
    """
    Plot density overlap summaries from `sg.tl.density_overlap`.

    Global: bar chart of metrics.
    By: bar chart per group; supports highlighting top-k worst/best groups.

    Notes:
    - For Bhattacharyya coefficient (bc): lower = more different.
    - For Hellinger distance: higher = more different.
    """
    obj = _load_density_overlap_uns(adata, store_key=store_key)

    if level not in ("global", "by"):
        raise ValueError("level must be 'global' or 'by'")

    if level == "global":
        g = obj.get("global", None)
        if g is None:
            raise KeyError(f"adata.uns['scgeo']['{store_key}']['global'] missing")

        vals = [float(g.get(m, np.nan)) for m in metrics]
        labs = list(metrics)

        if figsize is None:
            figsize = (4 + 0.9 * len(metrics), 3.2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.bar(labs, vals)
        ax.set_title("ScGeo: Density overlap (global)")
        ax.set_ylabel("value")

        # annotate n0/n1 if present
        n0 = g.get("n0", None)
        n1 = g.get("n1", None)
        if n0 is not None and n1 is not None:
            ax.text(
                0.99, 0.02, f"n0={n0}, n1={n1}",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=9
            )

        if show:
            plt.show()
        return fig, np.array([ax], dtype=object)

    # level == "by"
    by = obj.get("by", None)
    if by is None:
        raise KeyError(f"adata.uns['scgeo']['{store_key}']['by'] missing")

    df = _by_dict_to_df(by)
    if df.shape[0] == 0:
        raise ValueError("No per-group results in density_overlap['by'].")

    # choose default sort key
    if sort_by is None:
        # prefer bc if present, else hellinger
        sort_by = "bc" if "bc" in df.columns else (metrics[0] if metrics else df.columns[0])

    if sort_by not in df.columns:
        raise KeyError(f"sort_by='{sort_by}' not found in columns: {list(df.columns)}")

    # determine “difference direction”
    # bc lower = worse, hellinger higher = worse
    worse_is = "low" if sort_by == "bc" else "high"

    if highlight == "none" or top_k is None:
        df_plot = df.copy()
    else:
        if highlight not in ("worst", "best"):
            raise ValueError("highlight must be 'worst', 'best', or 'none'")
        ascending = (worse_is == "low") if highlight == "worst" else (worse_is != "low")
        df_plot = df.sort_values(sort_by, ascending=ascending).head(int(top_k)).copy()

    # plotting: one panel per metric
    n_panels = len(metrics)
    if figsize is None:
        figsize = (4.8 * n_panels, max(3.0, 0.25 * df_plot.shape[0] + 2.0))
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes[0]

    for j, m in enumerate(metrics):
        ax = axes[j]
        if m not in df_plot.columns:
            ax.set_axis_off()
            continue

        s = df_plot[m].astype(float)
        # order bars for readability
        s = s.sort_values(ascending=(m == "bc"))  # bc ascending; hellinger descending-ish later
        ax.barh(s.index.astype(str), s.values)
        ax.set_title(f"Density overlap by group: {m}")
        ax.set_xlabel(m)

    fig.suptitle(f"ScGeo: Density overlap (by) | highlight={highlight} top_k={top_k} sort_by={sort_by}", y=1.02)

    fig.tight_layout()

    if show:
        plt.show()
    return fig, axes
