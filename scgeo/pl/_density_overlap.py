from __future__ import annotations

from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_density_overlap_uns(adata, store_key: str):
    if "scgeo" not in adata.uns or store_key not in adata.uns["scgeo"]:
        raise KeyError(
            f"adata.uns['scgeo']['{store_key}'] not found. Run sg.tl.density_overlap first."
        )
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
    return pd.DataFrame(rows).set_index("group")


def _lollipop_h(ax, labels, values, *, annotate: bool = True):
    y = np.arange(len(labels))
    ax.hlines(y, 0, values, linewidth=2)
    ax.plot(values, y, marker="o", linestyle="None")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if annotate:
        # small right padding
        xmax = float(np.nanmax(values)) if len(values) else 1.0
        pad = 0.02 * (xmax if xmax > 0 else 1.0)
        for yi, v in zip(y, values):
            if np.isfinite(v):
                ax.text(v + pad, yi, f"{v:.3g}", va="center", fontsize=8)


def _lollipop_v(ax, labels, values, *, annotate: bool = True):
    x = np.arange(len(labels))
    ax.vlines(x, 0, values, linewidth=2)
    ax.plot(x, values, marker="o", linestyle="None")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if annotate:
        ymax = float(np.nanmax(values)) if len(values) else 1.0
        pad = 0.02 * (ymax if ymax > 0 else 1.0)
        for xi, v in zip(x, values):
            if np.isfinite(v):
                ax.text(xi, v + pad, f"{v:.3g}", ha="center", fontsize=8)


def density_overlap(
    adata,
    *,
    store_key: str = "density_overlap",
    level: str = "by",  # "global" | "by"
    metrics: Tuple[str, ...] = ("bc", "hellinger"),
    top_k: Optional[int] = 10,
    highlight: str = "worst",  # "worst" | "best" | "none"
    sort_by: Optional[str] = None,
    style: Literal["bar", "lollipop"] = "lollipop",
    annotate: bool = True,
    grid: bool = True,
    figsize=None,
    show: bool = True,
):
    """Plot density overlap summaries from ``sg.tl.density_overlap``.

    Global:
      - Shows one summary value per metric.

    By-group:
      - Shows per-group values for each metric.
      - ``highlight`` + ``top_k`` helps focus on worst/best groups.

    Notes:
      - For Bhattacharyya coefficient (bc): **lower = more different**.
      - For Hellinger distance: **higher = more different**.
    """

    obj = _load_density_overlap_uns(adata, store_key=store_key)

    if level not in ("global", "by"):
        raise ValueError("level must be 'global' or 'by'")

    if level == "global":
        g = obj.get("global", None)
        if g is None:
            raise KeyError(f"adata.uns['scgeo']['{store_key}']['global'] missing")

        vals = np.array([float(g.get(m, np.nan)) for m in metrics], dtype=float)
        labs = [str(m) for m in metrics]

        if figsize is None:
            figsize = (max(4.5, 1.2 * len(metrics) + 2.5), 3.2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if style == "lollipop":
            _lollipop_v(ax, labs, vals, annotate=annotate)
        else:
            ax.bar(labs, vals)
            if annotate:
                ymax = float(np.nanmax(vals)) if len(vals) else 1.0
                pad = 0.02 * (ymax if ymax > 0 else 1.0)
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        ax.text(i, v + pad, f"{v:.3g}", ha="center", fontsize=8)

        ax.set_title("ScGeo: Density overlap (global)")
        ax.set_ylabel("value")

        if grid:
            ax.grid(True, axis="y", alpha=0.25)

        # annotate n0/n1 if present
        n0 = g.get("n0", None)
        n1 = g.get("n1", None)
        if n0 is not None and n1 is not None:
            ax.text(
                0.99,
                0.02,
                f"n0={n0}, n1={n1}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontsize=9,
            )

        fig.tight_layout()
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
        sort_by = "bc" if "bc" in df.columns else (metrics[0] if metrics else df.columns[0])

    if sort_by not in df.columns:
        raise KeyError(f"sort_by='{sort_by}' not found in columns: {list(df.columns)}")

    # determine “difference direction”
    worse_is = "low" if sort_by == "bc" else "high"  # bc low=worse; hellinger high=worse

    if highlight == "none" or top_k is None:
        df_plot = df.copy()
    else:
        if highlight not in ("worst", "best"):
            raise ValueError("highlight must be 'worst', 'best', or 'none'")

        # For "worst": choose low for bc, high for others.
        want_low = (worse_is == "low")
        ascending = want_low if highlight == "worst" else (not want_low)
        df_plot = df.sort_values(sort_by, ascending=ascending).head(int(top_k)).copy()

    # plotting: one panel per metric
    n_panels = len(metrics)
    if figsize is None:
        figsize = (5.2 * n_panels, max(3.0, 0.28 * df_plot.shape[0] + 2.0))
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes[0]

    # ---- NEW: pick ONE shared order (based on sort_by + highlight) ----
    # For bc: "worse" = low; for others (e.g. hellinger): "worse" = high
    sort_worse_is_low = (sort_by == "bc")
    if highlight == "best":
        sort_ascending = not sort_worse_is_low
    else:
        # worst or none -> use "worst-first" order
        sort_ascending = sort_worse_is_low

    order = df_plot[sort_by].astype(float).sort_values(ascending=sort_ascending).index.tolist()

    # global reference (optional)
    g = obj.get("global", {}) if isinstance(obj.get("global", {}), dict) else {}

    def _metric_note(m: str) -> str:
        if m == "bc":
            return " (lower=worse)"
        if m == "hellinger":
            return " (higher=worse)"
        return ""

    for j, m in enumerate(metrics):
        ax = axes[j]
        if m not in df_plot.columns:
            ax.set_axis_off()
            continue

        # ---- NEW: use the shared order for every panel ----
        s = df_plot.loc[order, m].astype(float)
        labels = s.index.astype(str).tolist()
        values = s.values.astype(float)

        if style == "lollipop":
            _lollipop_h(ax, labels, values, annotate=annotate)
        else:
            ax.barh(labels, values)

        # Put "worst" at top visually (more natural reading order)
        ax.invert_yaxis()

        ax.set_title(f"Density overlap by group: {m}{_metric_note(m)}")
        ax.set_xlabel(m)

        # ---- NEW: stable x-lims ----
        if m == "bc":
            ax.set_xlim(0.0, 1.0)
        else:
            vmax = float(np.nanmax(values)) if len(values) else 1.0
            ax.set_xlim(0.0, max(1e-9, 1.05 * vmax))
        ax.margins(x=0.02)
        
        # ---- NEW: optional global reference line ----
        gv = g.get(m, None)
        if gv is not None and np.isfinite(float(gv)):
            ax.axvline(float(gv), linestyle="--", linewidth=1.0, alpha=0.3)

        if grid:
            ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle(
        f"ScGeo: Density overlap (by) | highlight={highlight} top_k={top_k} sort_by={sort_by}",
        y=1.02,
    )
    fig.tight_layout()

    if show:
        plt.show()
    return fig, axes
