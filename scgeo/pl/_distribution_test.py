from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ._result import PlotResult
from ._table_plot_utils import panel_axes


def _load_distribution_test_uns(adata, store_key: str):
    if "scgeo" not in adata.uns or store_key not in adata.uns["scgeo"]:
        raise KeyError(f"adata.uns['scgeo']['{store_key}'] not found. Run sg.tl.distribution_test first.")
    return adata.uns["scgeo"][store_key]


def _by_dict_to_df(by: dict) -> pd.DataFrame:
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


def distribution_test(
    adata,
    *,
    store_key: str = "distribution_test",
    level: str = "by",                 # "global" | "by"
    value: str = "p_perm",             # default focus for inference
    stat_key: str = "stat",
    top_k: Optional[int] = 10,
    highlight: str = "strongest",       # "strongest" | "weakest" | "none"
    figsize=None,
    show: bool = True,
    axes=None,
    return_data: bool = False,
    return_result: bool = False,
    separate_metrics: bool = False,
):
    """
    Plot distribution test summaries from `sg.tl.distribution_test`.

    Global: prints and plots stat + p_perm.
    By: bar chart by group; highlight top-k strongest (smallest p) groups by default.

    Assumes dict schema:
      global: {"stat":..., "p_perm":..., "n0":..., "n1":..., "n_perm":...}
      by: {group: {"stat":..., "p_perm":..., ...}, ...}

    Compose with ``axes={'summary': ax}``. ``return_result=True`` exports
    original and displayed rows without rerunning inference. The global legacy
    panel displays two different quantities; its common axis does not imply
    comparable units. ``return_data=True`` adds source tables to the tuple.
    Use ``separate_metrics=True`` for separate ``statistic`` and ``probability``
    axes in global mode; the default preserves the historical single-axis view.
    """
    obj = _load_distribution_test_uns(adata, store_key=store_key)

    if level not in ("global", "by"):
        raise ValueError("level must be 'global' or 'by'")

    if level == "global":
        g = obj.get("global", None)
        if g is None:
            raise KeyError(f"adata.uns['scgeo']['{store_key}']['global'] missing")

        stat = float(g.get(stat_key, np.nan))
        p = float(g.get(value, np.nan))

        if figsize is None:
            figsize = (5.0, 3.2)
        names = ["statistic", "probability"] if separate_metrics else ["summary"]
        fig, mapped, owns_figure = panel_axes(names, axes, figsize)
        ax = mapped[names[0]]
        if separate_metrics:
            ax.bar([stat_key], [stat])
            mapped["probability"].bar([value], [p])
            mapped["probability"].set(title=value, ylabel="Stored value")
        else:
            ax.bar([stat_key, value], [stat, p])
        ax.set_title("ScGeo: Distribution test (global)")
        ax.set_ylabel("value")

        n0 = g.get("n0", None)
        n1 = g.get("n1", None)
        n_perm = g.get("n_perm", None)
        meta = []
        if n0 is not None and n1 is not None:
            meta.append(f"n0={n0}, n1={n1}")
        if n_perm is not None:
            meta.append(f"n_perm={n_perm}")
        if meta:
            ax.text(0.99, 0.02, " | ".join(meta), ha="right", va="bottom", transform=ax.transAxes, fontsize=9)

        if show:
            plt.show()
        tables = {"global": pd.DataFrame([g]),
                  "displayed": pd.DataFrame({"metric": [stat_key, value], "value": [stat, p]})}
        if return_result:
            return PlotResult(fig, mapped, tables, {"plot": "distribution_test", "level": level,
                              "store_key": store_key, "value": value, "stat_key": stat_key,
                              "metrics_share_units": False, "separate_metrics": separate_metrics,
                              "analysis_recomputed": False},
                              warnings=[] if separate_metrics else ["The statistic and probability have different units; bar heights are not effect comparisons."])
        if return_data:
            return fig, np.array(list(mapped.values()), dtype=object), tables
        return fig, np.array(list(mapped.values()), dtype=object)

    # by
    by = obj.get("by", None)
    if by is None:
        raise KeyError(f"adata.uns['scgeo']['{store_key}']['by'] missing")

    df = _by_dict_to_df(by)
    if df.shape[0] == 0:
        raise ValueError("No per-group results in distribution_test['by'].")

    if value not in df.columns:
        raise KeyError(f"value='{value}' not found in columns: {list(df.columns)}")
    if stat_key not in df.columns:
        # stat may be missing in some by modes; allow p-only plot
        pass

    if highlight == "none" or top_k is None:
        df_plot = df.copy()
    else:
        if highlight not in ("strongest", "weakest"):
            raise ValueError("highlight must be 'strongest', 'weakest', or 'none'")
        # strongest = smallest p
        ascending = True if highlight == "strongest" else False
        df_plot = df.sort_values(value, ascending=ascending).head(int(top_k)).copy()

    if figsize is None:
        figsize = (6.0, max(3.0, 0.25 * df_plot.shape[0] + 2.0))

    fig, mapped, owns_figure = panel_axes(["summary"], axes, figsize)
    ax = mapped["summary"]

    s = df_plot[value].astype(float).sort_values(ascending=True)  # p small at top
    ax.barh(s.index.astype(str), s.values)
    ax.invert_yaxis()
    ax.set_xlabel(value)
    ax.set_title(f"Distribution test by group ({value}) | highlight={highlight} top_k={top_k}")

    if owns_figure:
        fig.tight_layout()
    if show:
        plt.show()
    tables = {"source": df, "displayed": df_plot.loc[s.index].copy()}
    if return_result:
        return PlotResult(fig, mapped, tables, {"plot": "distribution_test", "level": level,
                          "store_key": store_key, "value": value, "top_k": top_k,
                          "highlight": highlight, "analysis_recomputed": False})
    if return_data:
        return fig, np.array([ax], dtype=object), tables
    return fig, np.array([ax], dtype=object)
