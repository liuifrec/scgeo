from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from ._utils import _get_embedding_xy, _get_ax
from ._highlight import _topk_mask_robust, _get_scanpy_palette, _fallback_palette


@dataclass
class ConsensusStructureStats:
    topk: int
    n_topk: int
    group_counts: Dict[str, int]
    group_fracs: Dict[str, float]
    group_median_score: Dict[str, float]


def consensus_structure(
    adata,
    *,
    score_key: str = "cs_score",
    basis: str = "umap",
    groupby: str = "louvain",
    topk: int = 300,

    # background (cluster-colored)
    bg_size: float = 6.0,
    bg_alpha: float = 0.25,

    # top-k overlay
    hi_size: float = 30.0,
    hi_alpha: float = 0.95,
    cmap: str = "viridis",
    outline_lw: float = 1.6,
    outline_alpha: float = 0.95,

    # palette
    use_scanpy_colors: bool = True,

    # outputs
    min_frac_in_legend: float = 0.0,   # applied to returned/printed legend_data

    # display
    title: Optional[str] = None,
    figsize=(6.5, 5.5),
    ax=None,
    show: bool = True,

    # legend handling (separate)
    print_legend: bool = True,
) -> tuple[Any, Any, ConsensusStructureStats, List[Dict[str, Any]]]:
    """
    Consensus structure (UMAP only) + legend returned separately.

    Figure:
      - Background: all cells colored by `groupby`
      - Top-k: larger points colored by `score_key`
      - Top-k edges: outlined by `groupby` palette (Scanpy palette if available)

    Outputs:
      - Returns (fig, ax, stats, legend_data)
      - legend_data is a list of dicts:
          {"group": str, "fraction": float, "count": int, "color": <mpl color>}

    Note:
      - This function NEVER draws a legend inside the plot.
      - If you want a legend figure, build it from legend_data.
    """
    if score_key not in adata.obs:
        raise KeyError(f"obs['{score_key}'] not found")
    if groupby not in adata.obs:
        raise KeyError(f"obs['{groupby}'] not found")

    x, y = _get_embedding_xy(adata, basis=basis)
    score = np.asarray(adata.obs[score_key].to_numpy(), dtype=float)
    g = adata.obs[groupby].astype(str).to_numpy()

    # palette
    pal = _get_scanpy_palette(adata, groupby) if use_scanpy_colors else None
    if pal is None:
        pal = _fallback_palette(g)

    # top-k
    mask = _topk_mask_robust(score, int(topk))
    n_topk = int(mask.sum())
    if n_topk == 0:
        raise ValueError(
            f"topk={topk} produced 0 selected cells "
            f"(score_key='{score_key}' may be all-NaN/inf or constant?)"
        )

    # per-group stats among topk
    uniq = np.unique(g)
    group_counts: Dict[str, int] = {}
    group_fracs: Dict[str, float] = {}
    group_median_score: Dict[str, float] = {}

    for k in uniq:
        mk = (g == k)
        c = int((mk & mask).sum())
        group_counts[k] = c
        group_fracs[k] = (c / n_topk) if n_topk > 0 else 0.0
        vals = score[mk & mask]
        group_median_score[k] = float(np.nanmedian(vals)) if vals.size else float("nan")

    # legend_data (sorted)
    legend_data: List[Dict[str, Any]] = []
    for k in uniq:
        frac = float(group_fracs.get(k, 0.0))
        if frac <= 0:
            continue
        if frac < float(min_frac_in_legend):
            continue
        legend_data.append(
            {
                "group": str(k),
                "fraction": frac,
                "count": int(group_counts.get(k, 0)),
                "color": pal.get(k, "k"),
            }
        )
    legend_data.sort(key=lambda d: d["fraction"], reverse=True)

    # optional print (separate from figure)
    if print_legend:
        lines = []
        lines.append(f"[scgeo] consensus_structure legend (topk={n_topk}, groupby='{groupby}')")
        for d in legend_data:
            lines.append(f"  - {d['group']}: {d['fraction']*100:5.1f}% ({d['count']} / {n_topk})")
        print("\n".join(lines))

    # ----- plot: UMAP only (no legend drawn) -----
    fig, ax = _get_ax(ax=ax, figsize=figsize)

    # background colored by group
    for k in uniq:
        m = (g == k)
        if m.any():
            ax.scatter(
                x[m], y[m],
                s=bg_size,
                c=[pal.get(k, "grey")],
                alpha=bg_alpha,
                linewidths=0,
                zorder=1,
            )

    # top-k colored by score, outlined by group color
    edgecolors = [pal.get(k, "k") for k in g[mask]]
    sc = ax.scatter(
        x[mask], y[mask],
        s=hi_size,
        c=score[mask],
        cmap=cmap,
        alpha=hi_alpha,
        linewidths=float(outline_lw),
        edgecolors=edgecolors,
        zorder=3,
    )
    sc.set_alpha(float(outline_alpha))

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(score_key)

    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title or f"Consensus structure: top-{n_topk} by {score_key} (groupby={groupby})")
    fig.tight_layout()

    stats = ConsensusStructureStats(
        topk=int(topk),
        n_topk=int(n_topk),
        group_counts=group_counts,
        group_fracs=group_fracs,
        group_median_score=group_median_score,
    )

    if show:
        plt.show()

    return fig, ax, stats, legend_data
