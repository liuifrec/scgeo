from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, is_color_like

from ._highlight import _get_scanpy_palette, _topk_mask_robust
from ._result import PlotResult
from ._style import category_palette
from ._table_plot_utils import panel_axes
from ._utils import display_indices, observation_labels, resolve_embedding


@dataclass
class ConsensusStructureStats:
    topk: Optional[int]
    n_topk: int
    group_counts: dict[str, int]
    group_fracs: dict[str, float]
    group_median_score: dict[str, float]
    selection: str = "topk"
    threshold: Optional[float] = None

    @property
    def n_selected(self):
        """Selected count; ``n_topk`` remains available for legacy callers."""
        return self.n_topk


def _prevalence_tables(groups, score, selected, samples=None, missing_sample=None):
    """Describe selection using full input denominators before drawing filters."""
    source = pd.DataFrame(
        {
            "group": groups,
            "scored": np.isfinite(score),
            "selected": selected,
            "selected_score": np.where(selected, score, np.nan),
        }
    )
    aggregates = {
        "n_cells": ("selected", "size"),
        "n_scored": ("scored", "sum"),
        "count": ("selected", "sum"),
        "median_score": ("selected_score", "median"),
    }
    summary = source.groupby("group", sort=True, observed=True).agg(**aggregates)
    n_selected = int(selected.sum())
    summary["fraction"] = summary["count"] / n_selected if n_selected else 0.0
    sample_table = None
    if samples is not None:
        source["sample"] = np.asarray(samples)
        index = pd.MultiIndex.from_product(
            [summary.index, pd.unique(source["sample"])], names=["group", "sample"]
        )
        sample_table = (
            source.groupby(["group", "sample"], observed=True, sort=False)
            .agg(**aggregates)
            .reindex(index)
        )
        for column in ("n_cells", "n_scored", "count"):
            sample_table[column] = sample_table[column].fillna(0).astype(int)
        totals = source.groupby("sample", observed=True, sort=False).agg(
            sample_n_cells=("selected", "size"),
            sample_n_scored=("scored", "sum"),
            sample_n_selected=("selected", "sum"),
        )
        for column in totals:
            sample_table[column] = sample_table.index.get_level_values("sample").map(
                totals[column]
            )
        sample_table["fraction"] = sample_table[
            "count"
        ] / sample_table.sample_n_selected.replace(0, np.nan)
        sample_table["sample_is_missing"] = (
            sample_table.index.get_level_values("sample") == missing_sample
        )
        sample_table["sample_status"] = np.where(
            sample_table.sample_is_missing, "missing_sample_id", "identified_sample"
        )
    for table in (summary, sample_table):
        if table is None:
            continue
        table["n_unscored"] = table.n_cells - table.n_scored
        table["score_coverage"] = table.n_scored / table.n_cells.replace(0, np.nan)
        table["selected_fraction_all_cells"] = table["count"] / table.n_cells.replace(
            0, np.nan
        )
        table["selected_fraction_scored_cells"] = table[
            "count"
        ] / table.n_scored.replace(0, np.nan)
        table["status"] = np.select(
            [table.n_cells.eq(0), table.n_scored.eq(0)],
            ["no_cells_in_group", "no_finite_scores"],
            default="available",
        )
    return (
        summary.reset_index(),
        None if sample_table is None else sample_table.reset_index(),
    )


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
    min_frac_in_legend: float = 0.0,  # applied to returned/printed legend_data
    # display
    title: Optional[str] = None,
    figsize=(6.5, 5.5),
    ax=None,
    show: bool = True,
    # legend handling (separate)
    print_legend: bool = True,
    colorbar: bool = True,
    cax=None,
    norm=None,
    vmin=None,
    vmax=None,
    return_result: bool = False,
    selection: str = "topk",
    threshold=None,
    sample_key=None,
    max_points=None,
    random_state=0,
    rasterized="auto",
):
    """
    Consensus score structure on a stored embedding, with a separate legend.

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
      - ``return_result=True`` exports all cells, selection/display flags and
        full group summaries (including groups excluded from the legend).
      - ``colorbar=False`` or ``cax`` preserves supplied panel bounds. An
        automatic colorbar reserves space from the embedding axis. Supplied
        figures are never laid out or resized. The four-item return is unchanged.

    ``selection='threshold'`` selects finite scores >= the explicit finite
    ``threshold`` and ignores ``topk``. An empty threshold selection is valid.
    The default top-k rule and deterministic tie order are unchanged.
    ``summary.fraction`` describes the selected set's group composition;
    ``selected_fraction_all_cells`` and ``selected_fraction_scored_cells`` use
    each group's full input or finite-score denominator. ``score_coverage`` is
    the finite-score fraction. Optional ``samples`` exports the full group×sample
    roster, including absent combinations with zero counts and unavailable
    within-group rates. Its ``fraction`` denominator is selected cells in the
    same sample. Missing sample IDs are explicitly marked, not identified as
    biological samples. No statistical test or uncertainty is fitted.

    ``max_points`` limits visible cells only, retaining every selected cell
    with finite coordinates and one cell per group (group×sample when supplied).
    A budget too small for these requirements raises. Source counts, selection,
    and color normalization precede display filtering. ``rasterized='auto'``
    rasterizes cell layers above 2,000 displayed cells. ``hi_alpha`` controls
    score faces independently of ``outline_alpha``; the score colorbar remains
    opaque and uses the same normalization.
    """
    if score_key not in adata.obs:
        raise KeyError(f"obs['{score_key}'] not found")
    if groupby not in adata.obs:
        raise KeyError(f"obs['{groupby}'] not found")
    if sample_key is not None and sample_key not in adata.obs:
        raise KeyError(f"obs['{sample_key}'] not found")

    if isinstance(topk, bool) or not isinstance(topk, (int, np.integer)) or topk < 0:
        raise ValueError("topk must be a nonnegative integer.")
    if selection not in {"topk", "threshold"}:
        raise ValueError("selection must be 'topk' or 'threshold'.")
    if selection == "threshold":
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(threshold)
        ):
            raise ValueError("threshold selection requires a finite numeric threshold.")
        threshold = float(threshold)
    elif threshold is not None:
        raise ValueError("threshold requires selection='threshold'.")
    if (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be a nonnegative integer.")
    if not isinstance(rasterized, (bool, np.bool_)) and rasterized != "auto":
        raise ValueError("rasterized must be boolean or 'auto'.")
    for name, value in (
        ("bg_alpha", bg_alpha),
        ("hi_alpha", hi_alpha),
        ("outline_alpha", outline_alpha),
        ("min_frac_in_legend", min_frac_in_legend),
    ):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between zero and one.")
    for name, value in (
        ("bg_size", bg_size),
        ("hi_size", hi_size),
        ("outline_lw", outline_lw),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("Supply norm or vmin/vmax, not both.")
    if norm is not None and not isinstance(norm, Normalize):
        raise TypeError("norm must be a Matplotlib Normalize instance.")
    for name, value in (("vmin", vmin), ("vmax", vmax)):
        if value is not None and (
            isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite or None.")
    if vmin is not None and vmax is not None and vmin > vmax:
        raise ValueError("vmin must not exceed vmax.")
    cmap_object = plt.get_cmap(cmap)
    if ax is not None:
        panel_axes(["embedding"], {"embedding": ax}, None)
    if cax is not None:
        if ax is None or not colorbar:
            raise ValueError("cax requires ax and colorbar=True.")
        panel_axes(["embedding", "colorbar"], {"embedding": ax, "colorbar": cax}, None)
    key, xy = resolve_embedding(adata, basis=basis)
    x, y = xy[:, 0], xy[:, 1]
    finite_xy = np.isfinite(xy[:, :2]).all(axis=1)
    score = pd.to_numeric(adata.obs[score_key], errors="raise").to_numpy(
        dtype=float, na_value=np.nan
    )
    groups, categories, missing = observation_labels(adata.obs[groupby], groupby)
    g = groups.to_numpy()
    samples, missing_sample = None, None
    if sample_key is not None:
        samples, _, missing_sample = observation_labels(
            adata.obs[sample_key], sample_key
        )

    # palette
    pal = _get_scanpy_palette(adata, groupby) if use_scanpy_colors else None
    if pal is None or not all(is_color_like(c) for c in pal.values()):
        pal = category_palette(categories)
        palette_origin = "deterministic_hash"
    else:
        palette_origin = "scanpy_uns"
    if missing is not None:
        pal[missing] = "lightgrey"

    # The default top-k rule is unchanged; threshold selection is explicit.
    mask = (
        _topk_mask_robust(score, int(topk))
        if selection == "topk"
        else np.isfinite(score) & (score >= threshold)
    )
    n_topk = int(mask.sum())
    if selection == "topk" and n_topk == 0 and topk != 0:
        raise ValueError(
            f"topk={topk} produced 0 selected cells "
            f"(score_key='{score_key}' may be all-NaN/inf or constant?)"
        )

    summary, sample_table = _prevalence_tables(g, score, mask, samples, missing_sample)
    uniq = summary.group.to_numpy()
    group_counts = dict(zip(uniq, summary["count"].astype(int)))
    group_fracs = dict(zip(uniq, summary.fraction))
    group_median_score = dict(zip(uniq, summary.median_score))

    # Validate display budget and normalization before changing any axes.
    eligible = np.flatnonzero(finite_xy)
    required = np.flatnonzero(mask[eligible])
    strata = g[eligible]
    if samples is not None:
        strata = pd.MultiIndex.from_arrays([strata, samples.iloc[eligible]]).to_numpy()
    try:
        chosen = display_indices(
            strata,
            max_points=max_points,
            random_state=int(random_state),
            required_indices=required,
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot satisfy max_points with {len(required)} selected visible cells "
            f"and {len(pd.unique(strata))} observed display strata: {exc}"
        ) from exc
    background = np.zeros(adata.n_obs, dtype=bool)
    background[eligible[chosen]] = True
    displayed = mask & finite_xy
    raster = bool(background.sum() > 2000) if rasterized == "auto" else bool(rasterized)
    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    scale_values = score[mask] if n_topk else np.array([0.0, 1.0])
    norm.autoscale_None(scale_values)
    norm(scale_values)
    if not np.isfinite([norm.vmin, norm.vmax]).all() or norm.vmin > norm.vmax:
        raise ValueError(
            "Resolved score normalization bounds must be finite and ordered."
        )

    # legend_data (sorted)
    legend_data: list[dict[str, Any]] = []
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
        description = (
            f"topk={n_topk}"
            if selection == "topk"
            else f"score >= {threshold:g}, n={n_topk}"
        )
        lines.append(
            f"[scgeo] consensus_structure legend ({description}, groupby='{groupby}')"
        )
        for d in legend_data:
            lines.append(
                f"  - {d['group']}: {d['fraction'] * 100:5.1f}% ({d['count']} / {n_topk})"
            )
        print("\n".join(lines))

    # ----- plot: UMAP only (no legend drawn) -----
    fig, panels, owns = panel_axes(
        ["embedding"], None if ax is None else {"embedding": ax}, figsize
    )
    ax = panels["embedding"]
    artists = {"background": {}}

    # background colored by group
    for k in uniq:
        m = (g == k) & background
        if m.any():
            artists["background"][k] = ax.scatter(
                x[m],
                y[m],
                s=bg_size,
                c=[pal.get(k, "grey")],
                alpha=bg_alpha,
                linewidths=0,
                zorder=1,
                rasterized=raster,
            )

    # Separate score and outline collections preserve independent opacity.
    edgecolors = [pal.get(k, "k") for k in g[displayed]]
    sc = ax.scatter(
        x[displayed],
        y[displayed],
        s=hi_size,
        c=score[displayed],
        cmap=cmap_object,
        norm=norm,
        alpha=hi_alpha,
        linewidths=0,
        edgecolors="none",
        zorder=3,
        rasterized=raster,
    )
    artists["selected"] = sc
    artists["selected_outlines"] = ax.scatter(
        x[displayed],
        y[displayed],
        s=hi_size,
        facecolors="none",
        edgecolors=edgecolors,
        linewidths=outline_lw,
        alpha=outline_alpha,
        zorder=4,
        rasterized=raster,
    )
    mappable = ScalarMappable(norm=norm, cmap=cmap_object)
    mappable.set_array(score[mask])
    artists["score_mappable"] = mappable

    if colorbar:
        cb = fig.colorbar(mappable, ax=ax, cax=cax, fraction=0.046, pad=0.04)
        cb.set_label(score_key)
        artists["colorbar"] = cb

    basis_name = key.removeprefix("X_").upper()
    if finite_xy.any():
        # Subsampling the background must not crop the full input landscape.
        ax.update_datalim(xy[finite_xy, :2])
        ax.autoscale_view()
    ax.set_xlabel(f"{basis_name}1")
    ax.set_ylabel(f"{basis_name}2")
    ax.set_aspect("equal", adjustable="datalim")
    default_title = (
        f"Consensus structure: top-{n_topk} by {score_key} (groupby={groupby})"
        if selection == "topk"
        else f"Consensus structure: {score_key} ≥ {threshold:g} (n={n_topk}, groupby={groupby})"
    )
    ax.set_title(title or default_title)
    if owns:
        fig.tight_layout()

    stats = ConsensusStructureStats(
        topk=int(topk) if selection == "topk" else None,
        n_topk=int(n_topk),
        group_counts=group_counts,
        group_fracs=group_fracs,
        group_median_score=group_median_score,
        selection=selection,
        threshold=threshold,
    )

    if show:
        plt.show()

    if return_result:
        cells = pd.DataFrame(
            {
                "cell": adata.obs_names.astype(str),
                "obs_position": np.arange(adata.n_obs),
                "x": x,
                "y": y,
                "score": score,
                "group": adata.obs[groupby].to_numpy(),
                "group_label": g,
                "selected": mask,
                "highlighted": displayed,
                "background_displayed": background,
                "score_available": np.isfinite(score),
                "finite_coordinates": finite_xy,
            }
        )
        included = {d["group"] for d in legend_data}
        summary["color"] = summary.group.map(pal)
        summary["legend_included"] = summary.group.isin(included)
        data = {
            "cells": cells,
            "summary": summary,
            "legend": pd.DataFrame(
                legend_data, columns=["group", "fraction", "count", "color"]
            ),
        }
        notes = []
        if (mask & ~finite_xy).any():
            notes.append(
                f"{int((mask & ~finite_xy).sum())} selected cells lack finite coordinates; counts retain them."
            )
        if samples is not None:
            cells["sample"] = adata.obs[sample_key].to_numpy()
            cells["sample_label"] = samples.to_numpy()
            cells["sample_is_missing"] = samples.eq(missing_sample).to_numpy()
            data["samples"] = sample_table
            if missing_sample is not None:
                notes.append(
                    "Missing sample IDs remain separate from identified biological samples."
                )
        return PlotResult(
            fig,
            panels,
            data,
            {
                "plot": "consensus_structure",
                "basis": key,
                "score_key": score_key,
                "groupby": groupby,
                "topk": int(topk) if selection == "topk" else None,
                "n_selected": n_topk,
                "selection": selection,
                "threshold": threshold,
                "selection_scope": "all_finite_input_scores",
                "min_frac_in_legend": min_frac_in_legend,
                "missing_label": missing,
                "palette": pal,
                "palette_origin": palette_origin,
                "color_scale_scope": "selected_scores",
                "norm": {
                    "type": type(norm).__name__,
                    "vmin": norm.vmin,
                    "vmax": norm.vmax,
                },
                "cmap": cmap_object.name,
                "face_alpha": hi_alpha,
                "outline_alpha": outline_alpha,
                "sample_key": sample_key,
                "missing_sample_label": missing_sample,
                "summary_fraction_denominator": "all selected input cells",
                "sample_fraction_denominator": "selected cells in the same sample",
                "prevalence_scope": "all input cells before coordinate or display filtering",
                "sample_inference": "descriptive group-by-sample counts; no test or fitted uncertainty",
                "max_points": max_points,
                "random_state": int(random_state),
                "display_strata": "group_sample" if samples is not None else "group",
                "drawn_cells": int(background.sum()),
                "drawn_selected_cells": int(displayed.sum()),
                "rasterized": raster,
                "rasterized_requested": rasterized,
                "owns_figure": owns,
                "export_scope": "created_figure" if owns else "parent_figure",
            },
            warnings=notes,
            artists=artists,
        )
    return fig, ax, stats, legend_data
