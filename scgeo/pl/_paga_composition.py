from __future__ import annotations

from typing import Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scgeo as sg

from ._result import PlotResult
from ._table_plot_utils import numeric_column, panel_axes
from ._utils import _get_ax


def _pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_node_col(df: pd.DataFrame) -> str:
    c = _pick_col(df, ("node", "cluster", "name"))
    if c is not None:
        return c
    # fallback: make an index-derived node label
    df["_node"] = df.index.astype(str)
    return "_node"


def _get_effect_col(df: pd.DataFrame) -> str:
    c = _pick_col(df, ("logOR", "beta", "effect", "coef", "OR"))
    if c is None:
        raise KeyError("No effect column found. Expected one of: logOR, beta, effect, coef, OR")
    return c


def _get_p_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(df, ("q", "padj", "fdr", "p"))


def _composition_source(adata, store_key, kind, effect, p_col):
    source = sg.get.table(adata, store_key=store_key, kind=kind).copy(deep=True)
    work = source.copy(deep=True)
    node_col = _get_node_col(work)
    eff_col = effect or _get_effect_col(work)
    p_col = p_col or _get_p_col(work)
    values = numeric_column(work, eff_col)
    probabilities = (numeric_column(work, p_col, probability=True) if p_col is not None
                     else np.full(len(work), np.nan))
    if eff_col == "OR" and (values[np.isfinite(values)] <= 0).any():
        raise ValueError("OR values must be positive or missing; plot a log effect explicitly if stored.")
    null_effect = 1.0 if eff_col == "OR" else 0.0
    nodes = pd.DataFrame({
        "source_row": np.arange(len(source)), "node": work[node_col].astype(str).to_numpy(),
        "effect": values, "p_value": probabilities,
        "neglog10_p": -np.log10(np.maximum(probabilities, 1e-300)),
        "p_clipped_for_display": np.isfinite(probabilities) & (probabilities < 1e-300),
        "effect_available": np.isfinite(values), "p_available": np.isfinite(probabilities),
        "distance_from_null": np.abs(values - null_effect),
    })
    payload = adata.uns[store_key][kind]
    params = deepcopy(payload.get("params", {})) if isinstance(payload, dict) else {}
    metadata = {"store_key": store_key, "kind": kind, "effect_col": eff_col,
                "p_col": p_col, "null_effect": null_effect, "p_display_floor": 1e-300,
                "analysis_params": params, "analysis_recomputed": False}
    return source, nodes, metadata


def _top_k(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("top_k must be a nonnegative integer.")
    return int(value)


def _rank(nodes, sort_by):
    if sort_by == "p":
        return nodes.sort_values(["p_value", "distance_from_null", "source_row"],
                                 ascending=[True, False, True], na_position="last")
    if sort_by == "abs_effect":
        return nodes.sort_values(["distance_from_null", "p_value", "source_row"],
                                 ascending=[False, True, True], na_position="last")
    raise ValueError("sort_by must be one of: None, 'p', 'abs_effect'")


def paga_composition_volcano(
    adata,
    *,
    store_key: str = "scgeo",
    kind: str = "paga_composition_stats",
    effect: Optional[str] = None,
    p_col: Optional[str] = None,
    top_k: int = 10,
    label: bool = True,
    ax=None,
    figsize=(5.2, 4.2),
    title: str = "ScGeo: PAGA composition volcano",
    show: bool = True,
    return_data: bool = False,
    return_result: bool = False,
):
    """Plot stored composition effects against their supplied p/q values.

    Missing effects or p-values are omitted and counted, never filled with zero
    or one. If no p/q column exists, show an explicitly labelled effect-only
    strip whose constant ordinate has no inferential meaning. Zero p-values
    use a display floor of 1e-300; original values remain in exported sources.
    Probabilities outside [0, 1] and infinite effects raise before drawing.

    Auto effect selection accepts logOR, beta, effect, coef, then OR. Stored
    odds ratios remain on their original scale with a null reference at one;
    other effects use zero. No p-values, FDR or confidence intervals are fitted.
    ``return_result=True`` provides original source rows, transformed points,
    selection flags and stored analysis parameters. ``return_data=True`` adds
    these tables to the legacy ``(fig, axes_array)`` return.
    """
    top_k = _top_k(top_k)
    source, nodes, metadata = _composition_source(adata, store_key, kind, effect, p_col)
    effect_only = metadata["p_col"] is None
    nodes["displayed"] = nodes.effect_available & (effect_only | nodes.p_available)
    nodes["plot_y"] = 0.0 if effect_only else nodes.neglog10_p
    nodes.loc[~nodes.displayed, "plot_y"] = np.nan
    selected = _rank(nodes.loc[nodes.displayed], "abs_effect" if effect_only else "p").head(top_k)
    nodes["highlighted"] = nodes.source_row.isin(selected.source_row)
    nodes["labelled"] = nodes.highlighted & label
    fig, ax = _get_ax(ax=ax, figsize=figsize)
    shown = nodes.loc[nodes.displayed]
    artists = {"points": ax.scatter(shown.effect, shown.plot_y, s=18, alpha=.8, color="#527F9A")}
    artists["null"] = ax.axvline(metadata["null_effect"], linewidth=1, alpha=.5, color="#677783")
    if len(selected):
        artists["highlights"] = ax.scatter(selected.effect, selected.plot_y, s=32,
                                           color="#C77D4D", alpha=.95)
        if label:
            artists["labels"] = [ax.annotate(row.node, (row.effect, row.plot_y),
                                             xytext=(4, 3), textcoords="offset points", fontsize=8)
                                 for row in selected.itertuples()]
    ax.set(xlabel=metadata["effect_col"],
           ylabel="Effect only (no p/q supplied)" if effect_only else f"-log10({metadata['p_col']})",
           title=title)
    if effect_only:
        ax.set_yticks([])
        ax.set_ylim(-.5, .5)
    notes = []
    omitted = int((~nodes.displayed).sum())
    if omitted:
        note = f"{omitted} rows lack a finite effect" + ("/p pair" if not effect_only else "") + "; omitted."
        ax.text(.02, .98, note, transform=ax.transAxes, va="top", fontsize=8, color="#687680")
        notes.append(note)
    if effect_only:
        notes.append("No p/q column supplied; constant strip height is not an inferred probability.")
    if nodes.p_clipped_for_display.any():
        notes.append("p/q values below 1e-300 use that floor for the log display only.")
    result = PlotResult(fig, {"volcano": ax}, {"source": source, "nodes": nodes},
                        {**metadata, "plot": "paga_composition_volcano", "top_k": top_k,
                         "ordinate_mode": "effect_only" if effect_only else "neglog10_p"},
                        warnings=notes, artists=artists)
    if show:
        plt.show()
    if return_result:
        return result
    legacy = (fig, np.array([ax], dtype=object))
    return (*legacy, result.data) if return_data else legacy


def paga_composition_bar(
    adata,
    *,
    store_key: str = "scgeo",
    kind: str = "paga_composition_stats",
    effect: Optional[str] = None,
    p_col: Optional[str] = None,
    top_k: int = 15,
    sort_by: Optional[str] = None,
    ax=None,
    figsize=(6.2, 4.2),
    title: str = "ScGeo: Top composition shifts",
    show: bool = True,
    return_data: bool = False,
    return_result: bool = False,
):
    """Display stored composition effects with explicit unavailable entries.

    Rank by p/q then distance from the null, or ``sort_by='abs_effect'``.
    Missing effects occupy labelled rows without a zero-length bar; missing
    p-values are labelled unavailable. Original-scale OR effects use one as
    their null and bar baseline. No statistics or intervals are recomputed.
    ``return_result=True`` exports the untouched source, full node table with
    selection flags, displayed row order and stored analysis parameters.
    Legacy returns remain ``(fig, axes_array)``; ``return_data=True`` adds tables.
    """
    top_k = _top_k(top_k)
    source, nodes, metadata = _composition_source(adata, store_key, kind, effect, p_col)
    sort_by = sort_by or ("p" if metadata["p_col"] is not None else "abs_effect")
    top = _rank(nodes, sort_by).head(top_k).iloc[::-1].copy()
    nodes["selected"] = nodes.source_row.isin(top.source_row)
    nodes["displayed"] = nodes.selected & nodes.effect_available
    top["plot_y"] = np.arange(len(top))
    fig, ax = _get_ax(ax=ax, figsize=figsize)
    available = top.effect_available
    baseline = metadata["null_effect"]
    artists = {"bars": ax.barh(top.loc[available, "plot_y"], top.loc[available, "effect"] - baseline,
                                left=baseline, color="#527F9A", alpha=.9)}
    artists["null"] = ax.axvline(baseline, linewidth=1, alpha=.5, color="#677783")
    artists["annotations"] = []
    for row in top.itertuples():
        if not row.effect_available:
            text = "effect unavailable"
            if metadata["p_col"] is not None:
                text += f"; {metadata['p_col']}={row.p_value:.2g}" if row.p_available else "; p/q unavailable"
            artists["annotations"].append(ax.text(.02, row.plot_y, text,
                transform=ax.get_yaxis_transform(), va="center", fontsize=8, color="#8A5C55"))
        elif metadata["p_col"] is not None:
            text = f"{metadata['p_col']}={row.p_value:.2g}" if row.p_available else f"{metadata['p_col']} unavailable"
            artists["annotations"].append(ax.annotate(text, (row.effect, row.plot_y),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8, annotation_clip=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .5}))
    ax.set_yticks(top.plot_y, top.node)
    ax.set(xlabel=metadata["effect_col"], title=title)
    ax.margins(x=.25)
    notes = []
    if not top.effect_available.all():
        notes.append("Missing effects are labelled unavailable; no zero effect is substituted.")
    if not top.p_available.all():
        notes.append("Unavailable p/q values remain missing; no probability is imputed.")
    result = PlotResult(fig, {"bar": ax}, {"source": source, "nodes": nodes, "rows": top},
                        {**metadata, "plot": "paga_composition_bar", "top_k": top_k, "sort_by": sort_by},
                        warnings=notes, artists=artists)
    if show:
        plt.show()
    if return_result:
        return result
    legacy = (fig, np.array([ax], dtype=object))
    return (*legacy, result.data) if return_data else legacy


def paga_composition_panel(
    adata,
    *,
    store_key: str = "scgeo",
    kind: str = "paga_composition_stats",
    effect: Optional[str] = None,
    p_col: Optional[str] = None,
    top_k: int = 10,
    figsize=(12, 4.2),
    show: bool = True,
    axes=None,
    return_data: bool = False,
    return_result: bool = False,
):
    """Compose the existing composition volcano and effect bars.

    Supply named ``volcano`` and ``bar`` axes from one Figure or SubFigure.
    Caller-owned size/layout are preserved. The default return remains
    ``(fig, axes_array)``; opt into source tables with ``return_data=True`` or
    an exportable PlotResult with ``return_result=True``. Missing evidence is
    retained and labelled by each child renderer, never refitted or filled.
    """
    _top_k(top_k)
    _composition_source(adata, store_key, kind, effect, p_col)  # validate before drawing
    fig, mapped, owns_figure = panel_axes(["volcano", "bar"], axes, figsize)
    results = {}
    for name, function in (("volcano", paga_composition_volcano), ("bar", paga_composition_bar)):
        results[name] = function(adata, store_key=store_key, kind=kind, effect=effect,
                                 p_col=p_col, top_k=top_k, ax=mapped[name],
                                 return_result=True, show=False)
    if owns_figure:
        fig.tight_layout()
    result = PlotResult(fig, mapped, data={name: value.data for name, value in results.items()},
                        metadata={"plot": "paga_composition_panel", "analysis_recomputed": False,
                                  "panels": {name: value.metadata for name, value in results.items()},
                                  "export_scope": "created_figure" if owns_figure else "parent_figure"},
                        warnings=[note for value in results.values() for note in value.warnings],
                        artists={name: value.artists for name, value in results.items()})
    if show:
        plt.show()
    if return_result:
        return result
    legacy = (fig, np.array([mapped["volcano"], mapped["bar"]], dtype=object))
    return (*legacy, result.data) if return_data else legacy
