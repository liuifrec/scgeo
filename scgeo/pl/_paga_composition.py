from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scgeo as sg


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
    c = _pick_col(df, ("logOR", "beta", "effect", "coef"))
    if c is None:
        raise KeyError("No effect column found. Expected one of: logOR, beta, effect, coef")
    return c


def _get_p_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_col(df, ("q", "padj", "fdr", "p"))


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
):
    """
    Volcano plot: x = signed effect (logOR/beta/effect), y = -log10(q or p).
    Highlights top_k by smallest p/q then largest |effect|.
    """
    df = sg.get.table(adata, store_key=store_key, kind=kind).copy()
    node_col = _get_node_col(df)
    eff_col = effect or _get_effect_col(df)
    p_col = p_col or _get_p_col(df)

    if p_col is None:
        # still plot effect only (no y axis meaning)
        df["_mlog10p"] = 0.0
    else:
        p = pd.to_numeric(df[p_col], errors="coerce").fillna(1.0).clip(1e-300, 1.0)
        df["_mlog10p"] = -np.log10(p)

    x = pd.to_numeric(df[eff_col], errors="coerce").fillna(0.0).to_numpy()
    y = df["_mlog10p"].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(x, y, s=14, alpha=0.8)
    ax.axvline(0, linewidth=1, alpha=0.6)
    ax.set_xlabel(eff_col)
    ax.set_ylabel(f"-log10({p_col})" if p_col else "-log10(p/q) (missing)")
    ax.set_title(title)

    # pick top_k: smallest p/q then largest |effect|
    if top_k and top_k > 0:
        if p_col is None:
            rank_df = df.assign(_abs=np.abs(x)).sort_values("_abs", ascending=False)
        else:
            rank_df = df.assign(_p=pd.to_numeric(df[p_col], errors="coerce").fillna(1.0), _abs=np.abs(x))
            rank_df = rank_df.sort_values(["_p", "_abs"], ascending=[True, False])

        top = rank_df.head(top_k)
        xt = pd.to_numeric(top[eff_col], errors="coerce").fillna(0.0).to_numpy()
        yt = top["_mlog10p"].to_numpy()

        ax.scatter(xt, yt, s=28, alpha=0.95)

        if label:
            for _, r in top.iterrows():
                ax.text(
                    float(r[eff_col]) if pd.notna(r[eff_col]) else 0.0,
                    float(r["_mlog10p"]),
                    str(r[node_col]),
                    fontsize=8,
                    alpha=0.9,
                )

    if show:
        plt.show()
    return fig, np.array([ax], dtype=object)


def paga_composition_bar(
    adata,
    *,
    store_key: str = "scgeo",
    kind: str = "paga_composition_stats",
    effect: Optional[str] = None,
    p_col: Optional[str] = None,
    top_k: int = 15,
    sort_by: Optional[str] = None,  # "p" | "abs_effect" | None (auto)
    ax=None,
    figsize=(6.2, 4.2),
    title: str = "ScGeo: Top composition shifts",
    show: bool = True,
):
    """
    Bar plot of top_k nodes by evidence + effect.
    """
    df = sg.get.table(adata, store_key=store_key, kind=kind).copy()
    node_col = _get_node_col(df)
    eff_col = effect or _get_effect_col(df)
    p_col = p_col or _get_p_col(df)

    df["_eff"] = pd.to_numeric(df[eff_col], errors="coerce").fillna(0.0)
    df["_abs"] = df["_eff"].abs()

    if p_col is not None:
        df["_p"] = pd.to_numeric(df[p_col], errors="coerce").fillna(1.0)
    else:
        df["_p"] = 1.0

    if sort_by is None:
        # default: p then abs effect
        sort_by = "p"

    if sort_by == "p":
        df = df.sort_values(["_p", "_abs"], ascending=[True, False])
    elif sort_by == "abs_effect":
        df = df.sort_values(["_abs", "_p"], ascending=[False, True])
    else:
        raise ValueError("sort_by must be one of: None, 'p', 'abs_effect'")

    top = df.head(top_k).copy()
    top = top.iloc[::-1]  # reverse so top is at top

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.barh(top[node_col].astype(str), top["_eff"].to_numpy())
    ax.axvline(0, linewidth=1, alpha=0.6)
    ax.set_xlabel(eff_col)
    ax.set_title(title)

    # annotate p/q
    if p_col is not None:
        for i, (_, r) in enumerate(top.iterrows()):
            ax.text(
                float(r["_eff"]),
                i,
                f"  {p_col}={r['_p']:.2g}",
                va="center",
                fontsize=8,
                alpha=0.85,
            )

    if show:
        plt.show()
    return fig, np.array([ax], dtype=object)


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
):
    """
    1×2 panel: volcano + top-k bars.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    paga_composition_volcano(
        adata,
        store_key=store_key,
        kind=kind,
        effect=effect,
        p_col=p_col,
        top_k=top_k,
        ax=axes[0],
        show=False,
    )
    paga_composition_bar(
        adata,
        store_key=store_key,
        kind=kind,
        effect=effect,
        p_col=p_col,
        top_k=top_k,
        ax=axes[1],
        show=False,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
