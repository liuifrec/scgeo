from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import scanpy as sc
from anndata import AnnData


def _make_de_adata(
    adata,
    *,
    mask,
    obs_cols: list[str],
    layer: Optional[str] = None,
) -> AnnData:
    """
    Build a minimal AnnData for DE to avoid copying heavy obsm/uns/layers/obsp.
    """
    if layer is None:
        X = adata.X[mask]
    else:
        if layer not in adata.layers:
            raise KeyError(f"{layer!r} not found in adata.layers")
        X = adata.layers[layer][mask]

    obs = adata.obs.loc[mask, obs_cols].copy()
    var = adata.var.copy()

    return AnnData(X=X, obs=obs, var=var)


def alignment_driver_genes(
    adata,
    *,
    alignment_key: str,
    group1: str = "discordant",
    group2: str = "aligned",
    subset_key: Optional[str] = None,
    subset_values: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    method: str = "wilcoxon",
    pts: bool = True,
    min_cells: int = 20,
    key_added: str = "alignment_driver_genes",
) -> pd.DataFrame:
    """
    Identify genes associated with alignment or discordance between geometric shift
    and velocity direction.

    Parameters
    ----------
    adata
        AnnData object.
    alignment_key
        obs column containing alignment labels, e.g. "alignment_group".
    group1, group2
        The two groups to compare, usually "discordant" vs "aligned".
    subset_key
        Optional obs column for stratified analysis, e.g. "cluster_label_manual".
    subset_values
        Optional subset values to analyze. If None and subset_key is provided,
        all categories/unique values are used.
    layer
        Optional layer to use as expression matrix. If provided, it is copied into
        a minimal temporary AnnData as X before DE.
    method
        Differential expression method passed to scanpy.tl.rank_genes_groups.
    pts
        Whether to compute fraction of expressing cells.
    min_cells
        Minimum number of cells required in each group for a valid comparison.
    key_added
        Key used to store the concatenated result table in adata.uns.

    Returns
    -------
    pd.DataFrame
        Tidy dataframe of DE results with columns such as:
        ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', ...]
        plus ['subset', 'group1', 'group2'].
    """
    if alignment_key not in adata.obs:
        raise KeyError(f"{alignment_key!r} not found in adata.obs")

    results: list[pd.DataFrame] = []

    def _run_one(mask, subset_name: str) -> Optional[pd.DataFrame]:
        counts = adata.obs.loc[mask, alignment_key].astype(str).value_counts()
        if counts.get(group1, 0) < min_cells or counts.get(group2, 0) < min_cells:
            return None

        obs_cols = [alignment_key]
        if subset_key is not None:
            obs_cols.append(subset_key)

        sub = _make_de_adata(
            adata,
            mask=mask,
            obs_cols=obs_cols,
            layer=layer,
        )

        # Important: do NOT pass layer=... here, because sub.X is already the chosen matrix
        sc.tl.rank_genes_groups(
            sub,
            groupby=alignment_key,
            groups=[group1],
            reference=group2,
            method=method,
            pts=pts,
        )

        df = sc.get.rank_genes_groups_df(sub, group=group1)
        df["subset"] = subset_name
        df["group1"] = group1
        df["group2"] = group2
        return df

    if subset_key is None:
        mask = adata.obs[alignment_key].astype(str).isin([group1, group2]).to_numpy()
        df = _run_one(mask, "all")
        if df is None:
            raise ValueError(
                f"Not enough cells for comparison {group1!r} vs {group2!r}."
            )
        results.append(df)

    else:
        if subset_key not in adata.obs:
            raise KeyError(f"{subset_key!r} not found in adata.obs")

        if subset_values is None:
            vals = pd.unique(adata.obs[subset_key].astype(str))
            subset_values = [str(v) for v in vals]

        subset_series = adata.obs[subset_key].astype(str)
        align_series = adata.obs[alignment_key].astype(str)

        for val in subset_values:
            mask = (
                (subset_series == str(val))
                & (align_series.isin([group1, group2]))
            ).to_numpy()

            if mask.sum() == 0:
                continue

            df = _run_one(mask, str(val))
            if df is not None:
                results.append(df)

        if len(results) == 0:
            raise ValueError("No valid subset comparisons produced results.")

    out = pd.concat(results, ignore_index=True)
    adata.uns[key_added] = out
    return out