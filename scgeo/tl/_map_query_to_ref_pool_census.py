# scgeo/tl/_map_query_to_ref_pool_census.py
from __future__ import annotations

from typing import Optional, Sequence

def map_query_to_ref_pool_census(
    adata_q,
    *,
    pool=None,  # optional prebuilt ReferencePool (preferred for tests / advanced users)
    census=None,
    rep: str = "X_emb",
    embedding_name: Optional[str] = None,
    organism: str = "homo_sapiens",
    label_key: str = "cell_type",
    obs_columns: Optional[Sequence[str]] = None,
    k: int = 50,
    max_refs: int = 200_000,
    dedup: bool = True,
    index_metric: str = "euclidean",
    index_seed: int = 0,
    census_obs_filter: Optional[str] = None,
    # pass-through to map_query_to_ref_pool
    store_key: str = "map_query_to_ref",
    pred_key: str = "scgeo_pred",
    conf_key: str = "scgeo_conf",
    conf_entropy_key: str = "scgeo_conf_entropy",
    conf_margin_key: str = "scgeo_conf_margin",
    ood_key: str = "scgeo_ood",
    reject_key: str = "scgeo_reject",
    conf_method: str = "entropy_margin",
    ood_method: str = "distance",
    reject_conf: Optional[float] = None,
    reject_ood: Optional[float] = None,
    return_probs: bool = False,
    probs_key: str = "X_map_probs",
    label_order_key: str = "map_label_order",
) -> None:
    """
    Phase B canonical spell:
      Census -> ReferencePool -> map_query_to_ref_pool()

    Two modes:
      1) pool provided: purely local, no census needed.
      2) pool None: build from census using pp.build_reference_pool_from_census.
    """
    from ._map_query_to_ref_pool import map_query_to_ref_pool

    if pool is None:
        if census is None:
            raise ValueError("Provide either `pool` or `census`.")
        if embedding_name is None:
            raise ValueError("embedding_name is required when building pool from census.")

        from ..pp import build_reference_pool_from_census

        pool = build_reference_pool_from_census(
            census=census,
            adata_q=adata_q,
            rep=rep,
            embedding_name=embedding_name,
            label_key=label_key,
            obs_columns=obs_columns,
            k=int(k),
            organism=str(organism),
            max_refs=int(max_refs) if max_refs is not None else None,
            dedup=bool(dedup),
            index_metric=str(index_metric),
            index_seed=int(index_seed),
            census_obs_filter=census_obs_filter,
        )

    # Delegate to the proven mapping function
    map_query_to_ref_pool(
        adata_q,
        pool=pool,
        rep=rep,
        store_key=store_key,
        pred_key=pred_key,
        conf_key=conf_key,
        conf_entropy_key=conf_entropy_key,
        conf_margin_key=conf_margin_key,
        ood_key=ood_key,
        reject_key=reject_key,
        conf_method=conf_method,
        ood_method=ood_method,
        reject_conf=reject_conf,
        reject_ood=reject_ood,
        return_probs=return_probs,
        probs_key=probs_key,
        label_order_key=label_order_key,
    )
