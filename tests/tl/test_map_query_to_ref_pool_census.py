from __future__ import annotations

import numpy as np
from anndata import AnnData


def test_map_query_to_ref_pool_census_builds_pool_and_delegates(monkeypatch):
    from scgeo.tl._map_query_to_ref_pool_census import map_query_to_ref_pool_census

    adata_q = AnnData(X=np.zeros((3, 2), dtype=np.float32))
    adata_q.obsm["X_emb"] = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        dtype=np.float32,
    )

    fake_pool = object()
    seen = {}

    def fake_build_reference_pool_from_census(**kwargs):
        seen["build_kwargs"] = kwargs
        return fake_pool

    def fake_map_query_to_ref_pool(
        adata_q_in,
        *,
        pool,
        rep,
        store_key,
        pred_key,
        conf_key,
        conf_entropy_key,
        conf_margin_key,
        ood_key,
        reject_key,
        conf_method,
        ood_method,
        reject_conf,
        reject_ood,
        return_probs,
        probs_key,
        label_order_key,
    ):
        seen["delegate_called"] = True
        seen["pool"] = pool
        seen["rep"] = rep
        seen["pred_key"] = pred_key

    monkeypatch.setattr(
        "scgeo.pp.build_reference_pool_from_census",
        fake_build_reference_pool_from_census,
    )
    monkeypatch.setattr(
        "scgeo.tl._map_query_to_ref_pool.map_query_to_ref_pool",
        fake_map_query_to_ref_pool,
    )

    map_query_to_ref_pool_census(
        adata_q,
        census=object(),
        rep="X_emb",
        embedding_name="scvi",
        label_key="cell_type",
        obs_columns=["cell_type"],
        k=20,
    )

    assert seen["delegate_called"] is True
    assert seen["pool"] is fake_pool
    assert seen["rep"] == "X_emb"
    assert seen["pred_key"] == "scgeo_pred"
    assert seen["build_kwargs"]["embedding_name"] == "scvi"
    assert seen["build_kwargs"]["label_key"] == "cell_type"


def test_map_query_to_ref_pool_census_accepts_prebuilt_pool(monkeypatch):
    from scgeo.tl._map_query_to_ref_pool_census import map_query_to_ref_pool_census

    adata_q = AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata_q.obsm["X_emb"] = np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)

    fake_pool = object()
    seen = {}

    def fake_map_query_to_ref_pool(*args, **kwargs):
        seen["called"] = True
        seen["pool"] = kwargs["pool"]

    monkeypatch.setattr(
        "scgeo.tl._map_query_to_ref_pool.map_query_to_ref_pool",
        fake_map_query_to_ref_pool,
    )

    map_query_to_ref_pool_census(
        adata_q,
        pool=fake_pool,
        rep="X_emb",
    )

    assert seen["called"] is True
    assert seen["pool"] is fake_pool