from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scgeo.pp import ReferencePool, build_reference_pool_from_census


class _DummyIndex:
    def __init__(self, X, metric="euclidean", random_state=0):
        self.X = np.asarray(X, dtype=np.float32)

    def query(self, Xq, k=30):
        Xq = np.asarray(Xq, dtype=np.float32)
        n_q = Xq.shape[0]
        n_ref = self.X.shape[0]
        k_eff = min(k, n_ref)
        idx = np.tile(np.arange(k_eff), (n_q, 1))
        dist = np.zeros((n_q, k_eff), dtype=np.float32)
        return idx, dist


def test_build_reference_pool_from_census_smoke(monkeypatch):
    # small query AnnData
    adata_q = AnnData(X=np.zeros((4, 3), dtype=np.float32))
    adata_q.obsm["X_emb"] = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        dtype=np.float32,
    )

    # patch the imported helper from scgeo.data
    def fake_census_find_nearest_obs(census, *, embedding_name, organism, query_embedding, k=50, **kwargs):
        # emulate DataFrame output
        return pd.DataFrame(
            {
                "soma_joinid": [101, 102, 103, 101, 102, 104],
                "distance": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
            }
        )

    def fake_census_get_embedding(census, *, embedding_name, organism, obs_joinids):
        obs_joinids = np.asarray(obs_joinids)
        # return embedding matrix aligned to joinids
        return np.stack(
            [
                np.arange(len(obs_joinids), dtype=np.float32),
                np.arange(len(obs_joinids), dtype=np.float32) + 1,
                np.arange(len(obs_joinids), dtype=np.float32) + 2,
            ],
            axis=1,
        )

    def fake_fetch_obs_by_joinids(joinids, *, organism="homo_sapiens", census_version=None, obs_columns=None):
        joinids = np.asarray(joinids, dtype=np.int64)
        return pd.DataFrame(
            {
                "soma_joinid": joinids,
                "cell_type": [f"type_{j}" for j in joinids],
                "extra_col": [f"meta_{j}" for j in joinids],
            }
        )

    monkeypatch.setattr("scgeo.pp._reference_pool.census_find_nearest_obs", fake_census_find_nearest_obs, raising=False)
    monkeypatch.setattr("scgeo.pp._reference_pool.census_get_embedding", fake_census_get_embedding, raising=False)
    monkeypatch.setattr("scgeo.pp._reference_pool.fetch_obs_by_joinids", fake_fetch_obs_by_joinids, raising=False)
    monkeypatch.setattr("scgeo.pp._reference_pool._require_pynndescent", lambda: _DummyIndex)

    # because function imports inside body from ..data, patch scgeo.data too
    import scgeo.data as sg_data
    monkeypatch.setattr(sg_data, "census_find_nearest_obs", fake_census_find_nearest_obs)
    monkeypatch.setattr(sg_data, "census_get_embedding", fake_census_get_embedding)
    monkeypatch.setattr(sg_data, "fetch_obs_by_joinids", fake_fetch_obs_by_joinids)

    pool = build_reference_pool_from_census(
        census=object(),
        adata_q=adata_q,
        rep="X_emb",
        embedding_name="scvi",
        label_key="cell_type",
        obs_columns=["extra_col"],
        k=3,
        organism="homo_sapiens",
        max_refs=10,
        dedup=True,
        index_metric="euclidean",
        index_seed=0,
    )

    assert isinstance(pool, ReferencePool)
    assert pool.X.shape[1] == 3
    assert pool.label_key == "cell_type"
    assert pool.joinids is not None
    assert set(pool.joinids.tolist()) == {101, 102, 103, 104}
    assert "cell_type" in pool.obs
    assert "extra_col" in pool.obs
    assert len(pool.obs["cell_type"]) == pool.X.shape[0]
    assert pool.meta["source"] == "cellxgene-census"
    assert pool.meta["embedding_name"] == "scvi"


def test_build_reference_pool_from_census_requires_label_key(monkeypatch):
    adata_q = AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata_q.obsm["X_emb"] = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    def fake_census_find_nearest_obs(census, *, embedding_name, organism, query_embedding, k=50, **kwargs):
        return pd.DataFrame({"soma_joinid": [1, 2], "distance": [0.1, 0.2]})

    def fake_census_get_embedding(census, *, embedding_name, organism, obs_joinids):
        return np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)

    def fake_fetch_obs_by_joinids(joinids, *, organism="homo_sapiens", census_version=None, obs_columns=None):
        # deliberately omit label_key
        return pd.DataFrame({"soma_joinid": np.asarray(joinids, dtype=np.int64)})

    monkeypatch.setattr("scgeo.pp._reference_pool._require_pynndescent", lambda: _DummyIndex)

    import scgeo.data as sg_data
    monkeypatch.setattr(sg_data, "census_find_nearest_obs", fake_census_find_nearest_obs)
    monkeypatch.setattr(sg_data, "census_get_embedding", fake_census_get_embedding)
    monkeypatch.setattr(sg_data, "fetch_obs_by_joinids", fake_fetch_obs_by_joinids)

    with pytest.raises(KeyError, match="label_key"):
        build_reference_pool_from_census(
            census=object(),
            adata_q=adata_q,
            rep="X_emb",
            embedding_name="scvi",
            label_key="cell_type",
            obs_columns=None,
            k=2,
        )