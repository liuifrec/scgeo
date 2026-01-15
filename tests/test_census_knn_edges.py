#tests/test_census_knn_edges.py
import numpy as np
import pytest

from scgeo.pp._census_knn import build_query_to_ref_knn_edges_from_census


class DummyPool:
    """Minimal ReferencePool-like object for testing."""
    def __init__(self, joinid_to_col, n_ref):
        self.joinid_to_col = dict(joinid_to_col)
        self.n_ref = int(n_ref)


def test_build_query_to_ref_knn_edges_from_census_returns_coo_and_diag():
    pool = DummyPool({10: 0, 11: 1, 12: 2, 13: 3, 14: 4}, n_ref=5)

    # n_qry=2, k=3
    nn_joinids = [
        [10, 11, 999],   # 999 missing
        [12, 12, 13],    # duplicate joinid (12) in same query row
    ]
    nn_dists = [
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0],
    ]

    rows, cols, data, diag = build_query_to_ref_knn_edges_from_census(
        n_qry=2,
        ref_pool=pool,
        nn_joinids=nn_joinids,
        nn_dists=nn_dists,
        weight_mode="inv",
        drop_missing=True,
        return_diagnostics=True,
        return_csr=False,
    )

    assert rows.shape == cols.shape == data.shape
    assert diag is not None
    assert diag["n_qry"] == 2
    assert diag["n_ref"] == 5
    assert diag["k"] == 3
    # one missing neighbor dropped
    assert diag["missing"] == 1


def test_build_query_to_ref_knn_edges_from_census_returns_csr_and_sums_duplicates():
    pool = DummyPool({10: 0, 11: 1, 12: 2, 13: 3, 14: 4}, n_ref=5)

    nn_joinids = [
        [10, 11, 999],   # 999 missing
        [12, 12, 13],    # duplicate -> same (row=1, col=2)
    ]
    nn_dists = [
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0],
    ]

    C, diag = build_query_to_ref_knn_edges_from_census(
        n_qry=2,
        ref_pool=pool,
        nn_joinids=nn_joinids,
        nn_dists=nn_dists,
        weight_mode="inv",
        drop_missing=True,
        return_diagnostics=True,
        return_csr=True,
        sum_duplicates=True,
    )

    assert C.shape == (2, 5)
    assert diag is not None
    assert diag["missing"] == 1

    # Check weights: mode="inv" => w=1/(d+eps) approx 1/d for these values
    # row 0: joinid 10 (d=1 -> 1.0), joinid 11 (d=2 -> 0.5), 999 dropped
    row0 = C.getrow(0).toarray().ravel()
    assert row0[0] == pytest.approx(1.0, rel=1e-4, abs=1e-6)
    assert row0[1] == pytest.approx(0.5, rel=1e-4, abs=1e-6)

    # row 1: joinid 12 appears twice with d=1 and d=1 => weight ~1 + 1 = 2
    #        joinid 13 with d=2 => 0.5
    row1 = C.getrow(1).toarray().ravel()
    assert row1[2] == pytest.approx(2.0, rel=1e-4, abs=1e-6)
    assert row1[3] == pytest.approx(0.5, rel=1e-4, abs=1e-6)


def test_build_query_to_ref_knn_edges_from_census_raises_if_missing_and_drop_missing_false():
    pool = DummyPool({10: 0}, n_ref=1)

    nn_joinids = [[999]]
    nn_dists = [[1.0]]

    with pytest.raises(KeyError):
        build_query_to_ref_knn_edges_from_census(
            n_qry=1,
            ref_pool=pool,
            nn_joinids=nn_joinids,
            nn_dists=nn_dists,
            drop_missing=False,
            return_csr=False,
        )
