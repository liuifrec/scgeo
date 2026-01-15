import numpy as np
import scgeo as sg

class DummyPool:
    def __init__(self):
        self.ref_joinids = np.array(["a", "b", "c"], dtype=object)
        self.joinid_to_col = {"a": 0, "b": 1, "c": 2}
        self.n_ref = 3

def test_build_query_to_ref_knn_edges_from_census_basic():
    pool = DummyPool()
    nn_joinids = [["a", "b"], ["c", "x"]]   # x missing
    nn_dists = [[0.1, 0.2], [0.3, 0.4]]

    rows, cols, data, diag = sg.pp.build_query_to_ref_knn_edges_from_census(
        n_qry=2,
        ref_pool=pool,
        nn_joinids=nn_joinids,
        nn_dists=nn_dists,
        weight_mode="inv",
        drop_missing=True,
        return_diagnostics=True,
    )

    assert rows.tolist() == [0, 0, 1]      # (0->a), (0->b), (1->c)
    assert cols.tolist() == [0, 1, 2]
    assert len(data) == 3
    assert diag["missing"] == 1
