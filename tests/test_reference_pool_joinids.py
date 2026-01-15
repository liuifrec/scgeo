import numpy as np
import scgeo as sg

def test_reference_pool_joinids_mapping():
    X = np.random.randn(5, 3).astype(np.float32)
    joinids = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    obs = {"label": np.array(["a","a","b","b","b"], dtype=object)}
    pool = sg.pp.build_reference_pool(X, obs, label_key="label", joinids=joinids)

    assert pool.n_ref == 5
    m = pool.joinid_to_col
    assert m[10] == 0
    assert m[14] == 4