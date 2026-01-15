import numpy as np
import scgeo as sg


def test_reference_pool_search_shapes():
    rng = np.random.RandomState(0)
    X_ref = rng.normal(size=(50, 8)).astype(np.float32)
    labels = np.array(["A"] * 25 + ["B"] * 25, dtype=object)

    pool = sg.pp.build_reference_pool(X_ref, {"label": labels}, label_key="label", n_neighbors=15)
    Xq = rng.normal(size=(7, 8)).astype(np.float32)

    idx, dist = pool.search(Xq, k=10)

    assert idx.shape == (7, 10)
    assert dist.shape == (7, 10)
    assert np.all(idx >= 0)
    assert np.all(idx < 50)
