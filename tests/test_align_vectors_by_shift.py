import numpy as np
import pandas as pd
import anndata as ad


def test_align_vectors_by_shift():
    rs = np.random.RandomState(1)
    n, d = 20, 3
    obs = pd.DataFrame(
        {"grp": ["A"] * 10 + ["B"] * 10},
        index=[f"c{i}" for i in range(n)],
    )
    adata = ad.AnnData(X=np.zeros((n, 1)), obs=obs)
    adata.obsm["v"] = rs.normal(size=(n, d)).astype(np.float32)

    # fake by-shift deltas for grp A/B
    adata.uns["scgeo"] = {
        "shift": {
            "by": {
                "A": {"delta": np.array([1, 0, 0], dtype=np.float32)},
                "B": {"delta": np.array([0, 1, 0], dtype=np.float32)},
            }
        }
    }

    import scgeo as sg
    sg.tl.align_vectors(
        adata,
        vec_key="v",
        ref_from_shift=True,
        shift_level="by",
        shift_index_key="grp",
        obs_key="align_by",
    )
    assert "align_by" in adata.obs
