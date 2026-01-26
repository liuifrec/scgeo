# scripts/inspect_gse280305.py
from pathlib import Path
import scanpy as sc

DATA_DIR = Path("data/GSE280305_RAW")
FILES = sorted(DATA_DIR.glob("GSM*_count.loom"))

def tp_from_name(p: Path) -> str:
    # e.g., GSM8594491_D8_count.loom.gz -> D8
    name = p.name
    for tok in name.split("_"):
        if tok.startswith("D") and tok[1:].isdigit():
            return tok  # D8, D11, ...
    raise ValueError(f"Cannot parse timepoint from {name}")

adatas = []
for f in FILES:
    print("\n=== Loading:", f.name)
    # If scanpy can't read .loom.gz directly, gunzip once and point to the .loom
    ad = sc.read_loom(str(f))
    tp = tp_from_name(f)
    ad.obs["timepoint"] = tp
    ad.obs["sample"] = f.stem  # GSM..._D8_count.loom

    print("shape:", ad.shape)
    print("obs keys:", list(ad.obs_keys())[:50], ("..." if len(ad.obs_keys()) > 50 else ""))
    print("var keys:", list(ad.var_keys())[:50], ("..." if len(ad.var_keys()) > 50 else ""))
    print("layers:", list(ad.layers.keys()))
    print("obsm:", list(ad.obsm.keys()))
    print("uns:", list(ad.uns.keys())[:50], ("..." if len(ad.uns_keys()) > 50 else ""))

    # Ensure unique cell ids across samples
    ad.obs_names = [f"{tp}:{x}" for x in ad.obs_names]

    adatas.append(ad)

# Outer join genes is safest; you can later intersect if you want strict comparability
adata_all = sc.concat(adatas, join="outer", label="timepoint", keys=[a.obs["timepoint"][0] for a in adatas])

print("\n=== CONCAT RESULT ===")
print("shape:", adata_all.shape)
print("timepoints:", adata_all.obs["timepoint"].value_counts().to_dict())
print("layers:", list(adata_all.layers.keys()))
