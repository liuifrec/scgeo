# scgeo.get

## `scgeo.get.get_available_tables`

**Signature**  
`(adata, *, store_key: 'str' = 'shift') -> 'list[str]'`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.get.get_composition_table`

**Signature**  
`(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.get.get_ood_summary`

**Signature**  
`(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.get.get_robustness_table`

**Signature**  
`(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.get.get_shift_summary`

**Signature**  
`(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.get.get_velocity_alignment_summary`

**Signature**  
`(adata, *, store_key: 'str' = 'shift') -> 'pd.DataFrame'`

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.get.state_report`

**Signature**  
`(adata, *, node_key=None, robust_shift_key: 'str' = 'robust_shift', representation_key: 'str' = 'representation_stability', local_geometry_key: 'str' = 'local_geometry_stability', local_k=None, pair_aggregation: 'str' = 'median', include_worst_case: 'bool' = True, comparison_label: 'Optional[str]' = None, condition_key: 'Optional[str]' = None, group0: 'Any' = None, group1: 'Any' = None, strict: 'bool' = False) -> 'pd.DataFrame'`

**Docstring**  
Build a canonical state-level ScGeo report from stored analysis results.

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---

## `scgeo.get.table`

**Signature**  
`(adata, store_key: 'str' = 'scgeo', kind: 'str' = 'shift', level: 'str' = 'global') -> 'pd.DataFrame'`

**Docstring**  
Return tidy table from adata.uns[store_key][kind].

### I/O contract

_No I/O entry in `scgeo_io_manifest.json` for this function._

---
