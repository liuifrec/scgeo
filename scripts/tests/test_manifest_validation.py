"""Regression checks for drift that the old manifest validator accepted."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scgeo_manifest_validation", ROOT / "scripts/validate_manifests.py"
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
FILES = ("api_manifest.json", "scgeo_io_raw.json", "scgeo_io_manifest.json")


@pytest.fixture
def manifests(tmp_path):
    for name in FILES:
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def _rewrite(path, edit):
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def test_current_manifests_match_runtime(manifests):
    VALIDATOR.validate(manifests)


@pytest.mark.parametrize("filename", FILES)
def test_development_version_is_rejected(manifests, filename):
    _rewrite(manifests / filename, lambda data: data.update(scgeo_version="0.1.0-dev"))
    with pytest.raises(AssertionError, match="Version mismatch"):
        VALIDATOR.validate(manifests)


@pytest.mark.parametrize("filename", FILES)
def test_stale_census_signature_is_rejected(manifests, filename):
    name = "scgeo.tl.map_query_to_ref_pool_census"

    def omit_current_parameters(data):
        if filename == "api_manifest.json":
            entry = next(row for row in data["tl"] if row["name"] == name)
        else:
            entry = {**data["tl"], **data["skipped"]}[name]
        entry["signature"] = "(adata_q, *, embedding_name=None, k=50, dedup=True)"

    _rewrite(manifests / filename, omit_current_parameters)
    with pytest.raises(AssertionError, match="Signature mismatch"):
        VALIDATOR.validate(manifests)


@pytest.mark.parametrize("namespace", ["pp", "data", "get"])
def test_omitted_public_exports_are_rejected(manifests, namespace):
    _rewrite(manifests / "api_manifest.json", lambda data: data[namespace].pop())
    with pytest.raises(AssertionError, match="Export mismatch"):
        VALIDATOR.validate(manifests)
