#!/usr/bin/env python3
"""Read-only verification of the active interpreter against an approved checkout.

Uses only the standard library until dependency/import probes. Never modifies
sys.path, installs packages, fetches source, or registers a kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


OPTIONAL = {"scanpy": "scanpy", "scvelo": "scvelo", "ipykernel": "ipykernel",
            "pynndescent": "pynndescent", "statsmodels": "statsmodels"}


def _git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.PIPE, timeout=30
    ).decode("utf-8").strip()


def _source_report(source_root, expected_commit, imported_file):
    root = Path(source_root).resolve()
    result = {"checkout": str(root), "expected_commit": expected_commit,
              "verified": False}
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("expected_commit must be a full, lowercase 40-character Git SHA")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("source_root must be the checkout root")
    result["commit"] = _git(root, "rev-parse", "HEAD")
    result["dirty"] = bool(_git(root, "status", "--porcelain", "--untracked-files=normal"))
    package = Path(imported_file).resolve().parent
    result["installation"] = "checkout" if package == root / "scgeo" else "installed_copy"
    files = _git(root, "ls-tree", "-r", "--name-only", expected_commit, "scgeo").splitlines()
    files = [name for name in files if name.endswith(".py")]
    mismatches = []
    digest = hashlib.sha256()
    for name in files:
        expected = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{expected_commit}:{name}"],
            stderr=subprocess.PIPE, timeout=30)
        path = package / Path(name).relative_to("scgeo")
        actual = path.read_bytes() if path.is_file() else b""
        if actual != expected:
            mismatches.append(name)
        digest.update(name.encode() + b"\0" + hashlib.sha256(actual).digest())
    extras = sorted("scgeo/" + p.relative_to(package).as_posix()
                    for p in package.rglob("*.py")
                    if "scgeo/" + p.relative_to(package).as_posix() not in files)
    result.update(package_files=len(files), package_sha256=digest.hexdigest(),
                  mismatched_files=mismatches, unexpected_files=extras)
    result["verified"] = bool(files and result["commit"] == expected_commit
                              and not result["dirty"] and not mismatches and not extras)
    return result


def environment_report(*, expected_commit, source_root, require_optional=(), pip_check=True):
    """Return JSON-safe environment evidence; a failed check never installs/fixes anything.

    A clean checkout at the exact SHA is required as the source reference. Both
    editable imports and wheels are checked byte-for-byte against its Python
    sources. Package versions alone cannot identify development ScGeo 0.1.0.
    """
    report = {"schema_version": 1, "ok": False, "errors": [], "warnings": [],
              "python": {"version": platform.python_version(), "executable": sys.executable,
                         "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                         "virtualenv": sys.prefix != sys.base_prefix,
                         "conda_environment": (Path(sys.prefix) / "conda-meta").is_dir(),
                         "platform": platform.platform()},
              "packages": sorted(
                  [{"name": d.metadata["Name"], "version": d.version}
                   for d in metadata.distributions() if d.metadata["Name"]],
                  key=lambda d: (d["name"].lower(), d["version"]))}
    if sys.version_info < (3, 9):
        report["errors"].append("Python >=3.9 is required; the deployment guide uses 3.12.")
    report["optional"] = {}
    for name, dist in OPTIONAL.items():
        try:
            available = importlib.util.find_spec(name) is not None
            version = metadata.version(dist) if available else None
        except (ImportError, ValueError, metadata.PackageNotFoundError):
            available, version = False, None
        report["optional"][name] = {"available": available, "version": version,
                                    "import_checked": False}
    for name in require_optional:
        if name not in OPTIONAL:
            report["errors"].append(f"Unknown optional dependency: {name}")
            continue
        try:
            importlib.import_module(name)
            report["optional"][name]["import_checked"] = True
        except Exception as exc:
            report["errors"].append(f"Required optional dependency {name}: {type(exc).__name__}: {exc}")
    try:
        sg = importlib.import_module("scgeo")
        report["scgeo"] = {"file": str(Path(sg.__file__).resolve()),
                           "version": sg.__version__,
                           "distribution_version": metadata.version("scgeo")}
        if report["scgeo"]["version"] != report["scgeo"]["distribution_version"]:
            report["errors"].append("Imported ScGeo version and installed distribution metadata disagree.")
        report["source"] = _source_report(source_root, expected_commit, sg.__file__)
        if not report["source"]["verified"]:
            report["errors"].append("Imported ScGeo does not match a clean checkout at the approved commit.")
        features = {"sample_shift": "sample_key" in inspect.signature(sg.tl.robust_shift).parameters,
                    "representation_stability": callable(sg.tl.representation_stability),
                    "local_geometry_stability": callable(sg.tl.local_geometry_stability),
                    "state_report": callable(sg.get.state_report),
                    "input_contract": callable(sg.get.input_contract),
                    "plot_result": callable(sg.pl.PlotResult)}
        report["features"] = features
        if not all(features.values()):
            report["errors"].append("Required development interfaces are missing.")
        report["warnings"].append(
            "ScGeo's version string may still be 0.1.0; source verification, not that version, identifies this build.")
    except Exception as exc:
        report["errors"].append(f"ScGeo/source verification: {type(exc).__name__}: {exc}")
    try:
        import anndata as ad
        import numpy as np
        from scipy import sparse
        toy = ad.AnnData(sparse.csr_matrix(np.eye(3)))
        toy.obsm["X_check"] = np.zeros((3, 2))
        assert toy[:2].copy().obsm["X_check"].shape == (2, 2)
        report["anndata"] = {"version": metadata.version("anndata"),
                              "sparse_and_obsm_probe": "passed"}
    except Exception as exc:
        report["errors"].append(f"AnnData compatibility: {type(exc).__name__}: {exc}")
    if pip_check:
        try:
            check = subprocess.run([sys.executable, "-m", "pip", "check"],
                                   capture_output=True, text=True, timeout=60)
            report["pip_check"] = {"returncode": check.returncode,
                                   "output": (check.stdout + check.stderr).strip()}
            if check.returncode:
                report["errors"].append("pip check found an incompatible environment.")
        except (OSError, subprocess.SubprocessError) as exc:
            report["errors"].append(f"pip check could not run: {exc}")
    else:
        report["pip_check"] = {"skipped": True}
        report["warnings"].append("Dependency consistency was not checked with pip check.")
    report["ok"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--require-optional", nargs="*", choices=sorted(OPTIONAL), default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = environment_report(expected_commit=args.expected_commit, source_root=args.source_root,
                                require_optional=args.require_optional)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "errors": report["errors"], "report": str(args.output)}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
