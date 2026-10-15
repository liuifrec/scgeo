"""Check source/wheel/sdist consistency without importing ScGeo.

Run with Python >=3.11, PyYAML, and packaging after `python -m build`.
The manifest is an audit artifact, not a file to upload to a package index.
"""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import yaml


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_version(root: Path) -> tuple[dict, str]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    require(bool(re.fullmatch(r"\d+\.\d+\.\d+", version)), "Expected an X.Y.Z release version")
    tree = ast.parse((root / "scgeo/__init__.py").read_text(encoding="utf-8"))
    versions = [ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    recipe = yaml.safe_load((root / "conda-recipe/meta.yaml").read_text(encoding="utf-8"))
    require(versions == [version], "Package and pyproject versions differ")
    require(str(recipe["package"]["version"]) == version, "Conda version differs")
    require(project["name"] == recipe["package"]["name"] == "scgeo", "Unexpected package name")
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/"):
        require(ref == f"refs/tags/v{version}", "Release tag must exactly match the package version")
    return project, version


def check(root: Path, dist: Path, output: Path) -> dict:
    project, version = source_version(root)
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    require(len(wheels) == len(sdists) == 1, "Expected exactly one wheel and one sdist")
    require(len(list(dist.iterdir())) == 2, "dist/ must contain only the two distributions")
    wheel, sdist = wheels[0], sdists[0]
    require(wheel.name == f"scgeo-{version}-py3-none-any.whl", "Unexpected wheel filename")
    require(sdist.name == f"scgeo-{version}.tar.gz", "Unexpected sdist filename")
    sources = {p.relative_to(root).as_posix(): p.read_bytes() for p in (root / "scgeo").rglob("*.py")}
    require(bool(sources), "No package source files found")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate wheel entries")
        py_names = {n for n in names if n.startswith("scgeo/") and n.endswith(".py")}
        require(py_names == set(sources), "Wheel package sources differ from checkout")
        for path, data in sources.items():
            require(archive.read(path) == data, f"Wheel source mismatch: {path}")
        metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metadata_paths) == 1, "Expected one wheel metadata record")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        require(metadata["Name"] == "scgeo" and metadata["Version"] == version, "Incorrect wheel identity")
        require(metadata["Requires-Python"] == project["requires-python"], "Incorrect Python requirement")
        license_paths = [n for n in names if n.endswith("/licenses/LICENSE")]
        require(len(license_paths) == 1, "Wheel license is missing")
        require(archive.read(license_paths[0]) == (root / "LICENSE").read_bytes(), "Wheel license differs")
        requirements = metadata.get_all("Requires-Dist", [])
        core = [str(Requirement(r)) for r in requirements if Requirement(r).marker is None]
        require(set(core) == {str(Requirement(r)) for r in project["dependencies"]}, "Core dependencies differ")
        extras = {canonicalize_name(e) for e in metadata.get_all("Provides-Extra", [])}
        require(extras == {canonicalize_name(e) for e in project["optional-dependencies"]}, "Extras differ")
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), "Duplicate sdist entries")
        prefix = f"scgeo-{version}/"
        for member in members:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts, "Unsafe archive path")
            require(member.isdir() or member.isfile(), "Unexpected archive link or device")
            require(not any(part in {".git", ".cache", "results", "__pycache__"} for part in path.parts),
                    f"Generated/private artifact in sdist: {member.name}")
        required = ["LICENSE", "README.md", "CITATION.cff", "CHANGELOG.md", "MANIFEST.in",
                    "pyproject.toml", "pytest.ini", "scripts/smoke_installation.py",
                    "scripts/check_release.py", "scripts/verify_testpypi.py", "conda-recipe/meta.yaml"]
        for path in [*required, *sources]:
            handle = archive.extractfile(prefix + path)
            require(handle is not None, f"Required source file missing: {path}")
            require(handle.read() == (root / path).read_bytes(), f"Sdist source mismatch: {path}")
        require(any(n.startswith(prefix + "tests/") for n in names), "Source tests missing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "Invalid source commit")
    manifest = {"name": "scgeo", "version": version, "source_commit": commit,
                "git_ref": os.environ.get("GITHUB_REF", ""), "requires_dist": requirements,
                "files": {p.name: sha256(p.read_bytes()) for p in (wheel, sdist)}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "core-requirements.txt").write_text("\n".join(core) + "\n", encoding="utf-8")
    for path in ["scripts/smoke_installation.py", "scripts/verify_testpypi.py", "conda-recipe/meta.yaml"]:
        shutil.copyfile(root / path, output / Path(path).name)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path, default=Path("release-audit"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = check(root, args.dist.resolve(), args.output.resolve())
    print(f"PASS: source, wheel, sdist, dependencies, license and version {result['version']}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.write(f"version={result['version']}\n")
