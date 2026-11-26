"""Approval-gated promotion of a new ScGeo release from verified TestPyPI bytes.

The original 0.1.0 helper and provenance remain unchanged. This module never
uploads, executes package source, or rebuilds distributions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError, URLError

from pypi_promotion import (
    FILE_HOSTS, INDEX_HOSTS, JOBS, MAX_FILE_SIZE, REPOSITORY, REPOSITORY_ID,
    github, read_url, require, validate_url,
)


def check_version(version: str) -> str:
    require(bool(re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version)),
            "Expected an exact X.Y.Z version")
    require(tuple(map(int, version.split("."))) >= (0, 2, 0),
            "This workflow is for 0.2.0 and later; retain the original 0.1.0 record")
    return version


def names(version: str) -> set[str]:
    check_version(version)
    return {f"scgeo-{version}-py3-none-any.whl", f"scgeo-{version}.tar.gz"}


def validate_source(run: dict, jobs: dict, tag_commit: str, version: str, run_id: int) -> None:
    check_version(version)
    require(run.get("id") == run_id and run.get("run_attempt") == 1,
            "Expected the original staging run; do not reuse a rebuilt attempt")
    require(run.get("repository", {}).get("id") == REPOSITORY_ID
            and run.get("repository", {}).get("full_name") == REPOSITORY
            and run.get("head_repository", {}).get("id") == REPOSITORY_ID,
            "Unexpected staging repository")
    require(run.get("event") == "push" and run.get("path") == ".github/workflows/publish.yml",
            "Expected the release tag-push workflow, not a PR or branch run")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", tag_commit))
            and run.get("head_branch") == f"v{version}"
            and run.get("head_sha") == tag_commit, "Tag and staging commit differ")
    require(run.get("status") == "completed" and run.get("conclusion") == "success",
            "Staging did not complete successfully")
    entries = jobs.get("jobs", [])
    require(jobs.get("total_count") == len(entries) == len(JOBS)
            and {j.get("name") for j in entries} == JOBS,
            "Incomplete or unexpected staging jobs")
    require(all(j.get("status") == "completed" and j.get("conclusion") == "success"
                and j.get("run_id") == run_id for j in entries),
            "A required staging job did not pass")


def select_artifacts(records: dict, run_id: int, commit: str) -> dict:
    entries = records.get("artifacts", [])
    require(records.get("total_count") == len(entries), "Incomplete artifact list")
    chosen = {}
    for name in ("scgeo-distributions", "scgeo-release-audit"):
        matches = [entry for entry in entries if entry.get("name") == name]
        require(len(matches) == 1, f"Missing or ambiguous artifact: {name}")
        entry = matches[0]
        origin = entry.get("workflow_run", {})
        require(type(entry.get("id")) is int and entry["id"] > 0
                and entry.get("expired") is False
                and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", entry.get("digest") or ""))
                and origin.get("id") == run_id and origin.get("head_sha") == commit
                and origin.get("repository_id") == REPOSITORY_ID,
                f"Expired, changed or wrong-source artifact: {name}")
        chosen[name] = {"id": entry["id"], "digest": entry["digest"]}
    return chosen


def validate_manifest(manifest: dict, version: str, commit: str) -> dict:
    require(manifest.get("name") == "scgeo" and manifest.get("version") == version,
            "Wrong manifest package/version")
    require(manifest.get("source_commit") == commit
            and manifest.get("git_ref") == f"refs/tags/v{version}",
            "Manifest source differs from the approved tag")
    hashes = manifest.get("files", {})
    require(set(hashes) == names(version)
            and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                    for v in hashes.values()), "Invalid manifest filenames or hashes")
    return hashes


def check_directory(directory: Path, hashes: dict) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "Invalid distribution directory")
    require({p.name for p in directory.iterdir()} == set(hashes),
            "Unexpected or missing distribution files")
    for name, digest in hashes.items():
        path = directory / name
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size <= MAX_FILE_SIZE,
                f"Invalid distribution file: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f"Distribution hash mismatch: {name}")


def index_record(index: str, version: str) -> dict:
    check_version(version)
    host = INDEX_HOSTS[index]
    return json.loads(read_url(f"https://{host}/pypi/scgeo/{version}/json", {host}))


def validate_index(record: dict, index: str, version: str, hashes: dict) -> dict:
    require(record.get("info", {}).get("name", "").lower() == "scgeo"
            and record.get("info", {}).get("version") == version, "Wrong index release")
    entries = record.get("urls", [])
    require(len(entries) == len(hashes), "Index has missing, duplicate or extra files")
    published = {entry["filename"]: entry for entry in entries}
    require(set(published) == set(hashes), "Index filenames differ")
    for name, digest in hashes.items():
        entry = published[name]
        require(not entry.get("yanked") and entry.get("digests", {}).get("sha256") == digest,
                f"Index hash mismatch or yanked file: {name}")
        require(entry.get("packagetype") == ("bdist_wheel" if name.endswith(".whl") else "sdist"),
                "Unexpected distribution type")
        validate_url(entry["url"], FILE_HOSTS[index])
    return published


def verify_index(index: str, version: str, hashes: dict, destination: Path,
                 attempts: int = 1) -> None:
    require(1 <= attempts <= 12 and not destination.exists(), "Invalid verification destination/retries")
    for attempt in range(attempts):
        try:
            record = index_record(index, version)
            if len(record.get("urls", [])) < len(hashes) and attempts > 1:
                raise URLError("Files have not all appeared yet")
            published = validate_index(record, index, version, hashes)
            break
        except HTTPError as exc:
            if exc.code != 404 and exc.code < 500:
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(10)
        except (URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(10)
    payloads = {}
    for name, digest in hashes.items():
        data = read_url(published[name]["url"], FILE_HOSTS[index])
        require(hashlib.sha256(data).hexdigest() == digest, f"Index download differs: {name}")
        payloads[name] = data
    destination.mkdir(parents=True)
    for name, data in payloads.items():
        (destination / name).write_bytes(data)
    check_directory(destination, hashes)
    print(f"PASS: both {index} files match the tested release manifest")


def require_unpublished(version: str) -> None:
    try:
        index_record("pypi", version)
    except HTTPError as exc:
        if exc.code == 404:
            print(f"PASS: no production PyPI record for ScGeo {version}")
            return
        raise
    raise ValueError("This version already exists on production PyPI; stop and inspect, never re-upload")


def source(version: str, run_id: int, destination: Path) -> None:
    check_version(version)
    require(run_id > 0, "Invalid staging run ID")
    base = f"/repos/{REPOSITORY}"
    run = github(f"{base}/actions/runs/{run_id}")
    jobs = github(f"{base}/actions/runs/{run_id}/attempts/1/jobs?per_page=100")
    obj = github(f"{base}/git/ref/tags/v{version}")["object"]
    if obj["type"] == "tag":
        obj = github(f"{base}/git/tags/{obj['sha']}")["object"]
    require(obj["type"] == "commit", "Tag must resolve directly to a commit")
    commit = obj["sha"]
    validate_source(run, jobs, commit, version, run_id)
    # The manual production workflow runs on main with complete history.
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], check=True)
    artifacts = select_artifacts(github(f"{base}/actions/runs/{run_id}/artifacts?per_page=100"),
                                 run_id, commit)
    destination.mkdir(parents=True)
    record = {"repository": REPOSITORY, "version": version, "tag": f"v{version}",
              "source_commit": commit, "run_id": run_id, "run_attempt": 1,
              "artifacts": artifacts}
    (destination / "promotion-source.json").write_text(json.dumps(record, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"version={version}\ncommit={commit}\n")
        output.write(f"dist_id={artifacts['scgeo-distributions']['id']}\n")
        output.write(f"audit_id={artifacts['scgeo-release-audit']['id']}\n")
    print(f"PASS: original successful tag run {run_id}, v{version}, {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    src = sub.add_parser("source")
    src.add_argument("version")
    src.add_argument("run_id", type=int)
    src.add_argument("destination", type=Path)
    for command in ("prepare", "verify-pypi"):
        child = sub.add_parser(command)
        child.add_argument("version")
        child.add_argument("commit")
        child.add_argument("manifest", type=Path)
        child.add_argument("destination", type=Path)
        if command == "prepare":
            child.add_argument("distributions", type=Path)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("version")
    args = parser.parse_args()
    if args.command == "source":
        source(args.version, args.run_id, args.destination)
    elif args.command == "preflight":
        require_unpublished(args.version)
    else:
        hashes = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")),
                                   args.version, args.commit)
        if args.command == "prepare":
            check_directory(args.distributions, hashes)
            verify_index("testpypi", args.version, hashes, args.destination)
        else:
            verify_index("pypi", args.version, hashes, args.destination, attempts=12)


if __name__ == "__main__":
    main()
