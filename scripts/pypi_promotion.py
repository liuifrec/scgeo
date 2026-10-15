"""Verify and promote the already-staged ScGeo 0.1.0 files without rebuilding.

This helper only reads public metadata and downloads checksum-pinned files.
Uploading is performed separately by the approval-gated PyPA action.
Only Python's standard library is required (Python >=3.9).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPOSITORY = "liuifrec/scgeo"
REPOSITORY_ID = 1133169788
VERSION = "0.1.0"
TAG = "v0.1.0"
COMMIT = "6ab8a8ea4144f66613812b7b79976b1a67ad29d1"
RUN_ID = 38683215223
# These hashes were confirmed in the successful staging verification job
# 116253744331, not calculated from a new build.
FILES = {
    "scgeo-0.1.0-py3-none-any.whl": "16c85ba42c66fe75404966daedd4b7595369f7221017167ce29e4cfdca5d45e3",
    "scgeo-0.1.0.tar.gz": "87928f91cd5ecbe06347e5ea9f290602030e0af2a7f7a3fd831daaf488cea175",
}
ARTIFACTS = {
    "scgeo-distributions": (11949430229, "sha256:36c7929caac04b96426b9e5408c9432384fcfb89fe98c49ff0c41fda56b33e62"),
    "scgeo-release-audit": (11949625115, "sha256:a7eea67e06f4f7ff30109045f8471491a4f24623ce901e2cbdcf776c684b7089"),
}
JOBS = {
    "Build and inspect release artifacts",
    "Library suite / Python 3.9",
    "Library suite / Python 3.12",
    "Optional installation / scanpy,scvelo",
    "Optional installation / census",
    "Optional installation / notebook",
    "Native conda build from inspected sdist",
    "Approve upload to TestPyPI",
    "Verify TestPyPI file hashes and clean installation",
}
INDEX_HOSTS = {"testpypi": "test.pypi.org", "pypi": "pypi.org"}
FILE_HOSTS = {
    "testpypi": {"test-files.pythonhosted.org", "files.pythonhosted.org"},
    "pypi": {"files.pythonhosted.org"},
}
MAX_FILE_SIZE = 32 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_url(url: str, hosts: set[str]) -> None:
    parsed = urlparse(url)
    require(parsed.scheme == "https" and parsed.hostname in hosts,
            "Unexpected URL scheme or host")
    require(parsed.username is None and parsed.password is None
            and parsed.port in (None, 443) and not parsed.fragment,
            "Unexpected URL credentials, port or fragment")


class NoRedirect(HTTPRedirectHandler):
    """Fail closed: never forward a GitHub token to a redirected URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect refused during release verification")


def read_url(url: str, hosts: set[str], *, token: str = "", limit: int = MAX_FILE_SIZE) -> bytes:
    validate_url(url, hosts)
    headers = {"User-Agent": "scgeo-release-verification"}
    if token:
        require(urlparse(url).hostname == "api.github.com", "Token must only go to GitHub API")
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with build_opener(NoRedirect()).open(request, timeout=30) as response:
        validate_url(response.geturl(), hosts)
        data = response.read(limit + 1)
    require(len(data) <= limit, "Response exceeds release size limit")
    return data


def github(path: str) -> dict:
    require(path.startswith(f"/repos/{REPOSITORY}/"), "Unexpected GitHub API path")
    return json.loads(read_url("https://api.github.com" + path, {"api.github.com"},
                              token=os.environ.get("GH_TOKEN", "")))


def validate_source(run: dict, jobs: dict, tag_commit: str) -> None:
    require(run.get("id") == RUN_ID and run.get("run_attempt") == 1, "Unexpected staging run or attempt")
    require(run.get("repository", {}).get("id") == REPOSITORY_ID
            and run.get("repository", {}).get("full_name") == REPOSITORY
            and run.get("head_repository", {}).get("id") == REPOSITORY_ID,
            "Unexpected staging repository")
    require(run.get("event") == "push" and run.get("path") == ".github/workflows/publish.yml",
            "Staging must be the approved tag-push workflow, not a PR")
    require(run.get("head_branch") == TAG and run.get("head_sha") == COMMIT
            and tag_commit == COMMIT, "Staging source or tag has changed")
    require(run.get("status") == "completed" and run.get("conclusion") == "success",
            "Staging did not complete successfully")
    entries = jobs.get("jobs", [])
    require(jobs.get("total_count") == len(entries) == len(JOBS), "Incomplete or unexpected job list")
    require({j.get("name") for j in entries} == JOBS, "Staging job names differ")
    require(all(j.get("status") == "completed" and j.get("conclusion") == "success"
                and j.get("run_id") == RUN_ID for j in entries), "A staging job did not pass")


def validate_artifact(record: dict, name: str) -> None:
    artifact_id, digest = ARTIFACTS[name]
    require(record.get("id") == artifact_id and record.get("name") == name
            and record.get("digest") == digest and record.get("expired") is False,
            f"Missing, expired or changed staging artifact: {name}")
    origin = record.get("workflow_run", {})
    require(origin.get("id") == RUN_ID and origin.get("head_sha") == COMMIT
            and origin.get("repository_id") == REPOSITORY_ID,
            "Artifact is not from the approved staging source")


def check_source(destination: Path) -> None:
    base = f"/repos/{REPOSITORY}"
    run = github(f"{base}/actions/runs/{RUN_ID}")
    jobs = github(f"{base}/actions/runs/{RUN_ID}/attempts/1/jobs?per_page=100")
    obj = github(f"{base}/git/ref/tags/{TAG}")["object"]
    if obj["type"] == "tag":
        obj = github(f"{base}/git/tags/{obj['sha']}")["object"]
    require(obj["type"] == "commit", "Tag does not directly resolve to a commit")
    validate_source(run, jobs, obj["sha"])
    for name, (artifact_id, _) in ARTIFACTS.items():
        validate_artifact(github(f"{base}/actions/artifacts/{artifact_id}"), name)
    destination.mkdir(parents=True, exist_ok=True)
    record = {"repository": REPOSITORY, "run_id": RUN_ID, "run_attempt": 1,
              "source_commit": COMMIT, "tag": TAG, "version": VERSION,
              "files": FILES, "artifact_ids": {k: v[0] for k, v in ARTIFACTS.items()}}
    (destination / "promotion-source.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: successful staging run {RUN_ID}, unchanged {TAG}, and original artifacts")


def validate_manifest(manifest: dict) -> None:
    require(manifest.get("name") == "scgeo" and manifest.get("version") == VERSION,
            "Incorrect manifest package/version")
    require(manifest.get("source_commit") == COMMIT and manifest.get("git_ref") == f"refs/tags/{TAG}",
            "Incorrect manifest source")
    require(manifest.get("files") == FILES, "Manifest does not match approved staging hashes")


def check_directory(directory: Path) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "Distribution directory is missing or linked")
    require({p.name for p in directory.iterdir()} == set(FILES), "Unexpected or missing distribution files")
    for name, digest in FILES.items():
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "Distribution is not a regular file")
        require(path.stat().st_size <= MAX_FILE_SIZE, "Oversized distribution")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Distribution hash mismatch: {name}")


def index_record(index: str) -> dict:
    host = INDEX_HOSTS[index]
    return json.loads(read_url(f"https://{host}/pypi/scgeo/{VERSION}/json", {host}))


def validate_index(record: dict, index: str) -> dict:
    require(record.get("info", {}).get("name") == "scgeo"
            and record.get("info", {}).get("version") == VERSION, "Index returned a different release")
    entries = record.get("urls", [])
    require(len(entries) == len(FILES), "Index contains missing, duplicate or extra files")
    published = {entry["filename"]: entry for entry in entries}
    require(set(published) == set(FILES), "Index distribution filenames differ")
    for name, digest in FILES.items():
        entry = published[name]
        require(not entry.get("yanked") and entry.get("digests", {}).get("sha256") == digest,
                f"Index hash mismatch or yanked file: {name}")
        expected_type = "bdist_wheel" if name.endswith(".whl") else "sdist"
        require(entry.get("packagetype") == expected_type, "Unexpected distribution type")
        validate_url(entry["url"], FILE_HOSTS[index])
    return published


def download_registry(index: str, destination: Path, *, attempts: int = 1) -> None:
    require(1 <= attempts <= 12, "Invalid retry limit")
    require(not destination.exists(), "Download destination must not already exist")
    for attempt in range(attempts):
        try:
            record = index_record(index)
            # After an upload, the two files may appear at slightly different times.
            if len(record.get("urls", [])) < len(FILES) and attempts > 1:
                raise URLError("Release files have not all appeared yet")
            published = validate_index(record, index)
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
    # Validate all bytes before writing any package file.
    payloads = {}
    for name, digest in FILES.items():
        data = read_url(published[name]["url"], FILE_HOSTS[index])
        require(hashlib.sha256(data).hexdigest() == digest, f"Downloaded file hash mismatch: {name}")
        payloads[name] = data
    destination.mkdir(parents=True)
    for name, data in payloads.items():
        (destination / name).write_bytes(data)
    check_directory(destination)
    print(f"PASS: both {index} files match the original tested SHA256 hashes")


def require_unpublished() -> None:
    try:
        index_record("pypi")
    except HTTPError as exc:
        if exc.code == 404:
            print("PASS: this version is not present on production PyPI; name ownership is checked on upload")
            return
        raise
    raise ValueError("ScGeo 0.1.0 already has a PyPI record. Stop and inspect it; do not re-upload or overwrite.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source")
    source.add_argument("destination", type=Path)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("audit", type=Path)
    prepare.add_argument("distributions", type=Path)
    prepare.add_argument("destination", type=Path)
    sub.add_parser("preflight")
    verify = sub.add_parser("verify-pypi")
    verify.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "source":
        check_source(args.destination)
    elif args.command == "prepare":
        validate_manifest(json.loads((args.audit / "release-manifest.json").read_text(encoding="utf-8")))
        check_directory(args.distributions)
        download_registry("testpypi", args.destination)
    elif args.command == "preflight":
        require_unpublished()
    else:
        download_registry("pypi", args.destination, attempts=12)


if __name__ == "__main__":
    main()
