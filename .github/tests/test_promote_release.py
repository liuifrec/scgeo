"""Offline tests for future-version production promotion."""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import promote_release as p


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.version = "0.2.0"
        self.commit = "a" * 40
        self.run_id = 1234
        self.repo = {"id": p.REPOSITORY_ID, "full_name": p.REPOSITORY}
        self.run = {
            "id": self.run_id, "run_attempt": 1, "repository": self.repo,
            "head_repository": self.repo, "event": "push", "path": ".github/workflows/publish.yml",
            "head_branch": "v0.2.0", "head_sha": self.commit,
            "status": "completed", "conclusion": "success",
        }
        self.jobs = {"total_count": len(p.JOBS), "jobs": [
            {"name": name, "status": "completed", "conclusion": "success", "run_id": self.run_id}
            for name in sorted(p.JOBS)
        ]}
        self.data = {name: name.encode() for name in p.names(self.version)}
        self.hashes = {name: hashlib.sha256(value).hexdigest() for name, value in self.data.items()}
        self.manifest = {"name": "scgeo", "version": self.version,
                         "source_commit": self.commit, "git_ref": "refs/tags/v0.2.0",
                         "files": self.hashes}
        self.artifacts = {"total_count": 2, "artifacts": [
            {"id": i, "name": name, "digest": "sha256:" + "a" * 64, "expired": False,
             "workflow_run": {"id": self.run_id, "head_sha": self.commit,
                              "repository_id": p.REPOSITORY_ID}}
            for i, name in enumerate(("scgeo-distributions", "scgeo-release-audit"), 1)
        ]}
        self.index = {"info": {"name": "scgeo", "version": self.version}, "urls": [
            {"filename": name, "digests": {"sha256": digest}, "yanked": False,
             "packagetype": "bdist_wheel" if name.endswith(".whl") else "sdist",
             "url": f"https://files.pythonhosted.org/{name}"}
            for name, digest in self.hashes.items()
        ]}

    def test_source_and_required_jobs(self):
        p.validate_source(self.run, self.jobs, self.commit, self.version, self.run_id)
        for field, value in (("event", "pull_request"), ("run_attempt", 2),
                             ("head_branch", "main"), ("head_sha", "b" * 40),
                             ("conclusion", "failure")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_source({**self.run, field: value}, self.jobs, self.commit,
                                  self.version, self.run_id)
        jobs = copy.deepcopy(self.jobs)
        jobs["jobs"][0]["conclusion"] = "skipped"
        with self.assertRaises(ValueError):
            p.validate_source(self.run, jobs, self.commit, self.version, self.run_id)
        with self.assertRaises(ValueError):
            p.validate_source(self.run, self.jobs, "b" * 40, self.version, self.run_id)

    def test_artifacts_reject_wrong_origin_expiration_and_duplicates(self):
        self.assertEqual(len(p.select_artifacts(self.artifacts, self.run_id, self.commit)), 2)
        for field, value in (("expired", True), ("digest", None),
                             ("workflow_run", {"id": 9})):
            records = copy.deepcopy(self.artifacts)
            records["artifacts"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.select_artifacts(records, self.run_id, self.commit)
        records = copy.deepcopy(self.artifacts)
        records["artifacts"].append(records["artifacts"][0])
        records["total_count"] += 1
        with self.assertRaises(ValueError):
            p.select_artifacts(records, self.run_id, self.commit)

    def test_manifest_and_exact_directory(self):
        self.assertEqual(p.validate_manifest(self.manifest, self.version, self.commit), self.hashes)
        for version in ("0.1.0", "v0.2.0", "0.2.0-geometry-core", "0.2.0;echo"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                p.check_version(version)
        with self.assertRaises(ValueError):
            p.validate_manifest({**self.manifest, "git_ref": "refs/heads/main"}, self.version, self.commit)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, data in self.data.items():
                (folder / name).write_bytes(data)
            p.check_directory(folder, self.hashes)
            (folder / next(iter(self.data))).write_bytes(b"modified")
            with self.assertRaises(ValueError):
                p.check_directory(folder, self.hashes)

    def test_index_hash_and_host_guards(self):
        self.assertEqual(set(p.validate_index(self.index, "pypi", self.version, self.hashes)),
                         set(self.hashes))
        for field, value in (("yanked", True), ("url", "https://example.com/file"),
                             ("digests", {"sha256": "b" * 64})):
            record = copy.deepcopy(self.index)
            record["urls"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_index(record, "pypi", self.version, self.hashes)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(p, "index_record", return_value=self.index), \
                 patch.object(p, "read_url", side_effect=lambda url, _: self.data[url.rsplit("/", 1)[1]]):
                p.verify_index("pypi", self.version, self.hashes, Path(tmp) / "verified")

    def test_production_preflight_fails_closed(self):
        with patch.object(p, "index_record", return_value=self.index):
            with self.assertRaises(ValueError):
                p.require_unpublished(self.version)
        for status in (404, 403):
            with patch.object(p, "index_record", side_effect=HTTPError(
                    "https://pypi.org", status, "test", None, None)):
                if status == 404:
                    p.require_unpublished(self.version)
                else:
                    with self.assertRaises(HTTPError):
                        p.require_unpublished(self.version)
