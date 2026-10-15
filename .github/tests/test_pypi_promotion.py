"""Offline guard tests for the fixed, first-release PyPI promotion."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pypi_promotion", ROOT / "scripts/pypi_promotion.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def good_run():
    return {"id": m.RUN_ID, "run_attempt": 1,
            "repository": {"id": m.REPOSITORY_ID, "full_name": m.REPOSITORY},
            "head_repository": {"id": m.REPOSITORY_ID}, "event": "push",
            "path": ".github/workflows/publish.yml", "head_branch": m.TAG,
            "head_sha": m.COMMIT, "status": "completed", "conclusion": "success"}


def good_jobs():
    return {"total_count": len(m.JOBS), "jobs": [
        {"name": name, "run_id": m.RUN_ID, "status": "completed", "conclusion": "success"}
        for name in sorted(m.JOBS)]}


def good_manifest():
    return {"name": "scgeo", "version": m.VERSION, "source_commit": m.COMMIT,
            "git_ref": f"refs/tags/{m.TAG}", "files": dict(m.FILES)}


def good_index():
    return {"info": {"name": "scgeo", "version": m.VERSION}, "urls": [
        {"filename": name, "digests": {"sha256": digest}, "yanked": False,
         "packagetype": "bdist_wheel" if name.endswith(".whl") else "sdist",
         "url": f"https://files.pythonhosted.org/packages/{name}"}
        for name, digest in m.FILES.items()]}


class PromotionTests(unittest.TestCase):
    def test_release_is_fixed_to_approved_source(self):
        self.assertEqual(m.RUN_ID, 38683215223)
        self.assertEqual(m.COMMIT, "6ab8a8ea4144f66613812b7b79976b1a67ad29d1")
        self.assertEqual(m.VERSION, "0.1.0")
        self.assertEqual(m.FILES, {
            "scgeo-0.1.0-py3-none-any.whl": "16c85ba42c66fe75404966daedd4b7595369f7221017167ce29e4cfdca5d45e3",
            "scgeo-0.1.0.tar.gz": "87928f91cd5ecbe06347e5ea9f290602030e0af2a7f7a3fd831daaf488cea175"})

    def test_source_success(self):
        m.validate_source(good_run(), good_jobs(), m.COMMIT)

    def test_source_wrong_run_attempt_or_origin(self):
        for key, value in [("id", 1), ("run_attempt", 2), ("event", "pull_request"),
                           ("path", ".github/workflows/other.yml"), ("head_branch", "main"),
                           ("head_sha", "0" * 40), ("status", "in_progress"),
                           ("conclusion", "failure")]:
            with self.subTest(key=key):
                run = good_run()
                run[key] = value
                with self.assertRaises(ValueError):
                    m.validate_source(run, good_jobs(), m.COMMIT)

    def test_source_wrong_repository(self):
        for key in ("repository", "head_repository"):
            run = good_run()
            run[key]["id"] = 1
            with self.assertRaises(ValueError):
                m.validate_source(run, good_jobs(), m.COMMIT)

    def test_moved_tag(self):
        with self.assertRaises(ValueError):
            m.validate_source(good_run(), good_jobs(), "0" * 40)

    def test_jobs_all_required_and_successful(self):
        for conclusion in ("skipped", "failure", "cancelled", None):
            jobs = good_jobs()
            jobs["jobs"][0]["conclusion"] = conclusion
            with self.assertRaises(ValueError):
                m.validate_source(good_run(), jobs, m.COMMIT)

    def test_jobs_incomplete_or_duplicate(self):
        for kind in ("missing", "duplicate", "pagination"):
            jobs = good_jobs()
            if kind == "missing":
                jobs["jobs"].pop()
            elif kind == "duplicate":
                jobs["jobs"][0] = jobs["jobs"][1]
            else:
                jobs["total_count"] += 1
            with self.assertRaises(ValueError):
                m.validate_source(good_run(), jobs, m.COMMIT)

    def test_original_artifacts_only(self):
        for name, (artifact_id, digest) in m.ARTIFACTS.items():
            record = {"id": artifact_id, "name": name, "digest": digest, "expired": False,
                      "workflow_run": {"id": m.RUN_ID, "head_sha": m.COMMIT,
                                       "repository_id": m.REPOSITORY_ID}}
            m.validate_artifact(record, name)
            for key, value in [("id", 0), ("digest", "sha256:" + "0" * 64), ("expired", True)]:
                bad = copy.deepcopy(record)
                bad[key] = value
                with self.assertRaises(ValueError):
                    m.validate_artifact(bad, name)

    def test_manifest_success(self):
        m.validate_manifest(good_manifest())

    def test_manifest_rejects_rebuilds_or_other_versions(self):
        for key, value in [("name", "other"), ("version", "0.2.0"),
                           ("source_commit", "0" * 40), ("git_ref", "refs/heads/main"),
                           ("files", {})]:
            manifest = good_manifest()
            manifest[key] = value
            with self.assertRaises(ValueError):
                m.validate_manifest(manifest)

    def test_expected_download_hosts_only(self):
        for url in ["http://files.pythonhosted.org/a", "https://example.com/a",
                    "https://files.pythonhosted.org.evil.example/a",
                    "https://user@files.pythonhosted.org/a",
                    "https://files.pythonhosted.org:444/a", "https://files.pythonhosted.org/a#b"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                m.validate_url(url, m.FILE_HOSTS["pypi"])

    def test_github_token_not_sent_to_index(self):
        with self.assertRaises(ValueError):
            m.read_url("https://pypi.org/a", {"pypi.org"}, token="not-a-real-token")

    def test_redirect_is_refused(self):
        with self.assertRaises(ValueError):
            m.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example")

    def test_index_success(self):
        self.assertEqual(set(m.validate_index(good_index(), "pypi")), set(m.FILES))

    def test_index_wrong_name_version_type_hash_or_yanked(self):
        for kind in ("name", "version", "packagetype", "hash", "yanked"):
            record = good_index()
            if kind in ("name", "version"):
                record["info"][kind] = "wrong"
            elif kind == "hash":
                record["urls"][0]["digests"]["sha256"] = "0" * 64
            else:
                record["urls"][0][kind] = True if kind == "yanked" else "sdist"
            with self.assertRaises(ValueError):
                m.validate_index(record, "pypi")

    def test_index_duplicate_extra_or_missing_file(self):
        for kind in ("duplicate", "extra", "missing"):
            record = good_index()
            if kind == "duplicate":
                record["urls"][1] = record["urls"][0]
            elif kind == "extra":
                record["urls"].append(record["urls"][0])
            else:
                record["urls"].pop()
            with self.assertRaises(ValueError):
                m.validate_index(record, "pypi")

    def test_production_cannot_use_test_file_host(self):
        record = good_index()
        record["urls"][0]["url"] = "https://test-files.pythonhosted.org/packages/a"
        with self.assertRaises(ValueError):
            m.validate_index(record, "pypi")

    def test_directory_exact_files_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = {n: n.encode() for n in m.FILES}
            hashes = {n: hashlib.sha256(b).hexdigest() for n, b in payloads.items()}
            for name, data in payloads.items():
                (root / name).write_bytes(data)
            with patch.object(m, "FILES", hashes):
                m.check_directory(root)
                (root / "unexpected.txt").write_text("extra")
                with self.assertRaises(ValueError):
                    m.check_directory(root)
                (root / "unexpected.txt").unlink()
                (root / next(iter(hashes))).write_bytes(b"mutated")
                with self.assertRaises(ValueError):
                    m.check_directory(root)

    def test_directory_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("Symlinks unavailable on this platform")
            with self.assertRaises(ValueError):
                m.check_directory(link)

    def test_verified_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "download"
            payloads = {n: n.encode() for n in m.FILES}
            hashes = {n: hashlib.sha256(b).hexdigest() for n, b in payloads.items()}
            with patch.object(m, "FILES", hashes), patch.object(m, "index_record") as index:
                index.return_value = good_index()
                with patch.object(m, "read_url", side_effect=lambda u, h: payloads[u.rsplit("/", 1)[1]]):
                    m.download_registry("testpypi", destination)
                m.check_directory(destination)

    def test_corrupt_download_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "download"
            with patch.object(m, "index_record", return_value=good_index()), \
                 patch.object(m, "read_url", return_value=b"corrupt"):
                with self.assertRaises(ValueError):
                    m.download_registry("testpypi", destination)
            self.assertFalse(destination.exists())

    def test_download_does_not_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            m.download_registry("pypi", Path(tmp))

    def test_post_upload_polling_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(m, "index_record", side_effect=URLError("unavailable")) as getter, \
             patch.object(m.time, "sleep") as sleep:
            with self.assertRaises(URLError):
                m.download_registry("pypi", Path(tmp) / "download", attempts=3)
            self.assertEqual(getter.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_invalid_retry_limit(self):
        for attempts in (0, 13):
            with self.assertRaises(ValueError):
                m.download_registry("pypi", Path("unused"), attempts=attempts)

    def test_absent_production_version(self):
        error = HTTPError("https://pypi.org/a", 404, "not found", {}, io.BytesIO())
        with patch.object(m, "index_record", side_effect=error):
            m.require_unpublished()

    def test_network_failure_is_not_name_availability(self):
        for error in (URLError("offline"), HTTPError("https://pypi.org/a", 503, "error", {}, io.BytesIO())):
            with patch.object(m, "index_record", side_effect=error), self.assertRaises(type(error)):
                m.require_unpublished()

    def test_existing_production_release_blocks_upload(self):
        with patch.object(m, "index_record", return_value=good_index()), self.assertRaises(ValueError):
            m.require_unpublished()


if __name__ == "__main__":
    unittest.main()
