# Publish the staged ScGeo 0.1.0 to real PyPI

This is the one-time production promotion of the already-tested first release.
The workflow does not build a new wheel or source distribution. It leaves the
existing `v0.1.0` tag, TestPyPI files, scientific code and manuscript checkpoint
unchanged. Future release versions require a separately reviewed policy update.

## Owner configuration

On **real PyPI**, the pending/trusted publisher must identify:

| Field | Value |
| --- | --- |
| Project | `scgeo` |
| GitHub owner | `liuifrec` |
| Repository | `scgeo` |
| Workflow filename | `publish-pypi.yml` |
| Environment | `pypi` |

GitHub Settings > Environments > `pypi` must have required reviewer `liuifrec`,
self-review permitted, administrator bypass disabled, and a deployment rule
**Branch: `main`**. This is distinct from TestPyPI's **Tag: `v*`** rule.
Do not change the working TestPyPI configuration. No stored PyPI token is needed.

A successful dry run does not verify the private publisher configuration or
permission to upload. Those are exercised only after the owner's approval.

## Review, merge, then publish

1. Review this PR and its successful dry-run checks, then merge it. Merging runs
   verification only; it does not upload to either index.
2. In Actions, select **Promote ScGeo 0.1.0 to PyPI**.
3. Click **Run workflow**, choose **main**, and check the boolean labelled
   **Publish the verified ScGeo 0.1.0 files to real PyPI (requires approval)**.
   The default is unchecked, which performs another dry run without uploading.
4. Start the run. After the verification jobs pass, it waits at
   **Approve publication to REAL PyPI**.
5. Click **Review deployments**, select **pypi**, review the source and hashes in
   the run summary, and select **Approve and deploy**.
6. Wait for **Verify REAL PyPI hashes and fresh pip installation** to pass.
   This final check, not the upload alone, is the completion point.

Do not create a new version tag, move `v0.1.0`, rebuild the distributions, or
rerun the old TestPyPI publishing workflow. This workflow operates from `main`
and promotes only the previously verified distribution files.

## Fixed provenance and files

- Staging workflow: `.github/workflows/publish.yml`
- Successful staging run: `38683215223`, attempt `1`
- Source tag: `v0.1.0`
- Source commit: `6ab8a8ea4144f66613812b7b79976b1a67ad29d1`
- Original distribution artifact: `11949430229`
- Original audit artifact: `11949625115`

| File | SHA256 |
| --- | --- |
| `scgeo-0.1.0-py3-none-any.whl` | `16c85ba42c66fe75404966daedd4b7595369f7221017167ce29e4cfdca5d45e3` |
| `scgeo-0.1.0.tar.gz` | `87928f91cd5ecbe06347e5ea9f290602030e0af2a7f7a3fd831daaf488cea175` |

These hashes were confirmed by staging verification job `116253744331`.
The promotion checks the successful source run and every required staging job,
its repository/commit/tag, original unexpired artifact IDs and recorded digests,
the release manifest, both original distribution files, and fresh downloads
from TestPyPI. It uploads a new CI artifact containing those exact bytes, not a
new package build. Immediately before uploading, the isolated publishing job
checks the two literal SHA256 values again.

The original staging artifacts expire on **2026-11-14**. An expired, replaced or
missing artifact fails closed; do not silently select a newer run or rebuild.
Resolve that condition with a reviewed provenance-preservation step first.

## Permission and retry boundaries

Only the publishing job receives `id-token: write`, and only an explicit manual
run from `main` with the publish checkbox selected can reach it. That job performs
no source checkout, package build or package import. PRs and ordinary pushes run
read-only verification. An explicit `pypi` environment approval remains required.

The workflow refuses an existing production record for version 0.1.0 instead of
silently skipping uploads. If the upload succeeds but post-upload verification
fails, inspect the error and use **Re-run failed jobs**, not **Re-run all jobs**.
A partial upload, conflicting record or permissions error needs investigation;
never delete published files or change an existing tag to work around it.

After the real-index verification passes, update the installation documentation
to advertise `python -m pip install scgeo`, and submit a Bioconda recipe using the
real PyPI source archive and the same verified checksum. Neither documentation
promotion nor Bioconda submission is performed by this workflow.

The paper's frozen checkpoint remains
`9a0ed16cbaa57f935f9c9bc87d1643a25b51012c`. The publishing workflow commit is not
being represented as the source commit of the already-built 0.1.0 package.

## Official guidance

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Trusted Publishing security model](https://docs.pypi.org/trusted-publishers/security-model/)
- [GitHub artifact downloads](https://github.com/actions/download-artifact)
- [Reviewing deployments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/review-deployments)
