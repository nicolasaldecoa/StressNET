# Maintainer Handover

This file is for repository maintainers preparing the publication companion release.

Do **not** invent DOIs, ORCIDs, affiliations, or author names. Replace every `TBD` with real values when available, or remove placeholder metadata before the public release.

## Read the Docs

1. Create a project at <https://readthedocs.org/> linked to this GitHub repository.
2. Set the default branch and Python version to **3.10**. This matches the TensorFlow 2.8 pin in `pyproject.toml`.
3. Confirm `.readthedocs.yaml` at the repository root is detected.
4. After the first successful build, set the canonical documentation URL, often `https://<project-slug>.readthedocs.io/en/stable/`, in:
   - `pyproject.toml` under `[tool.poetry]` → `documentation`
   - `README.md`, replacing `TBD_READTHEDOCS_URL` if still present

## Zenodo

Keep Zenodo disabled until the paper companion release is ready.

1. Do **not** enable the Zenodo-GitHub integration for this repository yet.
2. Do **not** create a GitHub Release just for PyPI-only maintenance releases. A git tag is enough for the PyPI workflow.
3. When the paper release is ready, enable the Zenodo-GitHub integration and create a GitHub Release from the final publication tag.
4. On each **GitHub Release**, Zenodo creates an archive deposit. Review `.zenodo.json` metadata before or immediately after the first release.
5. Fill the software creator affiliation and ORCID in `.zenodo.json` if desired.
6. Add a related identifier for the peer-reviewed article (`scheme: "doi"`) when the paper DOI exists.

## Citation File

1. Update `CITATION.cff` with the final software version, release date, software author ORCIDs, and any final software authorship changes. Add other authors / contributors if missing.
2. Uncomment and complete `preferred-citation` once the associated article is published.
3. Do not publish `TBD` DOI, journal, or author values in active citation metadata.

## PyPI and GitHub Actions

The initial `0.1.0` PyPI release is intended only to claim the `stressnet` package name before the paper release. Future package updates should use the same manual workflow.

1. Confirm **Trusted Publishing** remains configured on PyPI for the `stressnet` project so the `publish` GitHub environment can upload wheels/sdists.
2. Ensure the GitHub repository has an environment named `pypi`.
3. Update the version intentionally using a semantic version tag, for example `v0.1.1`, `v0.2.0`, or `v1.0.0`.
4. Create and push the git tag.
5. Run the `Publish to PyPI` GitHub Actions workflow manually with `version_tag` set to that tag.
6. Do not create a GitHub Release for PyPI-only maintenance releases if Zenodo should remain inactive.
7. The publish workflow strips the leading `v` and runs `poetry version` before building.

## Continuous Integration

CI is pinned to **Python 3.10** because the declared dependencies (TensorFlow 2.8 / NumPy pin) do not support Python 3.11+ in this project.

If the dependency stack is upgraded, revisit both:

- `python` in `pyproject.toml`
- the Python version in GitHub Actions workflows

## Before the First Public Release

Use this checklist before tagging the repository for the article companion release:

- Enable Zenodo only when the paper companion GitHub Release is ready.
- Keep code/software authorship metadata (`pyproject.toml`, `LICENSE`, `stressnet/__init__.py`, `.zenodo.json`, `CITATION.cff` software authors) aligned with the final software authorship decision.
- Keep the associated manuscript author list in `README.md` and the commented `preferred-citation` block in `CITATION.cff` aligned with the final paper.
- Fill real affiliations and ORCIDs in `.zenodo.json` / `CITATION.cff` where available. Leave ORCID fields absent or empty rather than using fake values.
- Add the article DOI to `CITATION.cff` as `preferred-citation` and to Zenodo related identifiers once the DOI exists.
- Create the Read the Docs project, confirm the first build, and replace `TBD_READTHEDOCS_URL` / package documentation URLs with the canonical RTD URL.
- Confirm PyPI Trusted Publishing is configured for the `pypi` GitHub environment before publishing a release.
- Run the full pytest suite with coverage locally and confirm CI is green on a pull request.
- Re-run the example notebooks from a fresh clone / clean environment and confirm downloaded example data paths resolve.
- Create a GitHub Release with the final paper companion tag; confirm Zenodo archives the release and minting metadata is correct before sharing the DOI.
