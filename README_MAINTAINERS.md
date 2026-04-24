# Maintainer Handover

This file is for repository maintainers preparing the publication companion release.

Do **not** invent DOIs, ORCIDs, affiliations, or author names. Replace every `TBD` or `...` with real values when available, or remove placeholder entries before the public release.

## Read the Docs

1. Create a project at <https://readthedocs.org/> linked to this GitHub repository.
2. Set the default branch and Python version to **3.10**. This matches the TensorFlow 2.8 pin in `pyproject.toml`.
3. Confirm `.readthedocs.yaml` at the repository root is detected.
4. After the first successful build, set the canonical documentation URL, often `https://<project-slug>.readthedocs.io/en/stable/`, in:
   - `pyproject.toml` under `[tool.poetry]` → `documentation`
   - `README.md`, replacing `TBD_READTHEDOCS_URL` if still present

## Zenodo

1. Enable the Zenodo-GitHub integration for this repository.
2. On each **GitHub Release**, Zenodo creates an archive deposit. Review `.zenodo.json` metadata before or immediately after the first release.
3. Replace empty strings and `"..."` placeholder creators in `.zenodo.json` with real names, affiliations, and ORCIDs.
4. Add a related identifier for the peer-reviewed article (`scheme: "doi"`) when the paper DOI exists.

## Citation File

1. Update `CITATION.cff` with the final software version, release date, authors, and author ORCIDs.
2. Uncomment and complete `preferred-citation` once the associated article is published.
3. Do not publish `TBD` DOI, journal, or author values in active citation metadata.

## PyPI and GitHub Actions

1. Configure **Trusted Publishing** on PyPI for the `stressnet` project so the `publish` GitHub environment can upload wheels/sdists.
2. Ensure the GitHub repository has an environment named `pypi`.
3. Publish a GitHub Release whose tag matches semantic versioning, for example `v0.1.0`.
4. The publish workflow strips the leading `v` and runs `poetry version` before building.

## Continuous Integration

CI is pinned to **Python 3.10** because the declared dependencies (TensorFlow 2.8 / NumPy pin) do not support Python 3.11+ in this project.

If the dependency stack is upgraded, revisit both:

- `python` in `pyproject.toml`
- the Python version in GitHub Actions workflows

## Before the First Public Release

Use this checklist before tagging the repository for the article companion release:

- Replace all placeholder authors (`...`) with the final author/contributor list in `README.md`, `CITATION.cff`, `.zenodo.json`, `LICENSE`, and `docs/conf.py`.
- Fill real affiliations and ORCIDs in `.zenodo.json` / `CITATION.cff` where available. Leave ORCID fields absent or empty rather than using fake values.
- Add the article DOI to `CITATION.cff` as `preferred-citation` and to Zenodo related identifiers once the DOI exists.
- Create the Read the Docs project, confirm the first build, and replace `TBD_READTHEDOCS_URL` / package documentation URLs with the canonical RTD URL.
- Confirm PyPI Trusted Publishing is configured for the `pypi` GitHub environment before publishing a release.
- Run the full pytest suite with coverage locally and confirm CI is green on a pull request.
- Re-run the example notebooks from a fresh clone / clean environment and confirm downloaded example data paths resolve.
- Create a GitHub Release with a semantic version tag such as `v0.1.0`; confirm Zenodo archives the release and minting metadata is correct before sharing the DOI.
