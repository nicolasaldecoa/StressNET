# StressNET

**StressNET: an adaptable deep-learning model for mechanical stress inference in tissues**

[![CI](https://github.com/nicolasaldecoa/StressNET/actions/workflows/ci.yml/badge.svg)](https://github.com/nicolasaldecoa/StressNET/actions/workflows/ci.yml)
[![Docs](https://github.com/nicolasaldecoa/StressNET/actions/workflows/docs.yml/badge.svg)](https://github.com/nicolasaldecoa/StressNET/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/stressnet.svg)](https://pypi.org/project/stressnet/)
[![Python](https://img.shields.io/pypi/pyversions/stressnet.svg)](https://pypi.org/project/stressnet/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
<!-- Replace the placeholder below with the real badge after the first Zenodo release -->
[![DOI](https://zenodo.org/badge/nicolasaldecoa/StressNET.svg)](https://zenodo.org/badge/latestdoi/nicolasaldecoa/StressNET)

---

## What is StressNET?

StressNET is an open-source Python library for data-driven mechanical stress inference in tissues. It implements Graph Neural Network models that infer intercellular stress from tissue geometry, without requiring users to specify a fixed physical model. The package includes pretrained models for stress prediction and utilities to finetune them on in silico and in vivo datasets.

It can be used to:

- preprocess Surface Evolver outputs or skeletonized microscopy images into graph data,
- load pretrained or finetuned StressNET model weights,
- infer stress/tension values on tissue interfaces,
- compare StressNET predictions against ground truth or other method predictions when available,
- plot inferred stresses back on a ForSys frame.


## Installation

```bash
pip install stressnet
```

Or with Poetry:

```bash
poetry add stressnet
```

## Quick start

```python
import stressnet

print(stressnet.__version__)
```

From a repository checkout, run the test suite with:

```bash
poetry install --with dev
poetry run pytest
```

Or, if `make` is available:

```bash
make install-dev
make test
```

## Usage

The recommended entry points are the worked example notebooks in `examples/`:

- `examples/inference_in_silico.ipynb`: inference on synthetic Surface Evolver data.
- `examples/inference_in_vivo.ipynb`: inference on experimental skeleton/myosin image data.
- `examples/finetuning.ipynb`: finetuning a pre-trained StressNET model on user-provided graph data.

These notebooks show the complete workflow, including data loading, model loading, inference, metrics, and plotting.

## AI assistants for users

This repository includes `AGENTS.md`, a user-facing grounding document for AI assistants. It helps assistants write scripts or notebook cells that use StressNET to load data, run inference, compare metrics, and generate plots.

Recommended usage:

- **Cursor:** add `@AGENTS.md` to the chat context and ask for help using StressNET on your data. For always-on project context, Cursor users can also create a `.cursor/rules/*.mdc` rule that points to `AGENTS.md`.
- **GitHub Copilot / VS Code:** ask Copilot Chat to read `AGENTS.md` before writing a StressNET script or notebook cell. If your setup uses repository custom instructions, copy or reference the same content from `.github/copilot-instructions.md`; VS Code/Copilot also supports `AGENTS.md` when agent-instruction support is enabled.
- **Claude Code / Claude:** ask Claude to read `AGENTS.md` first, then provide your input file paths and desired outputs. If you prefer Claude-specific project memory, copy or reference the same content from `CLAUDE.md`.

To avoid stale duplicated instructions, treat `AGENTS.md` as the source document and make tool-specific files point back to it when possible.

## Development setup

Requires [Poetry](https://python-poetry.org/) and **Python 3.10** (TensorFlow 2.8 / NumPy constraints in `pyproject.toml`).

```bash
git clone https://github.com/nicolasaldecoa/StressNET.git
cd StressNET
poetry install --with dev
poetry run pytest
```

Common development shortcuts are available through the `Makefile`:

```bash
make lint       # run Ruff checks like CI
make lint-diff  # preview Ruff auto-fixes
make lint-fix   # apply Ruff auto-fixes
make test       # run pytest with coverage
make docs       # build local HTML documentation
make build      # build wheel and source distribution
make check      # run lint, tests, and package build
```

## Documentation

- **Read the Docs:** after maintainers create the RTD project, set the canonical URL in `pyproject.toml` and badges here (placeholder: `TBD_READTHEDOCS_URL`).
- **Local HTML build:**

```bash
poetry install --with docs
poetry run python docs/build_docs.py
```

Or, with `make`:

```bash
make install-docs
make docs
```

Open `docs/_build/html/index.html` in a browser. Maintainer-only release notes live in `README_MAINTAINERS.md`.


## Running examples

The example notebooks assume the repository layout is available. Clone the repo, start Jupyter from the repository root, and open notebooks under `examples/` so relative paths such as `examples/example_data/...` resolve as written.

Installing `stressnet` from PyPI provides the library package only; running the publication examples also requires this repository's `examples/` folder and any downloaded example data.

## Example finetuning dataset

The `radial_1d_2d_inverted_9v` dataset is intentionally not tracked in git because of its size.

Download it to the expected examples path with:

```bash
python examples/download_radial_1d_2d_inverted_9v.py
```

You can also import `ensure_dataset_downloaded` from `examples/download_radial_1d_2d_inverted_9v.py` in a notebook.

## How to cite

If you use the StressNET software, cite this repository/software package:

```bibtex
@software{stressnet,
  author  = {Aldecoa Rodrigo, Nicolás},
  title   = {{StressNET}: an adaptable deep-learning model for mechanical stress inference in tissues},
  url     = {https://github.com/nicolasaldecoa/StressNET},
  license = {BSD-3-Clause},
}
```

⚠️ Associated manuscript (unpublished, citation details to be updated after publication):

> **StressNET: an adaptable deep-learning model for mechanical stress inference in tissues**
> Nicolás Aldecoa Rodrigo, Augusto Borges, Jerónimo R. Miranda-Rodriguez, Guillherme Ventura, Jakub Sedzinski, Hernán López-Schier, Osvaldo Chara

⚠️ Once the paper is published, update `CITATION.cff` and this section with the journal, year, and DOI.

## License

[BSD 3-Clause](LICENSE) © Nicolás Aldecoa Rodrigo
