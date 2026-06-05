# StressNET

**StressNET: an adaptable deep-learning model for mechanical stress inference in tissues**

[![CI](https://github.com/nicolasaldecoa/StressNET/actions/workflows/ci.yml/badge.svg)](https://github.com/nicolasaldecoa/StressNET/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/stressnet/badge/?version=stable)](https://stressnet.readthedocs.io/en/stable/)
[![Docs build](https://github.com/nicolasaldecoa/StressNET/actions/workflows/docs.yml/badge.svg)](https://github.com/nicolasaldecoa/StressNET/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/stressnet.svg)](https://pypi.org/project/stressnet/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=ffd43b)](https://pypi.org/project/stressnet/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1208779358.svg)](https://doi.org/10.5281/zenodo.20481460)

---

## What is StressNET?

StressNET is an open-source Python library for data-driven mechanical stress inference in tissues. It implements Graph Neural Network models that infer intercellular stress from tissue geometry, without requiring users to specify a fixed physical model. The package includes pretrained models for stress prediction and utilities to finetune them on in silico and in vivo datasets.

It can be used to:

- preprocess Surface Evolver outputs or skeletonized microscopy images into graph data,
- load pretrained or finetuned StressNET model weights,
- infer stress/tension values on tissue interfaces,
- compare StressNET predictions against ground truth or other method predictions when available,
- plot inferred stress maps.


## Installation

```text
pip install stressnet
```

Or with Poetry:

```text
poetry add stressnet
```

## Quick start

```python
import stressnet

print(stressnet.__version__)
```

From a repository checkout, run the test suite with:

```text
poetry install --with dev
poetry run pytest
```

Or, if `make` is available:

```text
make install-dev
make test
```

## Usage

The recommended entry points are the worked example notebooks in `examples/`:

- `examples/inference_in_silico.ipynb`: inference on synthetic Surface Evolver data.
- `examples/inference_in_vivo.ipynb`: inference on experimental skeleton/myosin image data.
- `examples/finetuning.ipynb`: finetuning a pre-trained StressNET model on user-provided graph data.

These notebooks show the complete workflow, including data loading, model loading, inference, metrics, plotting and fine-tuning.

## AI assistants for users

This repository includes `AGENTS.md`, a user-facing grounding document for AI assistants. It helps assistants write scripts or notebook cells that use StressNET to load data, run inference, compare metrics, generate plots and fine-tune models on new datasets.

To avoid stale duplicated instructions, treat `AGENTS.md` as the source document and make tool-specific files point back to it when possible.

## Development setup

Requires [Poetry](https://python-poetry.org/) and **Python 3.10** (TensorFlow 2.8 / NumPy constraints in `pyproject.toml`).

```text
git clone https://github.com/nicolasaldecoa/StressNET.git
cd StressNET
poetry install --with dev
poetry run pytest
```

Common development shortcuts are available through the `Makefile`:

```text
make lint       # run Ruff checks like CI
make lint-diff  # preview Ruff auto-fixes
make lint-fix   # apply Ruff auto-fixes
make test       # run pytest with coverage
make docs       # build local HTML documentation
make build      # build wheel and source distribution
make check      # run lint, tests, and package build
```

## Documentation

- **Read the Docs (stable):** [https://stressnet.readthedocs.io/en/stable/](https://stressnet.readthedocs.io/en/stable/) (development: [latest](https://stressnet.readthedocs.io/en/latest/)).
- **Local HTML build:**

```text
poetry install --with docs
poetry run python docs/build_docs.py
```

Or, with `make`:

```text
make install-docs
make docs
```

Open `docs/_build/html/index.html` in a browser.


## Running examples

The example notebooks assume the repository layout is available. Clone the repo, start Jupyter from the repository root, and open notebooks under `examples/` so relative paths such as `examples/example_data/...` resolve as written.

Installing `stressnet` from PyPI provides the library package only; running the publication examples also requires this repository's `examples/` folder and any downloaded example data.

## Example finetuning dataset

The `radial_1d_2d_inverted_9v` dataset is intentionally not tracked in git because of its size.

Download it to the expected examples path with:

```text
python examples/download_radial_1d_2d_inverted_9v.py
```

You can also import `ensure_dataset_downloaded` from `examples/download_radial_1d_2d_inverted_9v.py` in a notebook.

## How to cite

If you use the StressNET software, cite this repository/software package:

```bibtex
@software{stressnet,
  author  = {Aldecoa Rodrigo, Nicolás},
  title   = {{StressNET}: an adaptable deep-learning model for mechanical stress inference in tissues},
  version = {1.0.1},
  url     = {https://github.com/nicolasaldecoa/StressNET},
  license = {BSD-3-Clause},
}
```

Software citation metadata is also available in [`CITATION.cff`](CITATION.cff).

The software accompanies the following research article (currently unpublished; a DOI will be added once available):

> Aldecoa Rodrigo, N. *, Borges, A. *, Miranda-Rodriguez, J. R., Ventura, G., Sedzinski, J., López-Schier, H., & Chara, O.
> *StressNET: an adaptable deep-learning model for mechanical stress inference in tissues.*
>
> *These authors contributed equally.

## License

[BSD 3-Clause](LICENSE) © Nicolás Aldecoa Rodrigo
