.PHONY: help install install-dev install-docs lint lint-diff lint-fix test docs build check clean

POETRY ?= poetry
PYTHON ?= $(POETRY) run python
RUFF_TARGETS ?= stressnet tests

help:
	@echo "StressNET development commands"
	@echo ""
	@echo "  make install       Install package dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make install-docs  Install documentation dependencies"
	@echo "  make lint          Run Ruff checks like CI"
	@echo "  make lint-diff     Preview Ruff auto-fixes without applying them"
	@echo "  make lint-fix      Apply Ruff auto-fixes"
	@echo "  make test          Run pytest with configured coverage"
	@echo "  make docs          Build local Sphinx HTML docs"
	@echo "  make build         Build wheel and source distribution"
	@echo "  make check         Run lint, tests, and package build"
	@echo "  make clean         Remove local build/test caches"

install:
	$(POETRY) install

install-dev:
	$(POETRY) install --with dev

install-docs:
	$(POETRY) install --with docs

lint:
	$(POETRY) run ruff check $(RUFF_TARGETS)

lint-diff:
	$(POETRY) run ruff check $(RUFF_TARGETS) --diff

lint-fix:
	$(POETRY) run ruff check $(RUFF_TARGETS) --fix

test:
	$(POETRY) run pytest

docs:
	$(PYTHON) docs/build_docs.py

build:
	$(POETRY) build

check: lint test build

clean:
	$(PYTHON) -c "import pathlib, shutil; paths = ['dist', 'build', 'htmlcov', 'docs/_build', '.pytest_cache', '.ruff_cache', '.coverage', '.tmp-wheel-test']; [shutil.rmtree(p, ignore_errors=True) if pathlib.Path(p).is_dir() else pathlib.Path(p).unlink(missing_ok=True) for p in paths]"
