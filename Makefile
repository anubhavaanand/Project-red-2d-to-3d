# Image-to-3D Benchmark — developer tasks.
#
# PKG-15: a hashed lockfile makes installs reproducible, but it is only useful if
# regenerating it is one command. `make lock` refreshes it from pyproject.toml;
# `make install-core` installs from it.
#
# Requires uv (https://docs.astral.sh/uv/).
#
# Only the `core` slice is locked. The engines/app/api/notebooks/sf3d extras all
# pull in torch and its CUDA tree (via `engines`), and those contain git
# dependencies (flash-attn, diff-gaussian-rasterization, diffoctreerast) that
# cannot be hashed. Locking them would produce a ~2 GB lockfile that is really
# just the engine install again, so they stay on docs/COMPILE.md.

PYTHON ?= python3.11
UV ?= uv
LOCK := requirements/lock-core.txt

.PHONY: help lock install-core install-dev test lint

help:
	@echo "make lock          regenerate the hashed core lockfile from pyproject.toml"
	@echo "make install-core  install the locked core stack (no torch)"
	@echo "make install-dev   editable install plus dev tooling"
	@echo "make test          run the test suite"
	@echo "make lint          ruff check + format check"

# Pinned to one Python version so the resolution is stable; bump deliberately.
lock:
	$(UV) pip compile --python-version 3.11 --generate-hashes \
		pyproject.toml --extra core --output-file $(LOCK)
	@echo "lock refreshed: $(LOCK)"

# Installing from a lockfile is the reproducible path; `pip install -e .` is not.
install-core:
	$(UV) pip install --python $(PYTHON) -r $(LOCK)
	$(UV) pip install --python $(PYTHON) -e . --no-deps

install-dev:
	$(UV) pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/