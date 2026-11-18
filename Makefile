# Image-to-3D Benchmark — developer tasks.
#
# PKG-15: a hashed lockfile makes installs reproducible, but it is only useful if
# regenerating it is one command. `make lock` refreshes every lockfile from
# pyproject.toml; `make install-core` / `install-api` install from them.
#
# Requires uv (https://docs.astral.sh/uv/). The CUDA-only extras (engines, app,
# notebooks, sf3d) are deliberately NOT locked here — they contain git
# dependencies (flash-attn, diff-gaussian-rasterization, diffoctreerast) that
# cannot be hashed. See docs/COMPILE.md for those.

PYTHON ?= python3.11
UV ?= uv
LOCKS := requirements/lock-core.txt requirements/lock-api.txt

.PHONY: help lock install-core install-api install-dev test lint

help:
	@echo "make lock          regenerate the hashed lockfiles from pyproject.toml"
	@echo "make install-core  install the locked core stack (no torch)"
	@echo "make install-api   install the locked API stack"
	@echo "make install-dev   editable install plus dev tooling"
	@echo "make test          run the test suite"
	@echo "make lint          ruff check + format check"

# Regenerate every lockfile. Pinned to one Python version so the resolution is
# stable; bump deliberately, not accidentally.
lock:
	$(UV) pip compile --python-version 3.11 --generate-hashes \
		pyproject.toml --extra core --output-file requirements/lock-core.txt
	$(UV) pip compile --python-version 3.11 --generate-hashes \
		pyproject.toml --extra api --output-file requirements/lock-api.txt
	@echo "locks refreshed: $(LOCKS)"

# Installing from a lockfile is the reproducible path; `pip install -e .` is not.
install-core:
	$(UV) pip install --python $(PYTHON) -r requirements/lock-core.txt
	$(UV) pip install --python $(PYTHON) -e . --no-deps

install-api: install-core
	$(UV) pip install --python $(PYTHON) -r requirements/lock-api.txt

install-dev:
	$(UV) pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/