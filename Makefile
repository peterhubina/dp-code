# dp-code — the five commands that make up the local development loop.
#
#   make install       install the package and its pinned dependencies
#   make smoke         the synthetic end-to-end run: minutes, no real data, no GPU
#   make test          the full pytest suite
#   make check-paths   the repo-wide absolute-path gate
#   make reference     regenerate docs/config-reference.md
#
# Nothing here touches .datasets/ or .scratch/. `make smoke` builds a complete
# miniature cohort inside a pytest tmp_path and trains on it for one fold and two
# epochs with CUDA_VISIBLE_DEVICES emptied; it cannot reach a real run directory
# and cannot take a GPU that a real run is using.
#
# NOTE: `make` itself is NOT in the project's container image (the
# vggfiit/vgg-torch base ships no build tools), so inside the container either
# `apt-get install -y make` first or run the commands directly — each target is
# one line and they are reproduced here so that nothing depends on this file:
#
#   install      pip install -e '.[dev]'
#   smoke        python -m pytest -m smoke -v
#   test         python -m pytest -q
#   check-paths  python tests/check_paths.py
#   reference    python -m dpcode.cli.config reference -o docs/config-reference.md

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.DEFAULT_GOAL := help
.PHONY: help install smoke test check-paths reference

help:
	@echo "make install       pip install -e '.[dev]'  — the only setup step"
	@echo "make smoke         synthetic end-to-end run (CPU, no real data)"
	@echo "make test          the full pytest suite, smoke run included"
	@echo "make check-paths   fail on a machine-specific absolute path"
	@echo "make reference     regenerate docs/config-reference.md"

# Editable, deliberately. A non-editable install copies `project` and `tools`
# into site-packages WITHOUT their sibling data directories (dataset_csv/,
# splits/, tools/data/), which those modules reach by relative path — so the
# copies would import and then read the wrong files, or none.
install:
	$(PYTHON) -m pip install -e '.[dev]'

# The check to run BEFORE downloading 98 GB of features and requesting access to
# two gated HuggingFace repositories.
smoke:
	$(PYTEST) -m smoke -v

test:
	$(PYTEST) -q

check-paths:
	$(PYTHON) tests/check_paths.py

# `dp-config reference` writes the document; this target only says where it goes.
# tests/test_config_compose.py::test_config_reference_is_current fails when the
# committed copy has drifted, because a stale generated reference is worse than
# none.
reference:
	$(PYTHON) -m dpcode.cli.config reference -o docs/config-reference.md
	@echo "wrote docs/config-reference.md"
