#!/usr/bin/env sh

set -e

hash uv || python3 -m pip install uv
uv venv --python=3.12 --allow-existing
uv sync --no-build --no-install-project
uv pip install dist/*.whl
uv run pytest -vvs
