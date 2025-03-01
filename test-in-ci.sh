#!/usr/bin/env sh

set -e

python3 -m pip install uv
uv venv --python=3.12 --allow-existing
source .venv/bin/activate
uv sync
uv run pytest -vvs
