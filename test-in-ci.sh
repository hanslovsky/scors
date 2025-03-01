#!/usr/bin/env sh

set -e

export SCCACHE_GHA_ENABLED=false
export RUSTC_WRAPPER=
hash uv || python3 -m pip install uv
uv venv --python=3.12 --allow-existing
uv sync --no-build --no-install-project
uv pip install dist/scors-*cp312-cp312*whl
uv run pytest -vvs
