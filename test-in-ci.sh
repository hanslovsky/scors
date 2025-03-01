#!/usr/bin/env sh

set -e

export SCCACHE_GHA_ENABLED=false
export RUSTC_WRAPPER=
hash uv || python3 -m pip install uv
uv venv --python=3.12 --allow-existing
uv sync --no-build --no-install-project
mkdir -p tmp_path
(cd tmp_path && unzip ../dist/scors*cp312-cp312*whl)
PYTHONPATH=./tmp_path uv run --no-project pytest -vvs
