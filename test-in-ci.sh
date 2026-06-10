#!/usr/bin/env sh

set -e

export SCCACHE_GHA_ENABLED=false
export RUSTC_WRAPPER=
hash uv || python3 -m pip install uv
uv venv --allow-existing
uv sync --no-build --no-install-project
mkdir -p tmp_path
PYVER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
(cd tmp_path && unzip ../dist/scors*${PYVER}*.whl)
PYTHONPATH=./tmp_path uv run --no-project pytest -vvs
