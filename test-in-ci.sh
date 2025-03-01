#!/usr/bin/env sh

set -e

export SCCACHE_GHA_ENABLED=false
export RUSTC_WRAPPER=
hash uv || python3 -m pip install uv
uv venv --python=3.12 --allow-existing
uv run maturin develop
uv run pytest -vvs
