#!/usr/bin/env bash

set -e

python3 -m pip install uv

if [ ! -d .venv ]; then
    uv venv --python=3.12
fi

source .venv/bin/activate

uv sync

python -m pytest -vvs
