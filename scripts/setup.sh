#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

uv venv --system-site-packages --python 3.11 .venv
UV_LINK_MODE=copy uv pip install --python .venv/bin/python -e '.[dev]'

