#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 <training-launcher-pid>" >&2
  exit 2
fi
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
launcher_pid=$1
while kill -0 "$launcher_pid" 2>/dev/null; do
  sleep 60
done

completed=$(find "$PROJECT_DIR/artifacts/lm" -mindepth 2 -maxdepth 2 -name result.json -type f | wc -l)
if (( completed != 12 )); then
  echo "training launcher ended with only $completed/12 completed runs" >&2
  exit 1
fi
exec bash "$PROJECT_DIR/scripts/evaluate_and_report.sh"
