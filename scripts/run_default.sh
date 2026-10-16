#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_default.sh <instance.xml>

Runs one Choco solve with normal/default propagator priorities and prints only
the XCSP solution output.
USAGE
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTANCE="$(readlink -f "$1")"
JAR="$PROJECT_ROOT/choco/dist/choco-rank.jar"

if [[ ! -f "$INSTANCE" ]]; then
  echo "Missing instance XML: $INSTANCE" >&2
  exit 1
fi

if [[ ! -f "$JAR" ]]; then
  echo "Missing Rank Choco jar: $JAR" >&2
  exit 1
fi

exec java -XX:+PerfDisableSharedMem -jar "$JAR" --solve-default "$INSTANCE"
