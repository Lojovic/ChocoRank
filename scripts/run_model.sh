#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_model.sh <instance.xml> <model-dir>

Runs one Choco solve with the gated Rank policy and prints only the XCSP
solution output. If the model directory has no class_policy.json, this falls
back to ordinary Rank model solving.

The wrapper starts one Java process that builds the Choco model, discovers the
concrete propagator classes, extracts XCSP features, predicts priorities, and
solves with those priorities.

USAGE
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTANCE="$(readlink -f "$1")"
MODEL_DIR="$(readlink -m "$2")"
JAR="$PROJECT_ROOT/choco/dist/choco-rank.jar"

if [[ ! -f "$INSTANCE" ]]; then
  echo "Missing instance XML: $INSTANCE" >&2
  exit 1
fi

if [[ ! -f "$JAR" ]]; then
  echo "Missing Rank Choco jar: $JAR" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Missing Rank model directory: $MODEL_DIR" >&2
  exit 1
fi

exec java -XX:+PerfDisableSharedMem -jar "$JAR" --solve-gated-model \
  "$PROJECT_ROOT" \
  "$MODEL_DIR" \
  "$PROJECT_ROOT/feature_extractor/xcsp2feat" \
  "$INSTANCE"
