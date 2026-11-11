#!/usr/bin/env bash
# Body-surface margin gate (reviewer item): stored programs and grounding,
# gate margins + half-width, per-task maps at the raised margin. No model calls.
#   scripts/run_body_gate.sh <site>
set -euo pipefail
SITE=$1
cd "$(dirname "$0")/.."
PY=${PY:-"$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"}   # override with PY=...
OUT=results/$SITE/body_gate
mkdir -p "$OUT"
run_chunks () {   # $1 robot args, $2 stored batch, $3 tag, $4 n tasks
  for skip in $(seq 0 4 $(( $4 - 1 ))); do
    f=$OUT/$3_s$skip.json
    [ -s "$f" ] && { echo "skip $f"; continue; }
    PYTHONPATH=. "$PY" scripts/run_batch.py --site "$SITE" $1 --skip "$skip" --instructions 4 \
      --seeds 3 --method prog-gate-body --replay-from "$2" --out "$f" 2>&1 \
      | grep -E "^\s*\[|wrote|Error|Traceback" || true
  done
}
run_chunks "--robot husky_ur5" results/$SITE/batch.json husky 13
run_chunks "--robot unitree_h1" results/$SITE/batch.json h1 12
run_chunks "--robot crazyflie21" results/$SITE/batch_uav.json uav 12
PYTHONPATH=. "$PY" scripts/run_batch.py --site "$SITE" --merge $OUT/*_s*.json \
  --out results/$SITE/batch_body_gate.json | tail -3
echo "DONE $SITE"
