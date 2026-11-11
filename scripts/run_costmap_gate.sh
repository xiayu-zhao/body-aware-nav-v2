#!/usr/bin/env bash
# Grounded-costmap baseline: the stored LLM grounding
# of each run on a conventional costmap, hazards inflated by the grounded
# margins as hard obstacles. No model calls. Same chunking as
# run_revision_controls.sh.
#
#   scripts/run_costmap_gate.sh scene2
set -euo pipefail
SITE=$1
cd "$(dirname "$0")/.."
PY=${PY:-"$(pwd)/.venv/bin/python"}
OUT=results/$SITE/costmap_gate
mkdir -p "$OUT"
METHODS="--method costmap-gate --method costmap3d-gate"

run_chunks () {   # $1 robot args, $2 stored batch, $3 tag, $4 n tasks
  for skip in $(seq 0 4 $(( $4 - 1 ))); do
    f=$OUT/$3_s$skip.json
    [ -s "$f" ] && { echo "skip $f"; continue; }
    PYTHONPATH=. "$PY" scripts/run_batch.py --site "$SITE" $1 --skip "$skip" --instructions 4 \
      --seeds 3 $METHODS --replay-from "$2" --out "$f" 2>&1 | grep -E "wrote|Error|Traceback" || true
  done
}

run_chunks "--robot husky_ur5" results/$SITE/batch.json husky 12
run_chunks "--robot unitree_h1" results/$SITE/batch.json h1 12
run_chunks "--robot crazyflie21" results/$SITE/batch_uav.json uav 12

PYTHONPATH=. "$PY" scripts/run_batch.py --site "$SITE" --merge $OUT/*_s*.json \
  --out results/$SITE/batch_costmap_gate.json | tail -3
