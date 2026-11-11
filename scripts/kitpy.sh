#!/usr/bin/env bash
# Run a Python script against Isaac Sim's USD bindings without booting Kit.
#
# The BodyNav venv is Python 3.10 and has no `pxr`; Isaac ships USD only for
# its own 3.12 interpreter, and that interpreter has no numpy on its
# default path.  This wires up both:
#
#     scripts/kitpy.sh scripts/usd_export_points.py --stage ...
#
# Anything touching USD runs here; everything else runs in BodyNav/.venv.
set -euo pipefail
ISAAC="${ISAAC_SIM_ROOT:-$HOME/isaacsim}"

find_one() {  # first match for a glob, or empty
  local m=( $1 ); [[ -e "${m[0]}" ]] && echo "${m[0]}" || echo ""
}
USD_EXT="$(find_one "$ISAAC/extscache/omni.usd.libs-*")"
NUMPY_EXT="$(find_one "$ISAAC/extscache/omni.kit.pip_archive-*/pip_prebundle")"
KITPY="$ISAAC/kit/python/bin/python3"

for p in "$USD_EXT" "$NUMPY_EXT" "$KITPY"; do
  if [[ -z "$p" || ! -e "$p" ]]; then
    echo "kitpy.sh: missing Isaac component (looked under $ISAAC)." >&2
    echo "Set ISAAC_SIM_ROOT if Isaac Sim lives elsewhere." >&2
    exit 1
  fi
done

export PYTHONPATH="$USD_EXT:$NUMPY_EXT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$USD_EXT/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$KITPY" "$@"
