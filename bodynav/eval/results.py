"""Reading stored result files, including those written before the rename.

The method was called BEAM while the evaluation was run, so every JSON under
``results/`` carries ``beam``-prefixed method names and two ``beam_*`` fields.
The code now writes ``bodynav`` everywhere; rather than rewrite the archived
numbers the paper reports, this module renames the legacy keys on the way in.
Anything already canonical passes through untouched.

    from bodynav.eval.results import load_results
    runs = load_results("results/labfloor/batch.json")["runs"]
    assert runs[0]["method"] == "bodynav"        # stored as "beam"
"""
from __future__ import annotations

import json
from pathlib import Path

#: Method names as stored before the rename -> canonical names.
LEGACY_METHODS = {"beam": "bodynav", "beam-no-agent": "bodynav-no-agent",
                  "beam-fixed-grid": "bodynav-fixed-grid"}
#: Per-run fields that carried the old name.
LEGACY_FIELDS = {"beam_map_m2": "bodynav_map_m2", "beam_voxels": "bodynav_voxels"}


def canonical_method(name: str) -> str:
    """``beam-no-agent`` -> ``bodynav-no-agent``; ``husky_ur5/beam`` -> ``husky_ur5/bodynav``."""
    if name in LEGACY_METHODS:
        return LEGACY_METHODS[name]
    if "/" in name:
        head, _, tail = name.rpartition("/")
        if tail in LEGACY_METHODS:
            return f"{head}/{LEGACY_METHODS[tail]}"
    return name


def canonicalise(obj):
    """Recursively rename legacy method keys, method values and ``beam_*`` fields."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = LEGACY_FIELDS.get(k, canonical_method(k) if isinstance(k, str) else k)
            out[key] = canonical_method(v) if key == "method" and isinstance(v, str) else canonicalise(v)
        return out
    if isinstance(obj, list):
        return [canonicalise(v) for v in obj]
    return obj


def load_results(path: str | Path):
    """Load a result JSON with legacy names mapped to the current ones."""
    return canonicalise(json.loads(Path(path).read_text()))
