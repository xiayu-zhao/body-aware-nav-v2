"""Output validation and safeguards (Sec. III-D, Sec. V-C failure modes).

Four observed failure types and their safeguards:

  1. unresolved references        -> nearest-match fallback (recorded)
  2. schema-noncompliant parses   -> schema-reminder retry
  3. empty / inconsistent masks   -> structural validation, regenerate
  4. unsafe parameterization      -> margin floor from class table

Everything is caught *before* reaching the planner. Open item from the
adversarial study (Sec. V-D): explicit ambiguity detection, clarification
requests, and formal infeasibility checks upstream of planning.
"""

from __future__ import annotations

from bodynav.core.types import ValueMapSet


class SchemaViolation(Exception):
    """Parse does not satisfy TASK_JSON_SCHEMA."""


class MaskValidationError(Exception):
    """Generated masks/maps are empty, mis-shaped, or non-finite."""


class UnsafeParameterization(Exception):
    """Resolved margins fall below the class-table floor."""


def validate_value_maps(maps: ValueMapSet, expected_support: int) -> None:
    """Shape / finiteness / non-empty checks (delegates to ValueMapSet.validate
    plus support-size agreement with the voxel grid)."""
    maps.validate()
    if len(maps.indices) != expected_support:
        raise MaskValidationError(
            f"map support {len(maps.indices)} != grid support {expected_support}"
        )


def check_margins(hazard_margins: dict[str, float], class_floor: dict[str, float]) -> None:
    """Reject margins below the per-class safety floor."""
    for hazard, margin in hazard_margins.items():
        floor = class_floor.get(hazard, 0.0)
        if margin < floor:
            raise UnsafeParameterization(f"{hazard}: {margin} m < floor {floor} m")
