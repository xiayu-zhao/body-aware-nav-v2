"""Robot platform profiles instantiating Eqs. 2-4 and the planning gates
(Sec. IV "Simulation and platforms").

Platform differences are confined to this constraint layer — the same
abstraction and planning code serves humanoid, aerial, and ground
platforms (a core claim of the manuscript).

Profiles ship as YAML in ``configs/robots/`` and load via
``load_profile``; the values below mirror the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "robots"


@dataclass(frozen=True)
class RobotProfile:
    """Embodiment constraints consumed across the pipeline.

    Height band feeds Eq. 2; reach/safety feed Eq. 3; foothold band feeds
    phi_R in Eq. 4; slope/clearance bounds feed the planner gates.
    """

    name: str
    kind: str  # "humanoid" | "aerial" | "ugv"
    # -- vertical operational band [h_min, h_max] relative to local ground
    height_band: tuple[float, float]
    # -- workspace kernel (Eq. 3)
    r_reach: float
    r_safety: float
    # -- planner gates (Sec. III-E)
    max_slope_deg: float
    min_lateral_clearance: float
    required_vertical_clearance: float
    # -- humanoid-only foothold refinement band (m); None for aerial/UGV
    foothold_band: float | None = None
    # -- aerial-only speed limits (horizontal, vertical) m/s
    speed_limits: tuple[float, float] | None = None
    # -- sensors (informational; used by the sim bridge)
    sensors: tuple[str, ...] = ()

    @property
    def is_aerial(self) -> bool:
        return self.kind == "aerial"


def load_profile(name: str, config_dir: Path | None = None) -> RobotProfile:
    """Load ``configs/robots/<name>.yaml`` into a RobotProfile."""
    path = (config_dir or _CONFIG_DIR) / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RobotProfile(
        name=data["name"],
        kind=data["kind"],
        height_band=tuple(data["height_band"]),
        r_reach=data["r_reach"],
        r_safety=data["r_safety"],
        max_slope_deg=data["max_slope_deg"],
        min_lateral_clearance=data["min_lateral_clearance"],
        required_vertical_clearance=data["required_vertical_clearance"],
        foothold_band=data.get("foothold_band"),
        speed_limits=tuple(data["speed_limits"]) if data.get("speed_limits") else None,
        sensors=tuple(data.get("sensors", ())),
    )
