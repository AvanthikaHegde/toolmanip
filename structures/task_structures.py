"""
Task data structures for the ToolManip framework.

Maps to paper equations (1), (2), (3):
  - ObjectProperties: No = {no, po, co}  (Eq. 1)
  - ToolProperties:   Nt = {nt, pt, ct}  (Eq. 2)
  - TaskStep:         si = {ei, ai, oi, phi_i}  (Eq. 3)

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ObjectProperties:
    """
    Represents target object properties — No = {no, po, co}.
    Paper Eq. (1): name, position, color of the target object.
    cx, cy: center pixel coordinates in the captured image.
    """
    name: str
    position: str
    color: str
    cx: int = 0
    cy: int = 0


@dataclass
class ToolProperties:
    """
    Represents selected tool properties — Nt = {nt, pt, ct}.
    Paper Eq. (2): name, position, color of the selected tool.
    cx, cy: center pixel coordinates in the captured image (used by robot for pick-up).
    """
    name: str
    position: str
    color: str
    cx: int = 0
    cy: int = 0


@dataclass
class TaskStep:
    """
    Represents a single task step — si = {ei, ai, oi, phi_i}.
    Paper Eq. (3): entity, action, target object, success criterion.

    success_criterion must be a concrete, observable, checkable condition
    so Module IV (Execution Monitoring) can verify completion.
    """
    step_index: int
    entity: str
    action: str
    target_object: str
    success_criterion: str


@dataclass
class TaskPlan:
    """
    Full task plan produced by Module I (Task Understanding and Planning).
    Consumed by Modules II, III, and IV downstream.
    """
    task_id: str
    task: str
    target_object: ObjectProperties
    selected_tool: ToolProperties
    steps: List[TaskStep] = field(default_factory=list)
