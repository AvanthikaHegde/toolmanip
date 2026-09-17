"""
Structures for Module I on a single-camera rig.

Follows Eq. (1) No = {no, po, co} and Eq. (2) Nt = {nt, pt, ct}, where p is a
POSITION IN WORDS, not a pixel. The paper is explicit that these fields "serve
as descriptive prompts for accurate object detection in the subsequent
Affordance Reasoning module" — Module I names things, Module II locates them.

No pixel coordinates are carried here on purpose. A VLM asked for a tool centre
returns a plausible round number rather than a measurement (observed: a uniform
lattice at x=640/2 with even y spacing, mean error 54 px / 41 mm against the
actual centres), and a number that looks authoritative and is not belongs
nowhere in the record.

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ToolCandidate:
    """One tool the VLM reports on the board — Eq. (2) before selection."""
    label: str                      # "Tool1", "Tool2", ...
    name: str
    colour: str
    position: str                   # in words, e.g. "third from the top"

    def as_prompt(self) -> str:
        """The detection prompt Module II's HRE will pass to LangSAM."""
        return self.name

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "name": self.name,
            "colour": self.colour,
            "position": self.position,
        }


@dataclass
class ModuleOneResult:
    """
    Complete Module I output for the single-camera setup.

    Mirrors task_structures.TaskPlan, plus the full candidate list so a wrong
    selection can be diagnosed without re-running the VLM.
    """
    task_id: str
    task: str
    capture_dir: str
    target_name: str
    target_position: str
    target_colour: str
    candidates: List[ToolCandidate] = field(default_factory=list)
    selected: Optional[ToolCandidate] = None
    reason: str = ""
    steps: List = field(default_factory=list)      # task_structures.TaskStep
    raw_outputs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "capture_dir": self.capture_dir,
            "target_object": {
                "name": self.target_name,
                "position": self.target_position,
                "colour": self.target_colour,
            },
            "candidates": [c.to_dict() for c in self.candidates],
            "selected_tool": self.selected.to_dict() if self.selected else None,
            "reason": self.reason,
            "task_steps": [
                {
                    "step_index": s.step_index,
                    "entity": s.entity,
                    "action": s.action,
                    "target_object": s.target_object,
                    "success_criterion": s.success_criterion,
                }
                for s in self.steps
            ],
            "raw_outputs": self.raw_outputs,
        }


class ToolSelectionError(RuntimeError):
    """Raised when the VLM output cannot be resolved to a tool on the board."""
