"""
VLM1 Task Planning Agent — Module I of the ToolManip framework.

Implements the Task Understanding and Planning module (Paper Section 3.2).
Sends real images to GPT-4o Vision: scene image in Step 1, tool board in Step 2.
Steps 3 and 4 are pure text reasoning.

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from utils import run_vlm_dialog, parse_task_plan, save_memory_record, save_task_plan
from structures.task_structures import TaskPlan, ObjectProperties, ToolProperties, TaskStep

TOOL_IMAGE  = "examples/tool_board.png"
SCENE_IMAGE = "examples/scene.png"

print(f"[Module I] Using tool image: {TOOL_IMAGE}")
print(f"[Module I] Using scene image: {SCENE_IMAGE}")

if not Path(TOOL_IMAGE).exists():
    print(
        f"\n[Module I] ERROR: Tool image not found at '{TOOL_IMAGE}'.\n"
        f"Please place your tool board image at {TOOL_IMAGE} and try again."
    )
    sys.exit(1)

if not Path(SCENE_IMAGE).exists():
    print(
        f"\n[Module I] ERROR: Scene image not found at '{SCENE_IMAGE}'.\n"
        f"Please place your scene image at {SCENE_IMAGE} and try again."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def run_task_planning() -> TaskPlan:
    """
    Execute Module I: Task Understanding and Planning.

    Prompts the user for a task instruction, sends the tool board image to
    GPT-4o Vision for tool identification, and returns a parsed TaskPlan.
    """
    task = input("\nEnter your task instruction (e.g. 'Putty removal', 'Bolt tightening'): ").strip()
    if not task:
        task = "Putty removal"
        print(f"[Module I] No input given, defaulting to: {task}")

    steps = [
        {
            "name": "step1_task_understanding",
            "content": SCENE_IMAGE,
            "is_image": True
        },
        {
            "name": "step2_tool_identification",
            "content": TOOL_IMAGE,
            "is_image": True
        },
        {
            "name": "step3_tool_selection",
            "content": f"Task: {task}\nAvailable tools with coordinates: [see previous step output]",
            "is_image": False,
            "task": task
        },
        {
            "name": "step4_task_planning",
            "content": f"Task: {task}",
            "is_image": False,
            "task": task
        }
    ]

    raw_outputs = run_vlm_dialog(
        agent_idx=1,
        steps=steps,
        history_k=3,
        model="gpt-4o",
        prompt_bundle_name="P1"
    )

    plan = parse_task_plan(raw_outputs, task)
    save_memory_record(plan, raw_outputs)
    save_task_plan(plan)
    return plan
