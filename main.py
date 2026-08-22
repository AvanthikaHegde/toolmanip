"""
ToolManip — Module I entry point.

Run with:
    python main.py

Requires OPENAI_API_KEY set in .env (except in mock mode).
Set IMAGE_MODE=mock in .env to test without an API key.

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).
"""

from agents.VLM1_TaskPlanning import run_task_planning
from agents.VLM2_AffordanceReasoning import run_affordance_reasoning
from structures.affordance_structures import AffordanceCoordinates


if __name__ == "__main__":
    plan = run_task_planning()

    print("\n=== RAW VLM OUTPUTS ===")
    # Raw outputs are stored in memory; re-load from results for display
    import json
    from pathlib import Path
    result_file = Path("results") / f"task_{plan.task_id}.json"
    memory_file = Path("memory") / "memory_1.json"

    if memory_file.exists():
        with open(memory_file) as f:
            records = json.load(f)
        record = next((r for r in records if r["task_id"] == plan.task_id), {})
        raw = record.get("raw_outputs", {})
        print(f"  Step 1 (Task Understanding): {raw.get('step1', '')}")
        print(f"  Step 2 (Tool Identification): {raw.get('step2', '')}")
        print(f"  Step 3 (Tool Selection):      {raw.get('step3', '')}")
        print(f"  Step 4 (Task Planning):\n    {raw.get('step4', '')}")

    print("\n=== PARSED TASK PLAN ===")
    print(f"Task: {plan.task}")
    print(
        f"Target: {plan.target_object.name} | "
        f"{plan.target_object.position} | "
        f"{plan.target_object.color} | "
        f"coords=({plan.target_object.cx}, {plan.target_object.cy})"
    )
    print(
        f"Tool:   {plan.selected_tool.name} | "
        f"{plan.selected_tool.position} | "
        f"{plan.selected_tool.color} | "
        f"coords=({plan.selected_tool.cx}, {plan.selected_tool.cy})"
    )
    for step in plan.steps:
        print(f"Step {step.step_index}: [{step.entity}] {step.action} on {step.target_object}")
        print(f"  ✓ {step.success_criterion}")

    print(f"\n[Memory saved] memory/memory_1.json")
    print(f"[Result saved] results/task_{plan.task_id}.json")

    # ------------------------------------------------------------------
    # Module II — Affordance Reasoning
    # ------------------------------------------------------------------
    affordance = run_affordance_reasoning()
    coords = affordance.coordinates

    print("\n=== AFFORDANCE REASONING ===")
    print(f"\nTOOL ({affordance.tool_name}):")
    print(f"  Grasp Point H:")
    print(f"    pixel      : ({coords.grasp_point_H.u}, {coords.grasp_point_H.v})")
    print(f"    normalized : ({coords.grasp_point_H.u_norm}, {coords.grasp_point_H.v_norm})")
    print(f"  Functional Point F:")
    print(f"    pixel      : ({coords.functional_point_F.u}, {coords.functional_point_F.v})")
    print(f"    normalized : ({coords.functional_point_F.u_norm}, {coords.functional_point_F.v_norm})")
    print(f"  Tool Direction : {coords.tool_direction_angle}°")

    print(f"\nTARGET ({affordance.target_name}):")
    print(f"  Start Point O:")
    print(f"    pixel      : ({coords.start_point_O.u}, {coords.start_point_O.v})")
    print(f"    normalized : ({coords.start_point_O.u_norm}, {coords.start_point_O.v_norm})")
    print(f"  End Point Q:")
    print(f"    pixel      : ({coords.end_point_Q.u}, {coords.end_point_Q.v})")
    print(f"    normalized : ({coords.end_point_Q.u_norm}, {coords.end_point_Q.v_norm})")
    print(f"  Target Direction : {coords.target_direction_angle}°")

    print(f"\nOPERATION VECTOR:")
    print(f"  From     : ({coords.start_point_O.u}, {coords.start_point_O.v})")
    print(f"  To       : ({coords.end_point_Q.u}, {coords.end_point_Q.v})")
    print(f"  Distance : {coords.operation_distance_px} px")
    print(f"  Direction: {coords.operation_vector} (unit vector)")
    print(f"\n  NOTE: When depth camera available, convert using:")
    print(f"  X = (u - cx) * Z / fx")
    print(f"  Y = (v - cy) * Z / fy")
    print(f"  Z = depth at pixel (u, v)")
    print(f"\n[Affordance saved] results/affordance_{affordance.task_id}.json")
    print(f"[Grid saved]       {affordance.verification_grid_path}")
