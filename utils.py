"""
VLM engine utilities for the ToolManip framework — Module I.

Handles:
- Environment loading (API key, image mode)
- Image encoding for VLM mode (base64 local files or passthrough URLs)
- Multi-step VLM dialog execution with conversation history
- Output parsing into typed TaskPlan dataclasses
- Memory persistence (memory/memory_1.json)
- Result saving (results/task_<id>.json)

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).
"""

import os
import json
import base64
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from structures.task_structures import (
    ObjectProperties, ToolProperties, TaskStep, TaskPlan
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

MEMORY_DIR = Path("memory")
RESULTS_DIR = Path("results")
MEMORY_FILE = MEMORY_DIR / "memory_1.json"

MEMORY_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Image encoding  (Section 3.1 of paper — multimodal affordance input)
# ---------------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    """
    Encode an image for GPT-4o vision input (used only in IMAGE_MODE='vlm').

    - If image_path is an HTTP/HTTPS URL: return as-is.
    - If image_path is a local file path: base64-encode and return as data URI.

    Paper Section 3.1: images are passed as multimodal affordance representations
    to the VLM; this function handles both deployment and development scenarios.
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = f"image/{ext}" if ext in {"jpg", "jpeg", "png", "gif", "webp"} else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _load_memory_records(history_k: int) -> list:
    """
    Load the last history_k memory records from memory_1.json for context.
    Returns an empty list if the file does not exist or is malformed.
    """
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[-history_k:]
        return [data]
    except (json.JSONDecodeError, OSError):
        return []


def _empty_memory_record(task_id: str, task: str) -> dict:
    """
    Return the canonical forward-compatible memory record schema.
    Null placeholders for fields written by Modules II–IV.

    Schema designed so downstream modules extend the same JSON record
    rather than creating separate files (Paper Section 3, pipeline design).
    """
    return {
        "task_id": task_id,
        "task": task,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target_object": {"name": "", "position": "", "color": "", "cx": 0, "cy": 0},
        "selected_tool": {"name": "", "position": "", "color": "", "cx": 0, "cy": 0},
        "task_steps": [],
        "raw_outputs": {
            "step1": "",
            "step2": "",
            "step3": "",
            "step4": ""
        },
        # Null placeholders for Modules II–IV
        "visual_prompted_image": None,
        "keypoint_coordinates": None,
        "motion_force_primitive": None,
        "execution_status": None
    }


def _save_memory(record: dict) -> None:
    """Persist memory record to memory/memory_1.json (append or overwrite)."""
    existing = _load_memory_records(history_k=1000)
    # Replace record with same task_id if present, else append
    updated = [r for r in existing if r.get("task_id") != record["task_id"]]
    updated.append(record)
    with open(MEMORY_FILE, "w") as f:
        json.dump(updated, f, indent=2)


# ---------------------------------------------------------------------------
# Core VLM dialog runner  (Paper Section 3.2 — sequential prompting chain)
# ---------------------------------------------------------------------------

def run_vlm_dialog(
    agent_idx: int,
    steps: list,
    history_k: int = 3,
    model: str = "gpt-4o",
    prompt_bundle_name: str = "P1"
) -> dict:
    """
    Run a multi-step VLM dialog for Module I task planning.

    Loads the named prompt bundle, injects prior memory records as context,
    then executes each step sequentially — passing each step's output as
    input context to the next step (conversation chaining).

    Args:
        agent_idx:          Index of this agent (1 for Task Planning).
        steps:              List of step dicts, each containing at minimum
                            'name' and 'content' (text or image payload).
        history_k:          Number of prior memory records to include as context.
        model:              OpenAI model identifier.
        prompt_bundle_name: Filename stem under prompts/ (e.g. "P1").

    Returns:
        dict mapping step name -> raw string output.

    Paper Section 3.2: the four prompting steps form a sequential chain
    where each step's output informs the next.
    """
    bundle_path = Path("prompts") / f"{prompt_bundle_name}.json"
    try:
        with open(bundle_path, "r") as f:
            bundle = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load prompt bundle {bundle_path}: {e}")

    step_keys = list(bundle["steps"].keys())
    client = OpenAI(api_key=API_KEY)

    conversation_history: list = []
    results: dict = {}
    prev_output: str = ""

    # Inject recent memory as system context
    prior_records = _load_memory_records(history_k)
    if prior_records:
        context_summary = json.dumps(prior_records, indent=2)
        conversation_history.append({
            "role": "system",
            "content": f"Prior task memory (last {len(prior_records)} records):\n{context_summary}"
        })

    for i, step in enumerate(steps):
        step_key = step_keys[i] if i < len(step_keys) else f"step{i+1}"
        prompt_cfg = bundle["steps"].get(step_key, {})

        role_text = prompt_cfg.get("role", "You are a helpful robot assistant.")
        instruction = prompt_cfg.get("instruction", "")
        output_format = prompt_cfg.get("output_format", "")

        # Build user message content
        user_content: list = []

        # Carry forward previous step output as context
        if prev_output:
            user_content.append({
                "type": "text",
                "text": f"Previous step output:\n{prev_output}\n\n"
            })

        # Add image or text payload
        payload = step.get("content")
        if step.get("is_image") and payload:
            encoded = encode_image(payload)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": encoded}
            })
        else:
            if payload:
                user_content.append({
                    "type": "text",
                    "text": str(payload)
                })

        # Append the formatted instruction
        full_instruction = instruction
        if output_format:
            full_instruction += f"\n\nRespond ONLY in this format: {output_format}"
        user_content.append({"type": "text", "text": full_instruction})

        # Build messages for this step
        messages = [{"role": "system", "content": role_text}]
        messages += conversation_history
        messages.append({"role": "user", "content": user_content})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            output = f"[ERROR: {e}]"

        results[step_key] = output
        prev_output = output

        # Add to conversation history for next step
        conversation_history.append({"role": "user", "content": user_content})
        conversation_history.append({"role": "assistant", "content": output})

    return results


# ---------------------------------------------------------------------------
# Output parser  (Paper Section 3.2 — structured output extraction)
# ---------------------------------------------------------------------------

def _parse_coords(raw: str) -> tuple:
    """
    Extract center pixel coordinates from a VLM output string.
    Handles formats like '(cx:234, cy:156)' or '(234, 156)'.
    Returns (cx, cy) as ints, defaults to (0, 0) if not found.
    """
    m = re.search(r'cx\s*:\s*(\d+).*?cy\s*:\s*(\d+)', raw, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', raw)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def _parse_three_fields(raw: str, fallback_name: str = "Unknown") -> tuple:
    """
    Parse 'Name, Position, Color, (cx:X, cy:Y)' from a raw VLM output string.
    Returns (name, position, color, cx, cy) with graceful fallbacks.
    """
    cx, cy = _parse_coords(raw)
    # Strip coordinate block before splitting on commas
    clean = re.sub(r'\(.*?\)', '', raw).strip().rstrip(',').strip()
    parts = [p.strip() for p in clean.split(",", 2)]
    name = parts[0] if len(parts) > 0 and parts[0] else fallback_name
    position = parts[1] if len(parts) > 1 and parts[1] else "unknown position"
    color = parts[2] if len(parts) > 2 and parts[2] else "unknown color"
    return name, position, color, cx, cy


def _parse_task_steps(raw: str) -> list:
    """
    Parse step descriptions from step4 raw output into TaskStep list.

    Expected format per step:
      Step1: <entity, action, object, success_criterion>
    """
    steps = []
    # Match patterns like "Step1: ..." or "Step 1: ..."
    pattern = re.compile(r"Step\s*(\d+)\s*:\s*(.+?)(?=Step\s*\d+\s*:|$)", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(raw)

    if not matches:
        # Fallback: split on semicolons
        raw_steps = [s.strip() for s in raw.split(";") if s.strip()]
        for idx, text in enumerate(raw_steps, start=1):
            parts = [p.strip() for p in text.split(",", 3)]
            steps.append(TaskStep(
                step_index=idx,
                entity=parts[0] if len(parts) > 0 else "robot",
                action=parts[1] if len(parts) > 1 else text,
                target_object=parts[2] if len(parts) > 2 else "target",
                success_criterion=parts[3] if len(parts) > 3 else "step completed"
            ))
        return steps

    for idx_str, content in matches:
        content = content.strip().rstrip(";").strip()
        # Remove angle brackets if present
        content = content.strip("<>")
        parts = [p.strip() for p in content.split(",", 3)]
        steps.append(TaskStep(
            step_index=int(idx_str),
            entity=parts[0] if len(parts) > 0 else "robot",
            action=parts[1] if len(parts) > 1 else content,
            target_object=parts[2] if len(parts) > 2 else "target",
            success_criterion=parts[3] if len(parts) > 3 else "step completed"
        ))

    return steps


def parse_task_plan(results: dict, task: str) -> TaskPlan:
    """
    Parse raw VLM string outputs from all 4 dialog steps into a typed TaskPlan.

    Applies graceful fallbacks for all fields so downstream modules always
    receive a valid TaskPlan object even on partial VLM failures.

    Args:
        results: Dict mapping step key -> raw string output from run_vlm_dialog.
        task:    Task description string.

    Returns:
        Populated TaskPlan dataclass.

    Paper Section 3.2: parsed outputs feed directly into the memory schema
    and downstream modules.
    """
    task_id = str(uuid.uuid4())[:8]

    # Step 1 — target object
    raw1 = results.get("step1_task_understanding", "")
    obj_name, obj_pos, obj_color, obj_cx, obj_cy = _parse_three_fields(raw1, "target object")
    target_object = ObjectProperties(name=obj_name, position=obj_pos, color=obj_color, cx=obj_cx, cy=obj_cy)

    # Step 3 — selected tool (step 2 gives tool list with coords; step 3 selects one)
    raw3 = results.get("step3_tool_selection", "")
    # Strip any leading explanation (take last line or first comma-separated block)
    raw3_clean = raw3.strip().split("\n")[-1].strip()
    tool_name, tool_pos, tool_color, tool_cx, tool_cy = _parse_three_fields(raw3_clean, "selected tool")
    selected_tool = ToolProperties(name=tool_name, position=tool_pos, color=tool_color, cx=tool_cx, cy=tool_cy)

    # Step 4 — task steps
    raw4 = results.get("step4_task_planning", "")
    steps = _parse_task_steps(raw4)

    return TaskPlan(
        task_id=task_id,
        task=task,
        target_object=target_object,
        selected_tool=selected_tool,
        steps=steps
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_memory_record(plan: TaskPlan, raw_outputs: dict) -> None:
    """
    Save a fully populated memory record to memory/memory_1.json.

    The schema includes null placeholders for fields written by Modules II–IV
    so all modules share a single evolving record per task run.

    Paper Section 3: shared memory enables cross-module information flow.
    """
    record = _empty_memory_record(plan.task_id, plan.task)
    record["target_object"] = {
        "name": plan.target_object.name,
        "position": plan.target_object.position,
        "color": plan.target_object.color,
        "cx": plan.target_object.cx,
        "cy": plan.target_object.cy
    }
    record["selected_tool"] = {
        "name": plan.selected_tool.name,
        "position": plan.selected_tool.position,
        "color": plan.selected_tool.color,
        "cx": plan.selected_tool.cx,
        "cy": plan.selected_tool.cy
    }
    record["task_steps"] = [
        {
            "step_index": s.step_index,
            "entity": s.entity,
            "action": s.action,
            "target_object": s.target_object,
            "success_criterion": s.success_criterion
        }
        for s in plan.steps
    ]
    record["raw_outputs"]["step1"] = raw_outputs.get("step1_task_understanding", "")
    record["raw_outputs"]["step2"] = raw_outputs.get("step2_tool_identification", "")
    record["raw_outputs"]["step3"] = raw_outputs.get("step3_tool_selection", "")
    record["raw_outputs"]["step4"] = raw_outputs.get("step4_task_planning", "")
    _save_memory(record)


def save_task_plan(plan: TaskPlan) -> None:
    """
    Save the parsed TaskPlan to results/task_<task_id>.json.

    Paper Section 3: results directory stores per-run artifacts for
    post-hoc analysis and evaluation.
    """
    out_path = RESULTS_DIR / f"task_{plan.task_id}.json"
    data = {
        "task_id": plan.task_id,
        "task": plan.task,
        "target_object": {
            "name": plan.target_object.name,
            "position": plan.target_object.position,
            "color": plan.target_object.color,
            "cx": plan.target_object.cx,
            "cy": plan.target_object.cy
        },
        "selected_tool": {
            "name": plan.selected_tool.name,
            "position": plan.selected_tool.position,
            "color": plan.selected_tool.color,
            "cx": plan.selected_tool.cx,
            "cy": plan.selected_tool.cy
        },
        "steps": [
            {
                "step_index": s.step_index,
                "entity": s.entity,
                "action": s.action,
                "target_object": s.target_object,
                "success_criterion": s.success_criterion
            }
            for s in plan.steps
        ]
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
