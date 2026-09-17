"""
Module I for a single-camera rig — task understanding, tool identification,
tool selection, task planning.

Same four steps as the paper (Section 3.2) and as agents/VLM1_TaskPlanning.py.
Two differences, both forced by the hardware:

  1. No scene camera. Step 1 reads the target object from the operator's text
     description instead of a scene image.
  2. The tool board image is a live capture bundle rather than a fixed file.

And one difference that is a fix rather than a constraint: prompt placeholders
are substituted. The P1 path builds its instruction with no .format() call
(utils.py:219), so GPT-4o receives the literal strings "{task}" and
"{tool_list}".

No pixel coordinates are produced here. Per Eq. (2) the position field is a
description, used as a detection prompt; locating the tool is Module II's job.

Writes memory/memory_1.json in the schema Module II reads, so the existing
Module II can consume this output unchanged.

Run:
    python -m agents.module1 "hammering a nail"
    python -m agents.module1                    # prompts for the task

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera import load_capture
from structures.task_structures import TaskStep
from structures.tool_structures import (
    ToolCandidate, ModuleOneResult, ToolSelectionError,
)
from utils import _empty_memory_record, _save_memory

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

PROMPT_BUNDLE = Path("prompts") / "P1_tool.json"
RESULTS_DIR = Path("results")
DEFAULT_CAPTURE = Path("captures") / "latest"
MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# VLM plumbing
# ---------------------------------------------------------------------------

def _encode(image_path: Path) -> str:
    import base64
    data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _build_step(cfg: dict, **subs) -> tuple:
    """
    Return (role, instruction) with placeholders filled.

    Uses str.format, so a bundle referencing a key the caller did not supply
    raises here instead of silently shipping "{task}" to the model.
    """
    role = cfg.get("role", "You are a helpful robot assistant.")
    instruction = cfg.get("instruction", "").format(**subs)
    fmt = cfg.get("output_format", "")
    if fmt:
        instruction += f"\n\nRespond ONLY in this format:\n{fmt}"
    return role, instruction


def _call(client: OpenAI, role: str, instruction: str,
          image_path: Optional[Path] = None,
          history: Optional[list] = None) -> str:
    content: list = []
    if image_path is not None:
        content.append({"type": "image_url",
                        "image_url": {"url": _encode(image_path)}})
    content.append({"type": "text", "text": instruction})

    messages = [{"role": "system", "content": role}]
    if history:
        messages += history
    messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _field(raw: str, key: str, default: str = "") -> str:
    """Pull a 'KEY: value' line out of a structured VLM reply."""
    m = re.search(rf"^{key}\s*:\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else default


_TOOL_LINE = re.compile(
    r"(Tool\s*\d+)\s*:\s*(?P<name>[^|\n]+?)\s*\|\s*"
    r"(?P<colour>[^|\n]+?)\s*\|\s*(?P<position>[^\n]+)",
    re.IGNORECASE,
)


def _parse_candidates(raw: str) -> List[ToolCandidate]:
    return [
        ToolCandidate(
            label=re.sub(r"\s+", "", m.group(1)),
            name=m.group("name").strip(),
            colour=m.group("colour").strip(),
            position=m.group("position").strip(),
        )
        for m in _TOOL_LINE.finditer(raw)
    ]


def _parse_selection(raw: str, candidates: List[ToolCandidate]) -> tuple:
    """
    Resolve the selection to one of the candidates.

    Looked up by LABEL, taking the tool's details from the identification step
    rather than from the selection reply — the model restates them, and a
    restatement is a chance to drift from what it saw in the image.
    """
    m = re.search(r"SELECTED\s*:\s*(Tool\s*\d+)", raw, re.IGNORECASE)
    label = re.sub(r"\s+", "", m.group(1)) if m else ""
    chosen = next((c for c in candidates if c.label.lower() == label.lower()), None)

    if chosen is None:
        # A model that names the right tool but mangles the label still chose it.
        want = _field(raw, "NAME").lower()
        if want:
            chosen = next((c for c in candidates
                           if c.name.lower() in want or want in c.name.lower()), None)

    if chosen is None:
        raise ToolSelectionError(
            f"Could not resolve the selected tool.\n"
            f"  Parsed label : {label!r}\n"
            f"  Candidates   : {[c.label + '=' + c.name for c in candidates]}\n"
            f"  Raw response : {raw[:400]}"
        )
    return chosen, _field(raw, "REASON")


_STEP_LINE = re.compile(r"Step\s*(\d+)\s*:\s*(.+)", re.IGNORECASE)


def _parse_steps(raw: str) -> List[TaskStep]:
    """
    Parse 'Step<N>: entity, action, object, criterion'.

    Split into at most 4 parts so a success criterion containing commas — which
    the more useful ones do — survives intact in the final field.
    """
    steps: List[TaskStep] = []
    for m in _STEP_LINE.finditer(raw):
        parts = [p.strip() for p in m.group(2).strip().strip("<>").split(",", 3)]
        steps.append(TaskStep(
            step_index=int(m.group(1)),
            entity=parts[0] if len(parts) > 0 else "robot",
            action=parts[1] if len(parts) > 1 else m.group(2),
            target_object=parts[2] if len(parts) > 2 else "",
            success_criterion=parts[3] if len(parts) > 3 else "",
        ))
    return steps


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _write_memory(result: ModuleOneResult) -> None:
    """
    Persist in the schema Module II reads, so it can consume this unchanged.

    Module II needs task_id, task, selected_tool.name and target_object.name;
    the tool name becomes the LangSAM detection prompt.
    """
    record = _empty_memory_record(result.task_id, result.task)
    record["target_object"] = {
        "name": result.target_name,
        "position": result.target_position,
        "color": result.target_colour,
        "cx": 0, "cy": 0,          # Module II supplies real coordinates
    }
    record["selected_tool"] = {
        "name": result.selected.name if result.selected else "",
        "position": result.selected.position if result.selected else "",
        "color": result.selected.colour if result.selected else "",
        "cx": 0, "cy": 0,
    }
    record["task_steps"] = [
        {
            "step_index": s.step_index,
            "entity": s.entity,
            "action": s.action,
            "target_object": s.target_object,
            "success_criterion": s.success_criterion,
        }
        for s in result.steps
    ]
    for k in ("step1", "step2", "step3", "step4"):
        record["raw_outputs"][k] = result.raw_outputs.get(k, "")
    record["capture_dir"] = result.capture_dir
    _save_memory(record)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_module1(task: str,
                capture_dir: Path = DEFAULT_CAPTURE,
                verbose: bool = True) -> ModuleOneResult:
    """Run all four Module I steps and persist the result for Module II."""
    if not API_KEY:
        raise ToolSelectionError("OPENAI_API_KEY is not set (.env).")

    capture_dir = Path(capture_dir)
    colour_path = capture_dir / "color.png"
    if not colour_path.exists():
        raise ToolSelectionError(
            f"No capture bundle at {capture_dir}. Run: python camera.py"
        )

    cap = load_capture(capture_dir)
    h, w = cap.color.shape[:2]

    with open(PROMPT_BUNDLE) as f:
        bundle = json.load(f)["steps"]

    client = OpenAI(api_key=API_KEY)
    task_id = str(uuid.uuid4())[:8]
    history: list = []

    def remember(instruction: str, answer: str) -> None:
        history.append({"role": "user", "content": [{"type": "text", "text": instruction}]})
        history.append({"role": "assistant", "content": answer})

    # --- step 1: what is being worked on -----------------------------------
    role, instr = _build_step(bundle["step1_task_understanding"], task=task)
    if verbose:
        print(f"[Module I] 1/4 Task understanding — {task!r}")
    raw1 = _call(client, role, instr)
    remember(instr, raw1)
    target_name = _field(raw1, "NAME", "unknown target")
    target_pos = _field(raw1, "POSITION")
    target_col = _field(raw1, "COLOUR") or _field(raw1, "COLOR")
    if verbose:
        print(f"           target: {target_name} | {target_pos} | {target_col}")

    # --- step 2: what is on the board --------------------------------------
    role, instr = _build_step(bundle["step2_tool_identification"], width=w, height=h)
    if verbose:
        print(f"[Module I] 2/4 Tool identification — {colour_path} ({w}x{h})")
    raw2 = _call(client, role, instr, image_path=colour_path)
    remember(instr, raw2)
    candidates = _parse_candidates(raw2)
    if not candidates:
        raise ToolSelectionError(f"No tools parsed.\nRaw:\n{raw2[:600]}")
    if verbose:
        for c in candidates:
            print(f"           {c.label}: {c.name} ({c.colour}) — {c.position}")

    # --- step 3: which one for this task -----------------------------------
    tool_list = "\n".join(f"{c.label}: {c.name} | {c.colour} | {c.position}"
                          for c in candidates)
    role, instr = _build_step(bundle["step3_tool_selection"],
                              task=task, target=target_name, tool_list=tool_list)
    if verbose:
        print("[Module I] 3/4 Tool selection")
    raw3 = _call(client, role, instr, history=history)
    remember(instr, raw3)
    chosen, reason = _parse_selection(raw3, candidates)
    if verbose:
        print(f"           selected: {chosen.label} — {chosen.name}")

    # --- step 4: how to do it ----------------------------------------------
    role, instr = _build_step(bundle["step4_task_planning"],
                              task=task, target=target_name, tool=chosen.name)
    if verbose:
        print("[Module I] 4/4 Task planning")
    raw4 = _call(client, role, instr, history=history)
    steps = _parse_steps(raw4)
    if not steps:
        raise ToolSelectionError(f"No task steps parsed.\nRaw:\n{raw4[:600]}")

    result = ModuleOneResult(
        task_id=task_id, task=task, capture_dir=str(capture_dir),
        target_name=target_name, target_position=target_pos,
        target_colour=target_col, candidates=candidates, selected=chosen,
        reason=reason, steps=steps,
        raw_outputs={"step1": raw1, "step2": raw2, "step3": raw3, "step4": raw4},
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"module1_{task_id}.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    _write_memory(result)

    if verbose:
        print(f"\n=== MODULE I OUTPUT ({task_id}) ===")
        print(f"Task    : {task}")
        print(f"Target  : {target_name} | {target_pos} | {target_col}")
        print(f"Tool    : {chosen.name} | {chosen.colour} | {chosen.position}")
        print(f"Reason  : {reason}")
        print("\nPlan:")
        for s in steps:
            print(f"  Step {s.step_index}: [{s.entity}] {s.action} -> {s.target_object}")
            print(f"          check: {s.success_criterion}")
        print(f"\n[Saved]  {out_path}")
        print(f"[Memory] memory/memory_1.json  (Module II reads this)")

    return result


def main() -> int:
    task = " ".join(sys.argv[1:]).strip() or input("Task: ").strip()
    if not task:
        print("No task given.")
        return 1
    try:
        run_module1(task)
    except ToolSelectionError as exc:
        print(f"[Module I] ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
