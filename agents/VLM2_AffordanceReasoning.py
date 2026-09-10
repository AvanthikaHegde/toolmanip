"""
VLM2 Affordance Reasoning Agent — Module II of the ToolManip framework.

Implements Paper Section 3.3: Affordance Reasoning.
Single execution path — no IMAGE_MODE routing:
  Load memory → HRE → SIF → save dotted images → display images →
  VLM call with real images → parse keypoints → resolve coordinates →
  verify → save → return

Reads Module I output from memory/memory_1.json.
Writes keypoint_coordinates and visual_prompted_image back to the same record.

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).
"""

import base64
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import cv2

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

from structures.affordance_structures import (
    AffordanceResult, AffordanceCoordinates, AffordanceResolutionError,
    KeypointResult, PixelCoordinate
)

# ---------------------------------------------------------------------------
# Image constants
# ---------------------------------------------------------------------------

TOOL_IMAGE  = "examples/tool_board_clean.png"
SCENE_IMAGE = "examples/putty_scene.png"

# Shortest O→Q the resolver will accept, as a fraction of the target image's
# smaller side. Below this the operation vector describes a span so small that
# operation_distance_px and operation_vector are dominated by keypoint
# quantisation rather than by the VLM's intent.
MIN_OPERATION_SPAN_FRACTION = 0.10

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

MEMORY_FILE = Path("memory") / "memory_1.json"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _load_latest_memory() -> dict:
    """Load the most recent task record from memory/memory_1.json."""
    if not MEMORY_FILE.exists():
        warnings.warn("[Module II] memory/memory_1.json not found. Run Module I first.", RuntimeWarning)
        return {}
    try:
        with open(MEMORY_FILE, "r") as f:
            records = json.load(f)
        if isinstance(records, list) and records:
            return records[-1]
        if isinstance(records, dict):
            return records
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f"[Module II] Failed to load memory: {exc}", RuntimeWarning)
    return {}


def _update_memory(task_id: str, keypoints: KeypointResult,
                   coords: AffordanceCoordinates,
                   dotted_tool_path: str, dotted_target_path: str) -> None:
    """Write Module II results back into the existing memory record."""
    if not MEMORY_FILE.exists():
        return
    try:
        with open(MEMORY_FILE, "r") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    if not isinstance(records, list):
        records = [records]

    for record in records:
        if record.get("task_id") == task_id:
            record["keypoint_coordinates"] = {
                "grasp_point": keypoints.grasp_point,
                "functional_point": keypoints.functional_point,
                "tool_direction": keypoints.tool_direction,
                "start_point": keypoints.start_point,
                "end_point": keypoints.end_point,
                "target_direction": keypoints.target_direction
            }
            record["resolved_coordinates"] = {
                "grasp_H": coords.grasp_point_H.to_dict(),
                "functional_F": coords.functional_point_F.to_dict(),
                "tool_direction_angle": coords.tool_direction_angle,
                "start_O": coords.start_point_O.to_dict(),
                "end_Q": coords.end_point_Q.to_dict(),
                "target_direction_angle": coords.target_direction_angle,
                "operation_distance_px": coords.operation_distance_px,
                "operation_vector": coords.operation_vector
            }
            record["visual_prompted_image"] = {
                "tool": dotted_tool_path,
                "target": dotted_target_path
            }
            break

    with open(MEMORY_FILE, "w") as f:
        json.dump(records, f, indent=2)


# ---------------------------------------------------------------------------
# VLM output parsing
# ---------------------------------------------------------------------------

def _parse_keypoint_output(raw: str) -> KeypointResult:
    """
    Parse the VLM's affordance selection string into a KeypointResult.

    Labels the VLM did not provide come back as None. Earlier versions
    substituted arbitrary defaults ("1", "2", "τ0", …), which made a failed
    or malformed VLM response indistinguishable from a real selection.
    """
    def _find(patterns: list, text: str):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(";").strip()
        return None

    grasp = _find([
        r"keypoint\s*h\s*[:\-]\s*([^\s;,]+)",
        r"h\s*[:\-]\s*([^\s;,]+)"
    ], raw)

    functional = _find([
        r"keypoint\s*f\s*[:\-]\s*([^\s;,]+)",
        r"f\s*[:\-]\s*([^\s;,]+)"
    ], raw)

    tool_dir = _find([
        r"vector\s*[τt]t?\s*[:\-]\s*([^\s;,]+)",
        r"[τt]t\s*[:\-]\s*([^\s;,]+)",
        r"tool\s*direction\s*[:\-]\s*([^\s;,]+)"
    ], raw)

    start = _find([
        r"keypoint\s*o\s*[:\-]\s*([^\s;,]+)",
        r"\bo\s*[:\-]\s*([^\s;,]+)"
    ], raw)

    end = _find([
        r"keypoint\s*q\s*[:\-]\s*([^\s;,]+)",
        r"\bq\s*[:\-]\s*([^\s;,]+)"
    ], raw)

    target_dir = _find([
        r"vector\s*[τt]o?\s*[:\-]\s*([^\s;,]+)",
        r"[τt]o\s*[:\-]\s*([^\s;,]+)",
        r"target\s*direction\s*[:\-]\s*([^\s;,]+)"
    ], raw)

    return KeypointResult(
        grasp_point=grasp,
        functional_point=functional,
        tool_direction=tool_dir,
        start_point=start,
        end_point=end,
        target_direction=target_dir
    )


# ---------------------------------------------------------------------------
# Image encoding for VLM call
# ---------------------------------------------------------------------------

def _encode_image_b64(image_path: str) -> str:
    """Encode a local image file as a base64 data URI for GPT-4o Vision."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = f"image/{ext}" if ext in {"jpg", "jpeg", "png", "gif", "webp"} else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def _call_vlm_with_images(prompt_text: str, tool_image_path: str,
                           target_image_path: str, model: str = "gpt-4o") -> str:
    """Send affordance prompt + two annotated images to GPT-4o."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    tool_uri   = _encode_image_b64(tool_image_path)
    target_uri = _encode_image_b64(target_image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": tool_uri}},
                {"type": "image_url", "image_url": {"url": target_uri}}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
    except Exception as exc:
        # Returning the error as the "response" text used to let an API failure
        # flow into the parser, where it surfaced later as six missing labels
        # instead of the one real cause.
        raise RuntimeError(f"[Module II] VLM call failed: {exc}") from exc
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(task: str, task_steps: list, tool_name: str,
                  target_name: str, n_tool_keypoints: int,
                  n_target_keypoints: int, n_directions: int) -> str:
    """
    Build the formatted affordance reasoning prompt from P2.json.

    The candidate counts are stated explicitly in the prompt. Without them the
    VLM has to infer the label range from the rendered dots and routinely
    answers out of range (e.g. "H: 14" when only 10 candidates exist).
    """
    bundle_path = Path("prompts") / "P2.json"
    try:
        with open(bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        step_cfg    = bundle["steps"]["step1_affordance_reasoning"]
        role        = step_cfg.get("role", "")
        instruction = step_cfg.get("instruction", "")
        output_format = step_cfg.get("output_format", "")
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        # Quietly swapping in a stub prompt changes what the experiment measures,
        # so a missing or malformed bundle is fatal instead.
        raise RuntimeError(
            f"[Module II] Could not load the affordance prompt bundle {bundle_path}: {exc}"
        ) from exc

    step_summary = "; ".join(
        f"Step{s.get('step_index', i+1)}: {s.get('action', '')} {s.get('target_object', '')}"
        for i, s in enumerate(task_steps)
    ) if task_steps else task

    filled = instruction.format(
        task=task,
        task_step=step_summary,
        tool=tool_name,
        object=target_name
    )

    ranges = (
        f"Valid labels for this image pair:\n"
        f"- Tool image (Image 1) keypoints H and F: integers 1 to {n_tool_keypoints}\n"
        f"- Target image (Image 2) keypoints O and Q: integers 1 to {n_target_keypoints}\n"
        f"- Direction vectors τt and τo: τ0 to τ{n_directions - 1}\n"
        f"O and Q must be different keypoints.\n"
        f"Do not output any label outside these ranges."
    )

    return f"{role}\n\n{filled}\n\n{ranges}\n\nRespond ONLY in this format: {output_format}"


# ---------------------------------------------------------------------------
# Coordinate resolver
# ---------------------------------------------------------------------------

def resolve_to_coordinates(
    keypoint_result: KeypointResult,
    tool_keypoints: list,
    target_keypoints: list,
    direction_angles: list,
    tool_image_size: tuple,
    target_image_size: tuple
) -> AffordanceCoordinates:
    """
    Resolve VLM label outputs to pixel + normalized coordinates.
    Label "4" means index 3 (1-based) in the keypoints list.
    This bridges symbolic VLM output to spatial robot coordinates.

    Raises AffordanceResolutionError if any label is missing, malformed, or
    outside the candidate set SIF actually generated, listing every problem
    at once. Nothing is substituted — an unresolvable selection is a failed
    run, not a run with placeholder coordinates.
    """
    problems: list = []

    def label_to_pixel(label, kp_list, img_size, field):
        w, h = img_size
        if label is None:
            problems.append(f"{field}: absent from the VLM response")
            return None
        try:
            idx = int(str(label).strip()) - 1
        except (ValueError, TypeError):
            problems.append(f"{field}: {label!r} is not a numeric keypoint label")
            return None
        if not 0 <= idx < len(kp_list):
            problems.append(
                f"{field}: label {label!r} is outside the candidate range 1..{len(kp_list)}"
            )
            return None
        u, v = int(kp_list[idx][0]), int(kp_list[idx][1])
        return PixelCoordinate(
            u=u, v=v,
            u_norm=round(u / w, 4),
            v_norm=round(v / h, 4),
            image_width=w,
            image_height=h
        )

    def tau_to_degrees(label, angles, field):
        if label is None:
            problems.append(f"{field}: absent from the VLM response")
            return None
        m = re.fullmatch(r"[τt]\s*(\d+)", str(label).strip(), re.IGNORECASE)
        if not m:
            problems.append(f"{field}: {label!r} is not a direction label (expected τN)")
            return None
        idx = int(m.group(1))
        if not 0 <= idx < len(angles):
            problems.append(
                f"{field}: {label!r} is outside the direction range τ0..τ{len(angles) - 1}"
            )
            return None
        return round(math.degrees(angles[idx]) % 360, 1)

    H  = label_to_pixel(keypoint_result.grasp_point,      tool_keypoints,   tool_image_size,   "H  (tool grasp point)")
    F  = label_to_pixel(keypoint_result.functional_point, tool_keypoints,   tool_image_size,   "F  (tool functional point)")
    O  = label_to_pixel(keypoint_result.start_point,      target_keypoints, target_image_size, "O  (target start point)")
    Q  = label_to_pixel(keypoint_result.end_point,        target_keypoints, target_image_size, "Q  (target end point)")
    tt = tau_to_degrees(keypoint_result.tool_direction,    direction_angles, "τt (tool direction)")
    to_ = tau_to_degrees(keypoint_result.target_direction, direction_angles, "τo (target direction)")

    # --- Geometric sanity of the operation vector -------------------------
    #
    # A label-valid selection can still be physically meaningless. Run
    # d120178b chose O=11, Q=13 — two adjacent perimeter keypoints 22.5px
    # apart in a 512px crop, half of them on the plate rather than the putty.
    # Every label was in range, so nothing objected, and a complete-looking
    # affordance JSON was written describing a 22px "scrape".
    #
    # Both checks below can be waived with ALLOW_DEGENERATE_AFFORDANCE=1, in
    # the same spirit as ALLOW_HRE_FALLBACK. The legitimate case is a
    # point-like operation — hammering a nail, seating a bolt — where O and Q
    # coincide by design and τo alone carries the direction.
    _waived = os.environ.get(
        "ALLOW_DEGENERATE_AFFORDANCE", "").strip().lower() in {"1", "true", "yes"}

    if O is not None and Q is not None and not _waived:
        span = math.dist((O.u, O.v), (Q.u, Q.v))
        floor = MIN_OPERATION_SPAN_FRACTION * min(*target_image_size)
        if span == 0:
            problems.append(
                f"O and Q both resolve to pixel ({O.u}, {O.v}) — the operation "
                f"vector is undefined with zero length"
            )
        elif span < floor:
            problems.append(
                f"O ({O.u}, {O.v}) and Q ({Q.u}, {Q.v}) are only {span:.1f}px "
                f"apart, under the {floor:.1f}px floor "
                f"({MIN_OPERATION_SPAN_FRACTION:.0%} of the "
                f"{min(*target_image_size)}px image side). The operation vector "
                f"spans almost none of the target, so any distance derived from "
                f"it is noise. Set ALLOW_DEGENERATE_AFFORDANCE=1 if this task "
                f"really is a point operation."
            )
        elif to_ is not None:
            # τo is the motion direction along the target surface, so it should
            # agree with the direction O->Q actually points. The VLM picks from
            # `direction_angles`, and the nearest of those to any true heading is
            # at most (360/n)/2 away — so a deviation beyond a full 360/n step
            # means it chose a direction its own two points contradict.
            span_deg = math.degrees(math.atan2(Q.v - O.v, Q.u - O.u)) % 360
            step = 360.0 / len(direction_angles)
            delta = abs((span_deg - to_ + 180) % 360 - 180)
            if delta > step:
                problems.append(
                    f"τo is {to_:.1f}° but O→Q points {span_deg:.1f}° "
                    f"({delta:.1f}° apart, over the {step:.1f}° tolerance). The "
                    f"chosen direction disagrees with the two points chosen to "
                    f"define it; one of the three is wrong."
                )

    if problems:
        raise AffordanceResolutionError(
            "Could not resolve the VLM's affordance selection to coordinates:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + f"\n\nCandidates available this run: "
            f"tool keypoints 1..{len(tool_keypoints)}, "
            f"target keypoints 1..{len(target_keypoints)}, "
            f"directions τ0..τ{len(direction_angles) - 1}."
        )

    dx = Q.u - O.u
    dy = Q.v - O.v
    dist = math.sqrt(dx ** 2 + dy ** 2)
    unit = [round(dx / dist, 4), round(dy / dist, 4)]

    return AffordanceCoordinates(
        grasp_point_H=H,
        functional_point_F=F,
        tool_direction_angle=tt,
        start_point_O=O,
        end_point_Q=Q,
        target_direction_angle=to_,
        operation_distance_px=round(dist, 2),
        operation_vector=unit
    )


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_affordance_result(result: AffordanceResult) -> None:
    """Save the AffordanceResult to results/affordance_{task_id}.json."""
    out_path = RESULTS_DIR / f"affordance_{result.task_id}.json"
    coords = result.coordinates
    data = {
        "task_id": result.task_id,
        "task": result.task,
        "tool_name": result.tool_name,
        "target_name": result.target_name,
        "keypoints": {
            "grasp_point": result.keypoints.grasp_point,
            "functional_point": result.keypoints.functional_point,
            "tool_direction": result.keypoints.tool_direction,
            "start_point": result.keypoints.start_point,
            "end_point": result.keypoints.end_point,
            "target_direction": result.keypoints.target_direction
        },
        "coordinates": {
            "grasp_H": coords.grasp_point_H.to_dict(),
            "functional_F": coords.functional_point_F.to_dict(),
            "tool_direction_angle": coords.tool_direction_angle,
            "start_O": coords.start_point_O.to_dict(),
            "end_Q": coords.end_point_Q.to_dict(),
            "target_direction_angle": coords.target_direction_angle,
            "operation_distance_px": coords.operation_distance_px,
            "operation_vector": coords.operation_vector
        },
        "dotted_tool_image_path":    result.dotted_tool_image_path,
        "dotted_target_image_path":  result.dotted_target_image_path,
        "verified_tool_image_path":  result.verified_tool_image_path,
        "verified_target_image_path": result.verified_target_image_path,
        "verification_grid_path":    result.verification_grid_path
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# MOCK MODE — uncomment to test without API calls
# ---------------------------------------------------------------------------

# MOCK_KEYPOINTS = KeypointResult(
#     grasp_point="8",
#     functional_point="2",
#     tool_direction="τ2",
#     start_point="4",
#     end_point="13",
#     target_direction="τ4"
# )
#
# def _run_mock(task_id, task, tool_name, target_name):
#     print("[Module II] Running in MOCK mode — no API calls.")
#     from structures.affordance_structures import PixelCoordinate, AffordanceCoordinates
#     dummy_px = PixelCoordinate(u=256, v=256, u_norm=0.5, v_norm=0.5,
#                                image_width=512, image_height=512)
#     coords = AffordanceCoordinates(
#         grasp_point_H=dummy_px, functional_point_F=dummy_px,
#         tool_direction_angle=0.0,
#         start_point_O=dummy_px, end_point_Q=dummy_px,
#         target_direction_angle=0.0,
#         operation_distance_px=0.0, operation_vector=[1, 0]
#     )
#     result = AffordanceResult(
#         task_id=task_id, task=task, tool_name=tool_name,
#         target_name=target_name, keypoints=MOCK_KEYPOINTS,
#         coordinates=coords,
#         dotted_tool_image_path="", dotted_target_image_path="",
#         verified_tool_image_path="", verified_target_image_path="",
#         verification_grid_path=""
#     )
#     _save_affordance_result(result)
#     _update_memory(task_id, MOCK_KEYPOINTS, coords, "", "")
#     return result


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def run_affordance_reasoning() -> AffordanceResult:
    """
    Execute Module II: Affordance Reasoning.

    Full pipeline:
      1.  Load memory/memory_1.json
      2.  Check images exist → exit cleanly if not
      3.  HRE on SCENE_IMAGE → target_roi
      4.  HRE on TOOL_IMAGE  → tool_roi
      5.  extract_functional_region on tool_roi → functional_roi
      6.  SIF on functional_roi + target_roi → sif_result
      7.  Save dotted images to results/
      8.  Display dotted images (press any key to continue)
      9.  Build prompt from P2.json
      10. Call GPT-4o with both dotted images
      11. Parse raw output → KeypointResult
      12. Resolve labels → AffordanceCoordinates
      13. draw_selected_keypoints() → save verified images
      14. show_verification_grid() → print table + save grid
      15. pixel_inspector() on verified tool image
      16. Save AffordanceResult
      17. Update memory
      18. Return AffordanceResult
    """
    # 1. Load memory
    record      = _load_latest_memory()
    task_id     = record.get("task_id", "unknown")
    task        = record.get("task", "Unknown task")
    tool_name   = record.get("selected_tool", {}).get("name", "tool")
    target_name = record.get("target_object", {}).get("name", "target")
    task_steps  = record.get("task_steps", [])

    # The tool and target names become the LangSAM text prompts, so falling
    # back to the literal strings "tool" and "target" would segment nothing
    # useful while still producing a complete-looking result file.
    if not record:
        print(
            "\n[Module II] ERROR: no usable record in memory/memory_1.json.\n"
            "  Module II needs the tool and target names from Module I to drive\n"
            "  segmentation. Run Module I first (python main.py)."
        )
        sys.exit(1)
    if tool_name.startswith("[ERROR") or target_name.startswith("[ERROR"):
        print(
            f"\n[Module II] ERROR: the latest Module I record ({task_id}) stored a\n"
            f"  failed VLM response instead of a tool/target name:\n"
            f"    tool   = {tool_name!r}\n"
            f"    target = {target_name!r}\n"
            f"  Re-run Module I before continuing."
        )
        sys.exit(1)

    # 2. Startup image checks
    missing = [p for p in (TOOL_IMAGE, SCENE_IMAGE) if not Path(p).exists()]
    if missing:
        for p in missing:
            print(f"\n[Module II] ERROR: Image not found: '{p}'")
            print(f"  Please place the image at {p} and try again.")
        sys.exit(1)

    # 3-5. HRE
    from keypoint_gen.hre import extract_roi, extract_functional_region
    from keypoint_gen.sif import generate_sif

    print("[Module II] Running HRE on scene image...")
    target_roi = extract_roi(
        image_path=SCENE_IMAGE,
        description=target_name,
        expansion_factor=0.1
    )

    print("[Module II] Running HRE on tool image...")
    tool_roi = extract_roi(
        image_path=TOOL_IMAGE,
        description=tool_name,
        expansion_factor=0.1
    )

    print("[Module II] Extracting functional region from tool ROI...")
    functional_roi = extract_functional_region(
        cropped_image=tool_roi["cropped_image"],
        functional_description=f"functional region of {tool_name}",
        expansion_factor=0.1
    )
    tool_roi_for_sif = {
        "cropped_image": functional_roi["cropped_image"],
        "mask": functional_roi.get("mask"),
        "bbox": functional_roi.get("bbox")
    }

    # 5b. Refuse to continue on a silent HRE fallback.
    #
    # When LangSAM does not run, extract_roi returns the whole image with an
    # all-255 mask. SIF then treats the entire photo as the object: the
    # "skeleton" becomes the middle row of the image and the "contour" becomes
    # the image border, so the candidate keypoints describe the picture rather
    # than the tool. That is the paper's "ToolManip w/o HRE" ablation
    # (Table 3: 67.5 mm grasp error vs 6.7 mm), so it must be opted into
    # deliberately rather than reached by accident.
    fallbacks = [
        (name, roi.get("_reason", "unknown"))
        for name, roi in (
            ("target ROI (scene image)", target_roi),
            ("tool ROI (tool board)", tool_roi),
            ("tool functional region", functional_roi),
        )
        if roi.get("_fallback")
    ]
    if fallbacks:
        print("\n[Module II] HRE FALLBACK DETECTED — segmentation did not run:")
        for name, reason in fallbacks:
            print(f"  - {name}: {reason}")
        print(
            "\n  The full image is being used as the ROI, so the SIF candidate\n"
            "  keypoints will describe the image frame rather than the object.\n"
            "  Results are not comparable to the paper's method.\n"
        )
        if os.environ.get("ALLOW_HRE_FALLBACK", "").strip().lower() not in {"1", "true", "yes"}:
            print(
                "  Aborting. To run this configuration on purpose (the w/o HRE\n"
                "  ablation), set ALLOW_HRE_FALLBACK=1 in your environment."
            )
            sys.exit(1)
        print("  ALLOW_HRE_FALLBACK is set — continuing as the w/o HRE ablation.\n")

    # 6. SIF
    print("[Module II] Running SIF to generate keypoints and direction vectors...")
    sif_result = generate_sif(tool_roi_for_sif, target_roi)

    # 7. Save dotted images
    dotted_tool_path   = str(RESULTS_DIR / f"dotted_tool_{task_id}.jpg")
    dotted_target_path = str(RESULTS_DIR / f"dotted_target_{task_id}.jpg")
    cv2.imwrite(dotted_tool_path,   sif_result["tool_image"])
    cv2.imwrite(dotted_target_path, sif_result["target_image"])
    print(f"[Module II] Dotted images saved: {dotted_tool_path}, {dotted_target_path}")

    # 8. Display dotted images
    print("[Module II] Showing dotted images — press any key to continue")
    cv2.imshow("Dotted Tool Image",   sif_result["tool_image"])
    cv2.imshow("Dotted Target Image", sif_result["target_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 9. Build prompt
    prompt = _build_prompt(
        task, task_steps, tool_name, target_name,
        n_tool_keypoints=len(sif_result["tool_keypoints"]),
        n_target_keypoints=len(sif_result["target_keypoints"]),
        n_directions=len(sif_result["directions"])
    )

    # 10. VLM call
    print("[Module II] Querying GPT-4o for affordance keypoint selection...")
    raw_output = _call_vlm_with_images(prompt, dotted_tool_path, dotted_target_path)
    print(f"[Module II] VLM raw output:\n  {raw_output}")

    # 11. Parse keypoints
    keypoints = _parse_keypoint_output(raw_output)

    # 12. Resolve to coordinates
    tool_h, tool_w     = sif_result["tool_image"].shape[:2]
    target_h, target_w = sif_result["target_image"].shape[:2]
    coords = resolve_to_coordinates(
        keypoints,
        sif_result["tool_keypoints"],
        sif_result["target_keypoints"],
        sif_result["directions"],
        tool_image_size=(tool_w, tool_h),
        target_image_size=(target_w, target_h)
    )

    # 13. Draw verified keypoints
    from keypoint_gen.verify import draw_selected_keypoints, show_verification_grid, pixel_inspector
    verified_tool_path, verified_target_path = draw_selected_keypoints(
        sif_result["tool_image"],
        sif_result["target_image"],
        coords,
        task_id
    )

    # 14. Verification grid
    grid_path = show_verification_grid(
        TOOL_IMAGE,
        sif_result["tool_image"],
        cv2.imread(verified_tool_path),
        coords,
        task_id
    )

    # 15. Interactive pixel inspector
    print("[Module II] Interactive inspector open.")
    print("Click dots to confirm pixel values. Press Q to continue.")
    verified_tool_img = cv2.imread(verified_tool_path)
    pixel_inspector(verified_tool_img, title="Verified Tool — Click to Inspect")

    # 16. Build and save result
    result = AffordanceResult(
        task_id=task_id,
        task=task,
        tool_name=tool_name,
        target_name=target_name,
        keypoints=keypoints,
        coordinates=coords,
        dotted_tool_image_path=dotted_tool_path,
        dotted_target_image_path=dotted_target_path,
        verified_tool_image_path=verified_tool_path,
        verified_target_image_path=verified_target_path,
        verification_grid_path=grid_path
    )
    _save_affordance_result(result)

    # 17. Update memory
    _update_memory(task_id, keypoints, coords, dotted_tool_path, dotted_target_path)

    return result
