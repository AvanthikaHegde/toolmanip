"""
Module II for a single-camera rig — tool-side affordance reasoning.

Produces {H, F, tau_t}: the grasp point, the functional point, and the force
direction at F. The paper's full output is {H, F, O, Q, tau_t, tau_o}, but O, Q
and tau_o describe the operation on the target and need a view of the
workspace, which this rig does not have.

Unlike agents/VLM2_AffordanceReasoning.py, the keypoints are carried back to
full-frame pixels and deprojected to metric XYZ.

THE COORDINATE CHAIN
--------------------
HRE nests two crops and rescales each to 512x512 (hre.py:228, per the paper),
so a SIF keypoint is three coordinate systems away from the camera frame:

    camera frame (640x480)
      |-- extract_roi()               bbox in FULL-FRAME coords, output 512x512
           |-- extract_functional_region()  bbox in TOOL-CROP coords, output 512x512
                |-- generate_sif()          keypoints in FUNCTIONAL-ROI coords

Each step is an anisotropic affine map: the mallet's 472x192 bbox is stretched
to 512x512, so x scales by 0.922 and y by 0.375. Composing these wrongly does
not raise — it yields a plausible pixel somewhere else on the board, and a
confident 3D point for the wrong object. RoiStage/map_to_full exist to make the
composition explicit and testable rather than inlined.

The anisotropy also means ANGLES ARE NOT PRESERVED. tau_t as drawn by SIF is an
angle in stretched 512x512 space; the true direction is recovered by mapping
two points along the ray back to the full frame and re-measuring there.

Run (after agents/module1.py):
    python -m agents.module2

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera import load_capture, deproject_pixel, Capture, Intrinsics

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

MEMORY_FILE = Path("memory") / "memory_1.json"
PROMPT_BUNDLE = Path("prompts") / "P2_tool.json"
RESULTS_DIR = Path("results")
MODEL = "gpt-4o"

# Depth sampling window at a keypoint. Keypoints sit on tool features — edges,
# shafts, rims — which is exactly where depth drops out, so the sample has to
# cover a neighbourhood without straying onto the board behind.
KEYPOINT_PATCH = 7

# When a keypoint has no depth, step toward the other keypoint looking for a
# valid sample. Bounded so a rescue cannot silently return a point at the far
# end of the tool.
RESCUE_STEP_PX = 3
RESCUE_MAX_FRACTION = 0.25


class AffordanceError(RuntimeError):
    """Raised when affordance reasoning cannot produce a usable result."""


# ---------------------------------------------------------------------------
# Coordinate chain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoiStage:
    """
    One crop-and-rescale performed by HRE.

    bbox is (x1, y1, x2, y2) in the PARENT image's coordinates; size is the
    (w, h) of the rescaled image this stage handed downstream.
    """
    bbox: Tuple[int, int, int, int]
    size: Tuple[int, int]

    @property
    def scale(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        w, h = self.size
        return ((x2 - x1) / w, (y2 - y1) / h)


def map_to_parent(u: float, v: float, stage: RoiStage) -> Tuple[float, float]:
    """Map a pixel in this stage's output image back into its parent image."""
    x1, y1, _, _ = stage.bbox
    sx, sy = stage.scale
    return (x1 + u * sx, y1 + v * sy)


def map_to_full(u: float, v: float, stages: List[RoiStage]) -> Tuple[float, float]:
    """
    Map a pixel from the innermost ROI all the way back to the camera frame.

    stages is ordered outermost-first, matching the order the crops were taken,
    so it is walked in reverse.
    """
    for stage in reversed(stages):
        u, v = map_to_parent(u, v, stage)
    return (u, v)


def map_angle_to_full(angle_rad: float, origin_uv: Tuple[float, float],
                      stages: List[RoiStage], probe: float = 32.0) -> float:
    """
    Re-measure a direction in full-frame coordinates.

    An angle cannot simply be carried through an anisotropic scale, so two
    points along the ray are mapped instead and the angle re-derived from them.
    """
    u0, v0 = origin_uv
    u1 = u0 + probe * math.cos(angle_rad)
    v1 = v0 + probe * math.sin(angle_rad)
    fu0, fv0 = map_to_full(u0, v0, stages)
    fu1, fv1 = map_to_full(u1, v1, stages)
    return math.atan2(fv1 - fv0, fu1 - fu0)


# ---------------------------------------------------------------------------
# Depth with rescue
# ---------------------------------------------------------------------------

def _depth_with_rescue(cap: Capture, uv: Tuple[int, int],
                       toward: Tuple[int, int]) -> tuple:
    """
    Deproject uv, stepping toward `toward` if that pixel has no depth.

    Thin specular features (a screwdriver shaft, a mallet rim) routinely have
    no valid depth. Rather than failing outright or silently returning the
    origin, the sample walks along the line to the other keypoint — which stays
    on the tool — and reports how far it had to move, so the caller can judge
    whether the rescued point is still meaningful.
    """
    u, v = int(round(uv[0])), int(round(uv[1]))
    tu, tv = int(round(toward[0])), int(round(toward[1]))

    xyz = deproject_pixel(cap.depth, u, v, cap.intrinsics, patch=KEYPOINT_PATCH)
    if xyz is not None:
        return xyz, 0.0, (u, v)

    span = math.hypot(tu - u, tv - v)
    if span < 1:
        return None, 0.0, (u, v)
    limit = span * RESCUE_MAX_FRACTION
    ux, uy = (tu - u) / span, (tv - v) / span

    d = float(RESCUE_STEP_PX)
    while d <= limit:
        cu, cv = int(round(u + ux * d)), int(round(v + uy * d))
        if 0 <= cu < cap.depth.shape[1] and 0 <= cv < cap.depth.shape[0]:
            xyz = deproject_pixel(cap.depth, cu, cv, cap.intrinsics,
                                  patch=KEYPOINT_PATCH)
            if xyz is not None:
                return xyz, d, (cu, cv)
        d += RESCUE_STEP_PX
    return None, 0.0, (u, v)


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------

def _encode(path: Path) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _ask_vlm(task: str, task_steps: str, tool: str,
             n_keypoints: int, n_directions: int, image_path: Path) -> str:
    with open(PROMPT_BUNDLE) as f:
        cfg = json.load(f)["steps"]["step1_tool_affordance"]

    instruction = cfg["instruction"].format(
        task=task, task_steps=task_steps, tool=tool,
        n_keypoints=n_keypoints, max_tau=n_directions - 1,
    )
    if cfg.get("output_format"):
        instruction += f"\n\nRespond ONLY in this format:\n{cfg['output_format']}"

    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[
            {"role": "system", "content": cfg["role"]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _encode(image_path)}},
                {"type": "text", "text": instruction},
            ]},
        ],
    )
    return resp.choices[0].message.content.strip()


def _parse_keypoints(raw: str, n_keypoints: int, n_directions: int) -> tuple:
    """Parse H, F and tau, validating every label against what SIF drew."""
    def grab(pattern: str) -> Optional[int]:
        m = re.search(pattern, raw, re.IGNORECASE)
        return int(m.group(1)) if m else None

    h = grab(r"Keypoint\s*H\s*:\s*(\d+)")
    f = grab(r"Keypoint\s*F\s*:\s*(\d+)")
    t = grab(r"Vector\s*t\w*\s*:\s*(?:tau|τ)?\s*(\d+)")

    problems = []
    if h is None or not 1 <= h <= n_keypoints:
        problems.append(f"H={h!r} outside 1..{n_keypoints}")
    if f is None or not 1 <= f <= n_keypoints:
        problems.append(f"F={f!r} outside 1..{n_keypoints}")
    if t is None or not 0 <= t < n_directions:
        problems.append(f"tau={t!r} outside 0..{n_directions - 1}")
    if h is not None and h == f:
        problems.append("H and F are the same keypoint")
    if problems:
        raise AffordanceError(
            "VLM keypoint selection is unusable:\n  - " + "\n  - ".join(problems)
            + f"\n  Raw response: {raw[:300]}"
        )
    return h, f, t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_module2(capture_dir: Optional[Path] = None, verbose: bool = True) -> dict:
    if not API_KEY:
        raise AffordanceError("OPENAI_API_KEY is not set (.env).")
    if not MEMORY_FILE.exists():
        raise AffordanceError("memory/memory_1.json not found. Run Module I first.")

    with open(MEMORY_FILE) as f:
        records = json.load(f)
    record = records[-1] if isinstance(records, list) else records

    task = record.get("task", "")
    tool_name = record.get("selected_tool", {}).get("name", "")
    task_id = record.get("task_id", str(uuid.uuid4())[:8])
    steps_txt = "; ".join(
        f"{s['step_index']}. {s['action']} {s['target_object']}"
        for s in record.get("task_steps", [])
    )
    if not tool_name:
        raise AffordanceError("No selected tool in memory. Run Module I first.")

    capture_dir = Path(capture_dir or record.get("capture_dir", "captures/latest"))
    cap = load_capture(capture_dir)
    colour_path = capture_dir / "color.png"
    full_h, full_w = cap.color.shape[:2]

    if verbose:
        print(f"[Module II] task : {task}")
        print(f"[Module II] tool : {tool_name}")
        print(f"[Module II] frame: {colour_path} ({full_w}x{full_h})")

    # --- HRE: two nested crops -------------------------------------------
    from keypoint_gen.hre import extract_roi, extract_functional_region
    from keypoint_gen.sif import generate_sif

    if verbose:
        print("[Module II] HRE stage 1 — locating the tool ...")
    tool_roi = extract_roi(image_path=str(colour_path), description=tool_name,
                           expansion_factor=0.1)
    if tool_roi.get("_fallback"):
        raise AffordanceError(
            f"HRE fell back to the whole image: {tool_roi.get('_reason')}. "
            f"Keypoints would describe the photo, not the tool "
            f"(the paper's 'w/o HRE' ablation)."
        )

    if verbose:
        print("[Module II] HRE stage 2 — locating the functional region ...")
    func_roi = extract_functional_region(
        cropped_image=tool_roi["cropped_image"],
        functional_description=f"functional region of {tool_name}",
        expansion_factor=0.1,
    )
    if func_roi.get("_fallback"):
        # Stage 2 failing is survivable: the tool crop is still a real ROI, so
        # fall back to it deliberately rather than aborting the run.
        if verbose:
            print(f"[Module II] stage 2 fell back ({func_roi.get('_reason')}); "
                  f"using the stage 1 tool crop.")
        sif_input = tool_roi
        stages = [RoiStage(tuple(tool_roi["bbox"]),
                           tool_roi["cropped_image"].shape[1::-1])]
    else:
        sif_input = func_roi
        stages = [
            RoiStage(tuple(tool_roi["bbox"]), tool_roi["cropped_image"].shape[1::-1]),
            RoiStage(tuple(func_roi["bbox"]), func_roi["cropped_image"].shape[1::-1]),
        ]

    if verbose:
        for i, st in enumerate(stages, 1):
            sx, sy = st.scale
            print(f"[Module II]   stage {i}: bbox={list(st.bbox)} "
                  f"size={st.size} scale=({sx:.3f}, {sy:.3f})")

    # --- SIF --------------------------------------------------------------
    if verbose:
        print("[Module II] SIF — placing candidate keypoints ...")
    sif = generate_sif(sif_input, {"cropped_image": None, "mask": None})
    kps: List[tuple] = sif["tool_keypoints"]
    dirs: List[float] = sif["directions"]
    if not kps:
        raise AffordanceError("SIF placed no keypoints — the mask is unusable.")

    RESULTS_DIR.mkdir(exist_ok=True)
    dotted_path = RESULTS_DIR / f"m2_dotted_{task_id}.png"
    cv2.imwrite(str(dotted_path), sif["tool_image"])
    if verbose:
        print(f"[Module II]   {len(kps)} keypoints, {len(dirs)} directions "
              f"-> {dotted_path}")

    # --- VLM selection ----------------------------------------------------
    if verbose:
        print("[Module II] Querying GPT-4o for H, F, tau_t ...")
    raw = _ask_vlm(task, steps_txt, tool_name, len(kps), len(dirs), dotted_path)
    if verbose:
        print(f"[Module II]   raw: {raw.replace(chr(10), ' | ')}")
    h_lab, f_lab, t_lab = _parse_keypoints(raw, len(kps), len(dirs))

    h_roi = kps[h_lab - 1]
    f_roi = kps[f_lab - 1]
    tau_roi = dirs[t_lab]

    # --- back to the camera frame ----------------------------------------
    h_full = map_to_full(h_roi[0], h_roi[1], stages)
    f_full = map_to_full(f_roi[0], f_roi[1], stages)
    tau_full = map_angle_to_full(tau_roi, f_roi, stages)

    h_px = (int(round(h_full[0])), int(round(h_full[1])))
    f_px = (int(round(f_full[0])), int(round(f_full[1])))

    # --- 3D ---------------------------------------------------------------
    h_xyz, h_resc, h_used = _depth_with_rescue(cap, h_px, f_px)
    f_xyz, f_resc, f_used = _depth_with_rescue(cap, f_px, h_px)

    span = None
    if h_xyz and f_xyz:
        span = float(np.linalg.norm(np.array(f_xyz) - np.array(h_xyz)))

    result = {
        "task_id": task_id, "task": task, "tool": tool_name,
        "capture_dir": str(capture_dir),
        "roi_stages": [{"bbox": list(s.bbox), "size": list(s.size),
                        "scale": list(s.scale)} for s in stages],
        "labels": {"H": h_lab, "F": f_lab, "tau": t_lab},
        "roi_px": {"H": list(h_roi), "F": list(f_roi)},
        "full_px": {"H": list(h_px), "F": list(f_px)},
        "sampled_px": {"H": list(h_used), "F": list(f_used)},
        "rescue_px": {"H": h_resc, "F": f_resc},
        "xyz_m": {
            "H": [round(c, 4) for c in h_xyz] if h_xyz else None,
            "F": [round(c, 4) for c in f_xyz] if f_xyz else None,
        },
        "tau_t_deg": round(math.degrees(tau_full), 2),
        "tau_t_roi_deg": round(math.degrees(tau_roi), 2),
        "grasp_to_functional_m": round(span, 4) if span else None,
        "raw_vlm": raw,
    }

    out_path = RESULTS_DIR / f"module2_{task_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    overlay = _draw_verification(cap, h_px, f_px, h_used, f_used,
                                 tau_full, tool_name, task,
                                 RESULTS_DIR / f"module2_{task_id}.png")

    if verbose:
        print(f"\n=== AFFORDANCE ({tool_name}) ===")
        print(f"  H  label {h_lab}: ROI {h_roi} -> full {h_px} px")
        print(f"     3D {_fmt(h_xyz)}" + (f"   [rescued {h_resc:.0f} px]" if h_resc else ""))
        print(f"  F  label {f_lab}: ROI {f_roi} -> full {f_px} px")
        print(f"     3D {_fmt(f_xyz)}" + (f"   [rescued {f_resc:.0f} px]" if f_resc else ""))
        print(f"  tau_t: {result['tau_t_deg']:.1f} deg in frame "
              f"(was {result['tau_t_roi_deg']:.1f} deg in stretched ROI)")
        if span:
            print(f"  |F-H| = {span * 1000:.0f} mm   <- compare to the real tool")
        print(f"\n[Saved]   {out_path}")
        print(f"[Overlay] {overlay}   <- H and F must sit on the tool")

    return result


def _fmt(xyz) -> str:
    if xyz is None:
        return "UNAVAILABLE (no valid depth)"
    return f"X={xyz[0]:+.4f} Y={xyz[1]:+.4f} Z={xyz[2]:+.4f} m"


def _draw_verification(cap: Capture, h_px, f_px, h_used, f_used,
                       tau: float, tool: str, task: str, out: Path) -> Path:
    """
    Draw H, F and tau_t on the ORIGINAL frame.

    Drawn on the uncropped capture on purpose: markers plotted on the ROI would
    look correct even if the crop offsets were composed wrongly. This is the
    only view that tests the mapping.
    """
    vis = cap.color.copy()
    cv2.arrowedLine(vis, f_px,
                    (int(f_px[0] + 70 * math.cos(tau)),
                     int(f_px[1] + 70 * math.sin(tau))),
                    (255, 0, 255), 2, tipLength=0.25)
    cv2.line(vis, h_px, f_px, (200, 200, 200), 1)

    for px, used, colour, name in ((h_px, h_used, (0, 255, 0), "H grasp"),
                                   (f_px, f_used, (0, 128, 255), "F functional")):
        cv2.drawMarker(vis, px, colour, cv2.MARKER_CROSS, 20, 2)
        cv2.circle(vis, px, 11, colour, 2)
        if tuple(used) != tuple(px):
            # Show where depth was actually sampled when the keypoint itself
            # had none, so a rescued point is never mistaken for a direct read.
            cv2.circle(vis, tuple(used), 5, (0, 0, 255), 2)
            cv2.line(vis, px, tuple(used), (0, 0, 255), 1)
        cv2.putText(vis, name, (px[0] + 14, px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    cv2.putText(vis, f"{task} / {tool}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), vis)
    return out


def main() -> int:
    try:
        run_module2()
    except AffordanceError as exc:
        print(f"[Module II] ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
