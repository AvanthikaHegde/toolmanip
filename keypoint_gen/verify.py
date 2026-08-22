"""
Verification utilities for Module II affordance reasoning results.

Provides visual verification of VLM-selected keypoints:
  draw_selected_keypoints() — overlay selections on dotted images with colour coding
  pixel_inspector()         — interactive click-to-inspect pixel coordinates
  show_verification_grid()  — 3-panel side-by-side view + coordinate table
"""

import math
from pathlib import Path

import cv2
import numpy as np

from structures.affordance_structures import AffordanceCoordinates

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 2
_CIRCLE_RADIUS = 8
_ARROW_LEN = 35


def _draw_arrow(img: np.ndarray, origin_u: int, origin_v: int,
                angle_deg: float, color: tuple) -> None:
    angle_rad = math.radians(angle_deg)
    ex = int(origin_u + _ARROW_LEN * math.cos(angle_rad))
    ey = int(origin_v + _ARROW_LEN * math.sin(angle_rad))
    cv2.arrowedLine(img, (origin_u, origin_v), (ex, ey), color, 2,
                    tipLength=0.35, line_type=cv2.LINE_AA)


def draw_selected_keypoints(
    tool_dotted: np.ndarray,
    target_dotted: np.ndarray,
    coords: AffordanceCoordinates,
    task_id: str
) -> tuple:
    """
    Draw VLM-selected keypoints on dotted images with distinct colors.

    Color scheme:
      H  grasp point      → GREEN  circle + label "H"
      F  functional point → RED    circle + label "F"
      τt tool direction   → BLUE   arrow from F
      O  start point      → YELLOW circle + label "O"
      Q  end point        → ORANGE circle + label "Q"
      τo target direction → CYAN   arrow from O
      O→Q operation path  → WHITE  line

    Returns (verified_tool_path, verified_target_path).
    """
    tool_img = tool_dotted.copy()
    target_img = target_dotted.copy()

    # --- Tool image ---
    H = coords.grasp_point_H
    F = coords.functional_point_F

    cv2.circle(tool_img, (H.u, H.v), _CIRCLE_RADIUS, (0, 255, 0), -1)
    cv2.putText(tool_img, "H", (H.u + _CIRCLE_RADIUS, H.v - _CIRCLE_RADIUS),
                _FONT, _FONT_SCALE, (0, 255, 0), _FONT_THICKNESS, cv2.LINE_AA)

    cv2.circle(tool_img, (F.u, F.v), _CIRCLE_RADIUS, (0, 0, 255), -1)
    cv2.putText(tool_img, "F", (F.u + _CIRCLE_RADIUS, F.v - _CIRCLE_RADIUS),
                _FONT, _FONT_SCALE, (0, 0, 255), _FONT_THICKNESS, cv2.LINE_AA)

    _draw_arrow(tool_img, F.u, F.v, coords.tool_direction_angle, (255, 0, 0))

    # --- Target image ---
    O = coords.start_point_O
    Q = coords.end_point_Q

    cv2.line(target_img, (O.u, O.v), (Q.u, Q.v), (255, 255, 255), 2, cv2.LINE_AA)

    cv2.circle(target_img, (O.u, O.v), _CIRCLE_RADIUS, (0, 255, 255), -1)
    cv2.putText(target_img, "O", (O.u + _CIRCLE_RADIUS, O.v - _CIRCLE_RADIUS),
                _FONT, _FONT_SCALE, (0, 255, 255), _FONT_THICKNESS, cv2.LINE_AA)

    cv2.circle(target_img, (Q.u, Q.v), _CIRCLE_RADIUS, (0, 165, 255), -1)
    cv2.putText(target_img, "Q", (Q.u + _CIRCLE_RADIUS, Q.v - _CIRCLE_RADIUS),
                _FONT, _FONT_SCALE, (0, 165, 255), _FONT_THICKNESS, cv2.LINE_AA)

    _draw_arrow(target_img, O.u, O.v, coords.target_direction_angle, (255, 255, 0))

    tool_path = str(RESULTS_DIR / f"verified_tool_{task_id}.jpg")
    target_path = str(RESULTS_DIR / f"verified_target_{task_id}.jpg")
    cv2.imwrite(tool_path, tool_img)
    cv2.imwrite(target_path, target_img)

    print(f"[Verify] Saved verified_tool  → {tool_path}")
    print(f"[Verify] Saved verified_target → {target_path}")
    return tool_path, target_path


def pixel_inspector(image: np.ndarray, title: str = "Inspector") -> None:
    """
    Interactive pixel inspector — click anywhere to see exact pixel coordinates.
    Press Q to close and continue pipeline.
    """
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = image.shape[:2]
            print(f"  Clicked pixel  : (u={x}, v={y})")
            print(f"  Normalized     : ({x/w:.4f}, {y/h:.4f})")

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse_callback)
    print(f"\n[Verify] Click anywhere on '{title}' to inspect pixels.")
    print(f"[Verify] Press Q to close and continue.")
    while True:
        cv2.imshow(title, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    try:
        cv2.destroyWindow(title)
    except cv2.error:
        pass


def show_verification_grid(
    original_path: str,
    dotted_image: np.ndarray,
    verified_image: np.ndarray,
    coords: AffordanceCoordinates,
    task_id: str
) -> str:
    """
    Show three images side by side and print coordinate table to console.

    Left:   Original image (what camera sees)
    Center: Dotted SIF image (what VLM sees)
    Right:  Verified image (what VLM selected)

    Saves grid to results/verification_grid_{task_id}.jpg.
    Returns path to saved grid image.
    """
    original = cv2.imread(original_path)
    if original is None:
        original = np.zeros_like(dotted_image)

    target_h = 480
    def _resize(img):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_LINEAR)

    orig_r = _resize(original)
    dot_r  = _resize(dotted_image)
    ver_r  = _resize(verified_image)

    # Add panel labels
    for panel, label in [(orig_r, "Original"), (dot_r, "VLM Input (SIF)"), (ver_r, "VLM Selection")]:
        cv2.putText(panel, label, (8, 24), _FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, label, (8, 24), _FONT, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

    grid = np.hstack([orig_r, dot_r, ver_r])
    grid_path = str(RESULTS_DIR / f"verification_grid_{task_id}.jpg")
    cv2.imwrite(grid_path, grid)

    # Print coordinate table
    H, F = coords.grasp_point_H, coords.functional_point_F
    O, Q = coords.start_point_O, coords.end_point_Q
    print("\n┌─────────────────┬──────────────┬────────────────────┐")
    print("│ Point           │ Pixel (u,v)  │ Normalized         │")
    print("├─────────────────┼──────────────┼────────────────────┤")
    print(f"│ Grasp H         │ ({H.u:4d},{H.v:4d}) │ ({H.u_norm:.4f},{H.v_norm:.4f})   │")
    print(f"│ Functional F    │ ({F.u:4d},{F.v:4d}) │ ({F.u_norm:.4f},{F.v_norm:.4f})   │")
    print(f"│ Start O         │ ({O.u:4d},{O.v:4d}) │ ({O.u_norm:.4f},{O.v_norm:.4f})   │")
    print(f"│ End Q           │ ({Q.u:4d},{Q.v:4d}) │ ({Q.u_norm:.4f},{Q.v_norm:.4f})   │")
    print("└─────────────────┴──────────────┴────────────────────┘")

    cv2.namedWindow("Verification Grid", cv2.WINDOW_NORMAL)
    cv2.imshow("Verification Grid", grid)
    cv2.waitKey(0)
    try:
        cv2.destroyWindow("Verification Grid")
    except cv2.error:
        pass

    print(f"[Verify] Grid saved → {grid_path}")
    return grid_path
