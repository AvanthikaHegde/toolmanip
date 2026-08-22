"""
SIF — Structured Interaction Field for the ToolManip framework.

Implements Paper Section 3.3.1: visual prompt generation by annotating
candidate keypoints and direction vectors onto ROI images.

  generate_sif(): Takes tool and target ROI dicts (output of HRE) and
    returns both images annotated with numbered keypoints and τ direction
    arrows, plus the coordinate lists for downstream use.

Keypoint placement strategy (paper Section 3.3.1):
  1. Extract morphological skeleton of the object mask.
  2. Scatter points symmetrically along the skeleton central axis.
  3. Scatter points along the contour, explicitly including corners.
  4. Scatter random points in the background region for robustness.
  5. Number all points sequentially starting from 1.

Direction vectors (paper Section 4.4):
  - Uniformly distributed angles 0–360°, labelled τ0, τ1, …
  - Drawn as arrows from the image center point.

Visual style (paper Figure 4 / Section 4.4):
  - High-contrast colour vs background.
  - Medium marker size.
  - Filled circles with white number labels.

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).

Dependencies:
  - opencv-python
  - numpy
  - scipy (for morphological skeleton)
"""

import math
import random
import warnings
from typing import List, Tuple

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    warnings.warn("[SIF] opencv-python not installed.", ImportWarning, stacklevel=2)

try:
    from scipy.ndimage import label as scipy_label
    from skimage.morphology import skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    warnings.warn(
        "[SIF] scikit-image not installed — skeleton-based axis points disabled. "
        "Falling back to grid sampling along the mask bounding box.",
        ImportWarning,
        stacklevel=2
    )

# ---------------------------------------------------------------------------
# Visual annotation constants  (paper Section 4.4)
# ---------------------------------------------------------------------------

_DOT_RADIUS = 6          # medium marker size
_DOT_COLOR = (0, 0, 255)  # bright red — high contrast on tool images
_TXT_COLOR = (255, 255, 255)  # white label text
_ARROW_COLOR = (0, 200, 0)    # green arrows
_ARROW_LENGTH = 30
_FONT = cv2.FONT_HERSHEY_SIMPLEX if _CV2_AVAILABLE else None
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_skeleton_points(mask: np.ndarray, n: int) -> List[Tuple[int, int]]:
    """
    Extract `n` points evenly spaced along the morphological skeleton
    of the binary mask.

    Paper Section 3.3.1: the skeleton approximates the object's central axis;
    sampling points symmetrically along it captures the primary structure.

    Falls back to bounding-box grid if scikit-image is not available.
    """
    if mask is None or mask.size == 0:
        return []

    binary = (mask > 127).astype(np.uint8)

    if _SKIMAGE_AVAILABLE:
        try:
            skel = skeletonize(binary).astype(np.uint8)
            ys, xs = np.where(skel > 0)
            if len(xs) < 2:
                raise ValueError("skeleton too sparse")
            # Sort along the dominant axis for symmetric sampling
            if xs.max() - xs.min() >= ys.max() - ys.min():
                order = np.argsort(xs)
            else:
                order = np.argsort(ys)
            xs, ys = xs[order], ys[order]
            indices = np.linspace(0, len(xs) - 1, n, dtype=int)
            return [(int(xs[i]), int(ys[i])) for i in indices]
        except Exception:
            pass  # fall through to bounding-box fallback

    # Fallback: evenly sample along the horizontal span of the mask
    ys_all, xs_all = np.where(binary > 0)
    if len(xs_all) == 0:
        return []
    x_min, x_max = int(xs_all.min()), int(xs_all.max())
    y_center = int(ys_all.mean())
    xs_sampled = np.linspace(x_min, x_max, n, dtype=int)
    return [(int(x), y_center) for x in xs_sampled]


def _extract_contour_points(mask: np.ndarray, n: int) -> List[Tuple[int, int]]:
    """
    Extract `n` points along the object contour, explicitly including corners.

    Paper Section 3.3.1: contour points anchor the keypoint field to the
    object's silhouette edges; corners are included because they are
    visually salient interaction candidates.
    """
    if not _CV2_AVAILABLE or mask is None or mask.size == 0:
        return []

    binary = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    pts = contour[:, 0, :]  # shape (N, 2)

    # Include corners (Douglas-Peucker approx)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corner_pts = [tuple(p[0]) for p in approx]

    # Evenly sample additional points from the full contour
    n_extra = max(0, n - len(corner_pts))
    indices = np.linspace(0, len(pts) - 1, n_extra, dtype=int)
    sampled = [tuple(pts[i]) for i in indices]

    combined = corner_pts + sampled
    # Deduplicate and trim to n
    seen = set()
    unique = []
    for p in combined:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique[:n]


def _background_points(mask: np.ndarray, n: int,
                        img_h: int, img_w: int) -> List[Tuple[int, int]]:
    """
    Scatter `n` random points in the background region of the image.

    Paper Section 3.3.1: background points provide robustness by giving
    the VLM reference points outside the object boundary.
    """
    if mask is None:
        bg_mask = np.ones((img_h, img_w), dtype=np.uint8)
    else:
        bg_mask = (mask < 128).astype(np.uint8)

    ys, xs = np.where(bg_mask > 0)
    if len(xs) == 0:
        return []

    rng = random.Random(42)
    indices = rng.sample(range(len(xs)), min(n, len(xs)))
    return [(int(xs[i]), int(ys[i])) for i in indices]


def _draw_keypoints(image: np.ndarray,
                    keypoints: List[Tuple[int, int]],
                    start_index: int = 1) -> np.ndarray:
    """
    Draw numbered filled circles onto the image for each keypoint.

    Paper Section 4.4: medium marker size, high-contrast colour,
    white numeric labels placed to the upper-right of each dot.
    """
    if not _CV2_AVAILABLE:
        return image
    out = image.copy()
    for i, (x, y) in enumerate(keypoints):
        idx = start_index + i
        cv2.circle(out, (x, y), _DOT_RADIUS, _DOT_COLOR, -1)
        cv2.putText(out, str(idx), (x + _DOT_RADIUS, y - _DOT_RADIUS),
                    _FONT, _FONT_SCALE, _TXT_COLOR, _FONT_THICKNESS, cv2.LINE_AA)
    return out


def _draw_directions(image: np.ndarray,
                     keypoints: List[Tuple[int, int]],
                     n_directions: int) -> Tuple[np.ndarray, List[float]]:
    """
    Draw `n_directions` uniformly spaced direction arrows around each keypoint.

    Paper Section 3.3.1 / Figure 4: direction vectors τ0…τ(n-1) are drawn as
    small arrows radiating from each keypoint location. Labels (τ0, τ1, …) are
    shown only on the first keypoint to avoid clutter.
    Arrow length = 20px.
    """
    _ARROW_LEN = 20
    if not _CV2_AVAILABLE:
        angles = [2 * math.pi * i / n_directions for i in range(n_directions)]
        return image, angles

    out = image.copy()
    angles = [2 * math.pi * i / n_directions for i in range(n_directions)]

    for kp_idx, (cx, cy) in enumerate(keypoints):
        label_this = (kp_idx == 0)  # only label first keypoint
        for i, angle in enumerate(angles):
            ex = int(cx + _ARROW_LEN * math.cos(angle))
            ey = int(cy + _ARROW_LEN * math.sin(angle))
            cv2.arrowedLine(out, (cx, cy), (ex, ey), _ARROW_COLOR, 1,
                            tipLength=0.3, line_type=cv2.LINE_AA)
            if label_this:
                label_x = int(cx + (_ARROW_LEN + 8) * math.cos(angle))
                label_y = int(cy + (_ARROW_LEN + 8) * math.sin(angle))
                cv2.putText(out, f"t{i}", (label_x, label_y),
                            _FONT, _FONT_SCALE, _ARROW_COLOR,
                            _FONT_THICKNESS, cv2.LINE_AA)

    return out, angles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sif(
    tool_roi: dict,
    target_roi: dict,
    n_axis_points: int = 5,
    n_contour_points: int = 6,
    n_background_points: int = 4,
    n_directions: int = 6
) -> dict:
    """
    Generate Structured Interaction Field visual prompts on tool and target ROIs.

    Paper Section 3.3.1 — Structured Interaction Field:
      Annotates both ROI images with candidate keypoints and direction vectors
      so the VLM can select interaction points by number.

    Keypoint placement (paper Section 3.3.1):
      1. Points along the morphological skeleton (central axis).
      2. Points along the contour, including corners.
      3. Random background points for robustness.
      All points are numbered sequentially starting from 1.

    Direction vectors (paper Section 4.4):
      Uniformly distributed angles 0–360°, labelled τ0, τ1, …,
      drawn as arrows from the image centre.

    Args:
        tool_roi:              dict from hre.extract_roi for the tool.
        target_roi:            dict from hre.extract_roi for the target.
        n_axis_points:         Points along skeleton/central axis per image.
        n_contour_points:      Points along contour (including corners) per image.
        n_background_points:   Random background points per image.
        n_directions:          Number of direction vectors per image.

    Returns:
        dict with keys:
          "tool_image"       : np.ndarray — annotated tool ROI
          "target_image"     : np.ndarray — annotated target ROI
          "tool_keypoints"   : List[tuple] — (x, y) coords, index = position in list
          "target_keypoints" : List[tuple]
          "directions"       : List[float] — angles in radians (same for both images)
    """
    results = {}

    for label, roi in [("tool", tool_roi), ("target", target_roi)]:
        img = roi.get("cropped_image")
        mask = roi.get("mask")

        if img is None or img.size == 0:
            h, w = 480, 640
            img = np.zeros((h, w, 3), dtype=np.uint8)
            mask = None

        h, w = img.shape[:2]

        # --- Keypoint placement ---
        axis_pts = _extract_skeleton_points(mask, n_axis_points)
        contour_pts = _extract_contour_points(mask, n_contour_points)
        bg_pts = _background_points(mask, n_background_points, h, w)

        all_keypoints: List[Tuple[int, int]] = []
        seen: set = set()
        for pt in axis_pts + contour_pts + bg_pts:
            key = (int(pt[0]), int(pt[1]))
            if key not in seen:
                seen.add(key)
                all_keypoints.append(key)

        # --- Annotate image ---
        annotated = _draw_keypoints(img, all_keypoints, start_index=1)
        annotated, direction_angles = _draw_directions(annotated, all_keypoints, n_directions)

        results[label] = {
            "annotated_image": annotated,
            "keypoints": all_keypoints,
            "directions": direction_angles
        }

    return {
        "tool_image": results["tool"]["annotated_image"],
        "target_image": results["target"]["annotated_image"],
        "tool_keypoints": results["tool"]["keypoints"],
        "target_keypoints": results["target"]["keypoints"],
        "directions": results["tool"]["directions"]
    }
