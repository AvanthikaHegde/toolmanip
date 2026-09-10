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

# Minimum gap between two keypoints, in pixels. Two markers closer than
# 2 * _DOT_RADIUS overlap outright, and the direction arrows radiate a further
# 20px, so anything under ~15px produces the unreadable knot of overlapping
# dots and τ labels seen in run ed3c5129 (keypoints 1, 6 and 10 all landed
# within 2px of each other at the putty's top corner).
#
# This also protects the Table 4 metric: three labels on one site means three
# different answers are all "correct", and Keypoint Match Rate stops being
# well-defined.
_MIN_KEYPOINT_SEPARATION = 2 * _DOT_RADIUS + 3

# Candidates drawn per requested point before separation filtering. Without
# oversampling, discarding a crowded candidate would return fewer keypoints
# than requested — which would silently corrupt Table 4, where the keypoint
# count is the swept variable.
_OVERSAMPLE = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_separated(candidates: List[Tuple[int, int]],
                      n: int,
                      accepted: List[Tuple[int, int]],
                      min_dist: int = _MIN_KEYPOINT_SEPARATION
                      ) -> List[Tuple[int, int]]:
    """
    Take `n` candidates that sit at least `min_dist` pixels from each other and
    from every point already in `accepted`, appending each pick to `accepted`.

    Replaces the previous exact-coordinate deduplication, which compared
    (int(x), int(y)) tuples and therefore treated (195,37), (196,37) and
    (197,37) as three distinct keypoints.

    Two passes, and the second one matters. Candidates arrive ordered along the
    structure they were sampled from — the skeleton runs end to end, the contour
    runs around the perimeter — so taking the first n that pass separation
    collapses every pick onto one end of the object. Filtering first and then
    sampling evenly across the survivors keeps the spread the caller asked for.

    `accepted` is mutated so separation holds *across* categories as well as
    within them: a contour point may not land on top of an axis point.
    """
    threshold = min_dist * min_dist

    kept: List[Tuple[int, int]] = []
    for p in candidates:
        x, y = int(p[0]), int(p[1])
        far_enough = all((x - ax) ** 2 + (y - ay) ** 2 >= threshold
                         for ax, ay in accepted)
        if far_enough and all((x - kx) ** 2 + (y - ky) ** 2 >= threshold
                              for kx, ky in kept):
            kept.append((x, y))

    if len(kept) <= n:
        picked = kept
    else:
        # Evenly spaced along the survivors, endpoints included.
        idx = np.linspace(0, len(kept) - 1, n).astype(int)
        picked = [kept[i] for i in dict.fromkeys(idx.tolist())]

    accepted.extend(picked)
    return picked


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
        #
        # Each extractor is oversampled, then filtered down to the requested
        # count with a minimum-separation rule. The skeleton endpoint, a
        # Douglas-Peucker corner and an evenly-spaced contour sample routinely
        # converge on the same corner of an object, so sampling exactly n from
        # each category and deduplicating on exact coordinates leaves several
        # labels stacked on one site.
        axis_cand = _extract_skeleton_points(mask, n_axis_points * _OVERSAMPLE)
        contour_cand = _extract_contour_points(mask, n_contour_points * _OVERSAMPLE)
        bg_cand = _background_points(mask, n_background_points * _OVERSAMPLE, h, w)

        accepted: List[Tuple[int, int]] = []
        axis_pts = _select_separated(axis_cand, n_axis_points, accepted)
        contour_pts = _select_separated(contour_cand, n_contour_points, accepted)
        bg_pts = _select_separated(bg_cand, n_background_points, accepted)

        all_keypoints: List[Tuple[int, int]] = axis_pts + contour_pts + bg_pts

        requested = n_axis_points + n_contour_points + n_background_points
        if len(all_keypoints) < requested:
            warnings.warn(
                f"[SIF] {label}: {len(all_keypoints)} keypoints placed but "
                f"{requested} requested "
                f"(axis {len(axis_pts)}/{n_axis_points}, "
                f"contour {len(contour_pts)}/{n_contour_points}, "
                f"background {len(bg_pts)}/{n_background_points}). The mask is "
                f"too small to hold that many points "
                f"{_MIN_KEYPOINT_SEPARATION}px apart. Any keypoint-count sweep "
                f"is measuring the wrong count for this image.",
                RuntimeWarning,
                stacklevel=2
            )

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
