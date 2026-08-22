"""
HRE — Hierarchical Region Extraction for the ToolManip framework.

Implements Paper Section 3.3.1: two-stage ROI extraction using LangSAM
for text-grounded image segmentation.

  Stage 1 (extract_roi):       Segment the object/tool from the full scene.
  Stage 2 (extract_functional_region): Isolate the functional sub-region
           (e.g. blade edge of a scraper) from the already-cropped tool ROI.

If LangSAM is not installed, both functions fall back gracefully — they
return the full input image as the ROI with a printed warning and do not crash.

Reference: Zhou et al., "Towards zero-shot robot tool manipulation in
industrial context", Robotics and Computer-Integrated Manufacturing 98 (2026).

Dependencies:
  - lang-sam  (optional; pip install lang-sam)
  - opencv-python
  - numpy
"""

import warnings
import numpy as np

TARGET_SIZE = 512  # bilinear rescale target (paper Section 3.3.1)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    warnings.warn(
        "[HRE] opencv-python is not installed. "
        "HRE will return full images as ROI fallback.",
        ImportWarning,
        stacklevel=2
    )

# ---------------------------------------------------------------------------
# LangSAM availability check
# ---------------------------------------------------------------------------

_LANGSAM_AVAILABLE = False
_langsam_instance = None  # lazy singleton to avoid repeated model loading

try:
    from lang_sam import LangSAM  # noqa: F401
    _LANGSAM_AVAILABLE = True
except ImportError:
    pass

if not _LANGSAM_AVAILABLE:
    warnings.warn(
        "[HRE] LangSAM is not installed (pip install lang-sam). "
        "Falling back to full-image ROI — segmentation disabled.",
        ImportWarning,
        stacklevel=2
    )


def _get_langsam():
    """Lazy-load LangSAM model singleton to avoid reloading on every call."""
    global _langsam_instance
    if _langsam_instance is None:
        from lang_sam import LangSAM
        _langsam_instance = LangSAM()
    return _langsam_instance


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _full_image_fallback(image: np.ndarray, reason: str) -> dict:
    """
    Return a fallback ROI using the entire image.

    Paper Section 3.3.1: used when LangSAM is unavailable or segmentation
    produces no valid mask. Ensures the pipeline continues without crashing.

    The caller is responsible for deciding whether a fallback is acceptable —
    see the _fallback check in VLM2_AffordanceReasoning.run_affordance_reasoning.
    The reason is always printed because several of the paths that land here
    (a LangSAM API mismatch, an unreadable image) otherwise look like success.
    """
    print(f"[HRE] FALLBACK — using full image as ROI. Reason: {reason}")
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    return {
        "cropped_image": image.copy(),
        "bbox": [0, 0, w, h],
        "mask": mask,
        "_fallback": True,
        "_reason": reason
    }


def _expand_bbox(x1: int, y1: int, x2: int, y2: int,
                 img_w: int, img_h: int, factor: float) -> tuple:
    """
    Expand a bounding box by `factor` fraction on each side, clamped to
    image boundaries.

    Paper Section 3.3.1: expansion factor α prevents the ROI from clipping
    the object boundary, preserving context for keypoint placement.
    """
    dw = int((x2 - x1) * factor)
    dh = int((y2 - y1) * factor)
    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(img_w, x2 + dw)
    y2 = min(img_h, y2 + dh)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_roi(
    image_path: str,
    description: str,
    expansion_factor: float = 0.1
) -> dict:
    """
    Stage 1: Segment object/tool from a full scene image using LangSAM.

    Paper Section 3.3.1 — Hierarchical Region Extraction (Stage 1):
      Uses text-grounded segmentation to find the object described by
      `description`, crops the bounding box (expanded by `expansion_factor`),
      and rescales the crop using bilinear interpolation.

    Args:
        image_path:        Path to the full scene or tool board image.
        description:       Natural-language description of the target
                           (e.g. "scraper tool", "white putty on metal plate").
        expansion_factor:  Fraction α by which to expand the bounding box
                           on each side to retain context (default 0.1 = 10%).

    Returns:
        dict with keys:
          "cropped_image" : np.ndarray  — ROI crop (BGR, uint8)
          "bbox"          : [x1, y1, x2, y2] in original image coords
          "mask"          : np.ndarray  — binary mask (uint8, 0/255)
          "_fallback"     : bool        — True if LangSAM was not used
    """
    if not _CV2_AVAILABLE:
        warnings.warn("[HRE] cv2 unavailable, returning placeholder ROI.", RuntimeWarning)
        h, w = 480, 640
        return _full_image_fallback(np.zeros((h, w, 3), dtype=np.uint8), "cv2 unavailable")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        warnings.warn(f"[HRE] Cannot read image: {image_path}. Returning empty fallback.", RuntimeWarning)
        h, w = 480, 640
        return _full_image_fallback(np.zeros((h, w, 3), dtype=np.uint8), f"cannot read {image_path}")

    h, w = image_bgr.shape[:2]

    if not _LANGSAM_AVAILABLE:
        print(f"[HRE] LangSAM unavailable — returning full image as ROI for '{description}'")
        return _full_image_fallback(image_bgr, "LangSAM unavailable")

    # Convert BGR → RGB for LangSAM (which expects PIL or RGB arrays)
    try:
        from PIL import Image as PILImage
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)
        pil_image = pil_image.resize((1024, 1024), PILImage.BILINEAR)

        model = _get_langsam()
        try:
            result = model.predict(pil_image, description)
            if isinstance(result, tuple) and len(result) >= 2:
                masks, boxes = result[0], result[1]
            else:
                raise ValueError("Unexpected predict() return format")

            if masks is None or len(masks) == 0:
                return _full_image_fallback(image_bgr,
                                            "no mask found")
        except Exception as exc:
            print(f"[HRE] LangSAM predict failed: {exc}")
            print("[HRE] Try: pip install -U lang-sam")
            return _full_image_fallback(image_bgr, str(exc))

        # Use the first (highest-confidence) mask and its bounding box
        mask_np = masks[0].cpu().numpy().astype(np.uint8) * 255
        box = boxes[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Expand bounding box
        x1, y1, x2, y2 = _expand_bbox(x1, y1, x2, y2, w, h, expansion_factor)

        cropped = image_bgr[y1:y2, x1:x2]
        mask_crop = mask_np[y1:y2, x1:x2]

        # Bilinear rescale to TARGET_SIZE (paper Section 3.3.1)
        cropped = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(mask_crop, (TARGET_SIZE, TARGET_SIZE),
                               interpolation=cv2.INTER_NEAREST)

        return {
            "cropped_image": cropped,
            "bbox": [x1, y1, x2, y2],
            "mask": mask_crop,
            "_fallback": False
        }

    except Exception as exc:
        warnings.warn(f"[HRE] LangSAM failed: {exc}. Returning full image as ROI.", RuntimeWarning)
        return _full_image_fallback(image_bgr, str(exc))


def extract_functional_region(
    cropped_image: np.ndarray,
    functional_description: str,
    expansion_factor: float = 0.1
) -> dict:
    """
    Stage 2: Isolate the functional sub-region from a cropped tool image.

    Paper Section 3.3.1 — Hierarchical Region Extraction (Stage 2):
      Applied after Stage 1 to further isolate the tool's working surface
      (e.g. "blade edge of scraper", "brush head") from the already-cropped
      tool ROI. Uses a second LangSAM pass on the cropped image.

    Args:
        cropped_image:          np.ndarray (BGR) — output from extract_roi.
        functional_description: Text description of the functional sub-region
                                (e.g. "blade edge", "brush head").
        expansion_factor:       Fraction α for bounding box expansion.

    Returns:
        Same structure as extract_roi:
          "cropped_image", "bbox", "mask", "_fallback"
    """
    if not _CV2_AVAILABLE or cropped_image is None or cropped_image.size == 0:
        h, w = (cropped_image.shape[:2] if cropped_image is not None and cropped_image.size > 0
                else (64, 64))
        placeholder = np.zeros((h, w, 3), dtype=np.uint8) if cropped_image is None else cropped_image
        return _full_image_fallback(placeholder, "cv2 unavailable or empty input")

    h, w = cropped_image.shape[:2]

    if not _LANGSAM_AVAILABLE:
        print(f"[HRE] LangSAM unavailable — returning full crop as functional ROI "
              f"for '{functional_description}'")
        return _full_image_fallback(cropped_image, "LangSAM unavailable")

    try:
        from PIL import Image as PILImage
        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)
        pil_image = pil_image.resize((1024, 1024), PILImage.BILINEAR)

        model = _get_langsam()
        try:
            result = model.predict(pil_image, functional_description)
            if isinstance(result, tuple) and len(result) >= 2:
                masks, boxes = result[0], result[1]
            else:
                raise ValueError("Unexpected predict() return format")

            if masks is None or len(masks) == 0:
                return _full_image_fallback(cropped_image,
                                            "no mask found")
        except Exception as exc:
            print(f"[HRE] LangSAM predict failed: {exc}")
            print("[HRE] Try: pip install -U lang-sam")
            return _full_image_fallback(cropped_image, str(exc))

        mask_np = masks[0].cpu().numpy().astype(np.uint8) * 255
        box = boxes[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        x1, y1, x2, y2 = _expand_bbox(x1, y1, x2, y2, w, h, expansion_factor)

        sub_crop = cropped_image[y1:y2, x1:x2]
        mask_crop = mask_np[y1:y2, x1:x2]

        # Bilinear rescale to TARGET_SIZE (paper Section 3.3.1)
        sub_crop = cv2.resize(sub_crop, (TARGET_SIZE, TARGET_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(mask_crop, (TARGET_SIZE, TARGET_SIZE),
                               interpolation=cv2.INTER_NEAREST)

        return {
            "cropped_image": sub_crop,
            "bbox": [x1, y1, x2, y2],
            "mask": mask_crop,
            "_fallback": False
        }

    except Exception as exc:
        warnings.warn(
            f"[HRE] Stage 2 LangSAM failed: {exc}. Returning full crop as ROI.",
            RuntimeWarning
        )
        return _full_image_fallback(cropped_image, str(exc))
