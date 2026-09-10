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

import traceback
import warnings

import numpy as np

TARGET_SIZE = 512  # bilinear rescale target (paper Section 3.3.1)

# GroundingDINO detection thresholds. These are LangSAM's own defaults; they are
# named here because stage 2 (small functional sub-regions such as a blade edge)
# often needs a lower box threshold than stage 2's parent object did.
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

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
# LangSAM segmentation core
# ---------------------------------------------------------------------------

def _langsam_segment(image_bgr: np.ndarray, description: str,
                     expansion_factor: float, stage: str,
                     box_threshold: float = BOX_THRESHOLD,
                     text_threshold: float = TEXT_THRESHOLD) -> dict:
    """
    Run one text-grounded LangSAM pass and return the expanded-bbox crop.

    Shared by both HRE stages, which differ only in what they are pointed at.

    Installed LangSAM (0.2.1) contract, verified against the package source:

        predict(images_pil: list[Image], texts_prompt: list[str]) -> list[dict]

        each dict: {"boxes":  (N,4) xyxy np.ndarray,
                    "scores": (N,)  np.ndarray,
                    "labels": list[str],
                    "masks":  (N,H,W) np.ndarray, or [] when labels is empty,
                    "mask_scores": np.ndarray}

    Two things this function is deliberate about:

    1. Both arguments are lists and the return is a list of dicts. The previous
       code passed bare objects and unpacked a (masks, boxes) tuple, which
       raised on every call and was caught into a full-image fallback.

    2. The image is NOT resized before predict(). GDINO.predict post-processes
       with target_sizes taken from the PIL image it is handed, so boxes come
       back in that image's coordinate space. The previous code resized to
       1024x1024 and then indexed the original array with the returned boxes,
       so the crop landed in the wrong place. Passing native resolution keeps
       boxes, mask, and image_bgr in one coordinate space.

    masks and boxes arrive as numpy - LangSAM.predict already calls
    .cpu().numpy() internally - so no further conversion is needed.
    """
    from PIL import Image as PILImage

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(image_rgb)  # native resolution - see docstring

    model = _get_langsam()

    # GDINO and SAM are driven separately rather than through LangSAM.predict().
    #
    # LangSAM.predict() gates the SAM call on `if result["labels"]:`. When GDINO
    # finds nothing it can still return labels == [""] - a non-empty list holding
    # one empty string, which is truthy - so SAM is invoked with a (0, 4) box
    # array and dies on an internal assertion:
    #     sam2/modeling/sam/mask_decoder.py:203
    #     assert image_embeddings.shape[0] == tokens.shape[0]   # 1 != 0
    # Splitting the two steps lets "nothing detected" be reported as what it is
    # instead of surfacing as an AssertionError from deep inside SAM.
    gdino_results = model.gdino.predict([pil_image], [description],
                                        box_threshold, text_threshold)
    if not gdino_results:
        return _full_image_fallback(image_bgr, f"{stage}: GDINO returned no results")

    gd = {k: (v.cpu().numpy() if hasattr(v, "cpu") else v)
          for k, v in gdino_results[0].items()}
    boxes = np.asarray(gd.get("boxes", []))
    scores = np.asarray(gd.get("scores", []))

    if boxes.size == 0:
        return _full_image_fallback(
            image_bgr,
            f"{stage}: no detection for {description!r} "
            f"(box_threshold={box_threshold}, text_threshold={text_threshold})"
        )

    masks, _mask_scores, _ = model.sam.predict_batch([np.asarray(pil_image)],
                                                     xyxy=[boxes])
    masks = masks[0] if len(masks) else []
    if len(masks) == 0:
        return _full_image_fallback(
            image_bgr, f"{stage}: SAM returned no mask for {description!r}"
        )

    # predict() does not guarantee confidence ordering, so pick explicitly.
    best = int(np.argmax(scores)) if len(scores) == len(boxes) else 0

    mask_arr = np.squeeze(np.asarray(masks[best]))
    if mask_arr.ndim == 3:          # SAM can return a multimask stack
        mask_arr = mask_arr[0]
    mask_np = mask_arr.astype(bool).astype(np.uint8) * 255

    box = np.asarray(boxes[best]).astype(int)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1, x2, y2 = _expand_bbox(x1, y1, x2, y2, w, h, expansion_factor)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return _full_image_fallback(
            image_bgr, f"{stage}: degenerate bbox {[x1, y1, x2, y2]}"
        )

    cropped = image_bgr[y1:y2, x1:x2]
    mask_crop = mask_np[y1:y2, x1:x2]

    # Bilinear rescale to TARGET_SIZE (paper Section 3.3.1)
    cropped = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    mask_crop = cv2.resize(mask_crop, (TARGET_SIZE, TARGET_SIZE),
                           interpolation=cv2.INTER_NEAREST)

    coverage = float((mask_crop > 0).mean())
    print(f"[HRE] {stage}: {description!r} -> bbox {[x1, y1, x2, y2]} "
          f"score {float(scores[best]):.3f}, mask covers {coverage:.1%} of crop")

    return {
        "cropped_image": cropped,
        "bbox": [x1, y1, x2, y2],
        "mask": mask_crop,
        "_fallback": False,
        "_score": float(scores[best]) if len(scores) else None,
        "_mask_coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_roi(
    image_path: str,
    description: str,
    expansion_factor: float = 0.1,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD
) -> dict:
    """
    Stage 1: Segment object/tool from a full scene image using LangSAM.

    Paper Section 3.3.1 - Hierarchical Region Extraction (Stage 1):
      Uses text-grounded segmentation to find the object described by
      `description`, crops the bounding box (expanded by `expansion_factor`),
      and rescales the crop using bilinear interpolation.

    Args:
        image_path:        Path to the full scene or tool board image.
        description:       Natural-language description of the target
                           (e.g. "scraper tool", "white putty on metal plate").
        expansion_factor:  Fraction alpha by which to expand the bounding box
                           on each side to retain context (default 0.1 = 10%).

    Returns:
        dict with keys:
          "cropped_image" : np.ndarray  - ROI crop (BGR, uint8)
          "bbox"          : [x1, y1, x2, y2] in original image coords
          "mask"          : np.ndarray  - binary mask (uint8, 0/255)
          "_fallback"     : bool        - True if LangSAM was not used
    """
    if not _CV2_AVAILABLE:
        warnings.warn("[HRE] cv2 unavailable, returning placeholder ROI.", RuntimeWarning)
        h, w = 480, 640
        return _full_image_fallback(np.zeros((h, w, 3), dtype=np.uint8), "cv2 unavailable")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        warnings.warn(f"[HRE] Cannot read image: {image_path}. Returning empty fallback.",
                      RuntimeWarning)
        h, w = 480, 640
        return _full_image_fallback(np.zeros((h, w, 3), dtype=np.uint8),
                                    f"cannot read {image_path}")

    if not _LANGSAM_AVAILABLE:
        print(f"[HRE] LangSAM unavailable - returning full image as ROI for {description!r}")
        return _full_image_fallback(image_bgr, "LangSAM unavailable")

    try:
        return _langsam_segment(image_bgr, description, expansion_factor,
                                "stage 1", box_threshold, text_threshold)
    except Exception as exc:
        # Printed in full: an exception here previously became a silent
        # full-image ROI, which is the paper's w/o HRE ablation by accident.
        traceback.print_exc()
        warnings.warn(f"[HRE] Stage 1 failed: {exc}. Returning full image as ROI.",
                      RuntimeWarning)
        return _full_image_fallback(image_bgr,
                                    f"stage 1 raised {type(exc).__name__}: {exc}")


def extract_functional_region(
    cropped_image: np.ndarray,
    functional_description: str,
    expansion_factor: float = 0.1,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD
) -> dict:
    """
    Stage 2: Isolate the functional sub-region from a cropped tool image.

    Paper Section 3.3.1 - Hierarchical Region Extraction (Stage 2):
      Applied after Stage 1 to further isolate the tool's working surface
      (e.g. "blade edge of scraper", "brush head") from the already-cropped
      tool ROI. Uses a second LangSAM pass on the cropped image.

    Args:
        cropped_image:          np.ndarray (BGR) - output from extract_roi.
        functional_description: Text description of the functional sub-region
                                (e.g. "blade edge", "brush head").
        expansion_factor:       Fraction alpha for bounding box expansion.

    Returns:
        Same structure as extract_roi:
          "cropped_image", "bbox", "mask", "_fallback"
    """
    if not _CV2_AVAILABLE or cropped_image is None or cropped_image.size == 0:
        h, w = (cropped_image.shape[:2] if cropped_image is not None and cropped_image.size > 0
                else (64, 64))
        placeholder = (np.zeros((h, w, 3), dtype=np.uint8)
                       if cropped_image is None else cropped_image)
        return _full_image_fallback(placeholder, "cv2 unavailable or empty input")

    if not _LANGSAM_AVAILABLE:
        print(f"[HRE] LangSAM unavailable - returning full crop as functional ROI "
              f"for {functional_description!r}")
        return _full_image_fallback(cropped_image, "LangSAM unavailable")

    try:
        return _langsam_segment(cropped_image, functional_description,
                                expansion_factor, "stage 2",
                                box_threshold, text_threshold)
    except Exception as exc:
        traceback.print_exc()
        warnings.warn(f"[HRE] Stage 2 failed: {exc}. Returning full crop as ROI.",
                      RuntimeWarning)
        return _full_image_fallback(cropped_image,
                                    f"stage 2 raised {type(exc).__name__}: {exc}")
