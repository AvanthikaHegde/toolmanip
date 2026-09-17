"""
RealSense D435 capture layer for ToolManip.

Standalone module — nothing in the existing pipeline imports it yet.

Provides the one thing Modules I and II are missing: a real frame plus the
metric information needed to turn a pixel into a 3D point.

A capture bundle is four files written together, because a colour image on its
own cannot be deprojected later:

    color.png        BGR8, what the VLM sees
    depth.npy        uint16, depth ALIGNED to the colour frame
    depth_vis.png    colorised depth, for eyeballing only
    intrinsics.json  fx, fy, ppx, ppy, depth_scale, resolution

Depth is aligned into the colour frame at capture time, so a pixel (u, v)
means the same thing in both arrays. Without that alignment every deprojected
point is wrong in a way that still looks plausible.

CLI:
    python camera.py                 # capture -> captures/<timestamp>/ + captures/latest/
    python camera.py --preview       # show the frame before saving
    python camera.py --out examples/tool_live.png   # also copy the colour frame here

Library:
    from camera import capture, deproject_pixel, load_capture
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

CAPTURES_DIR = Path("captures")
LATEST_DIR = CAPTURES_DIR / "latest"

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_WARMUP = 30          # frames discarded so auto-exposure settles
DEFAULT_PATCH = 5            # odd; window side used for robust depth sampling

# Profile that fits inside USB 2.1 bandwidth. Only used with allow_usb2=True;
# at 640x480@30 a USB 2.1 link accepts the stream and then delivers no frames.
USB2_WIDTH, USB2_HEIGHT, USB2_FPS = 424, 240, 15

# The link speed is renegotiated on every enumeration and a marginal cable can
# train SuperSpeed on one attempt and fall back on the next, so a 2.1 reading
# is retried before it is believed.
USB_RETRIES = 3


class CameraError(RuntimeError):
    """Raised when the camera cannot deliver a usable frame."""


@dataclass
class Intrinsics:
    """Pinhole parameters of the colour stream that depth was aligned into."""
    width: int
    height: int
    fx: float
    fy: float
    ppx: float          # principal point x — the 'cx' in the paper's formula
    ppy: float          # principal point y — the 'cy'
    model: str
    coeffs: list
    depth_scale: float  # multiply raw uint16 depth by this to get metres

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "fx": self.fx, "fy": self.fy,
            "ppx": self.ppx, "ppy": self.ppy,
            "model": self.model, "coeffs": list(self.coeffs),
            "depth_scale": self.depth_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(**d)


@dataclass
class Capture:
    """One aligned colour+depth frame and everything needed to deproject it."""
    color: np.ndarray          # (H, W, 3) uint8 BGR
    depth: np.ndarray          # (H, W)    uint16, aligned to colour
    intrinsics: Intrinsics
    timestamp: str

    @property
    def valid_depth_fraction(self) -> float:
        return float(np.count_nonzero(self.depth)) / self.depth.size


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def _negotiated_usb(retries: int = USB_RETRIES) -> str:
    """
    Read the negotiated USB speed, re-querying on a 2.x result.

    A marginal cable does not fail consistently: the same physical setup trains
    SuperSpeed on one enumeration and drops to 2.1 on the next. Re-querying
    distinguishes a genuinely USB-2-only path from a flapping one.
    """
    if rs is None:
        raise CameraError("pyrealsense2 is not installed. pip install pyrealsense2")

    usb = ""
    for attempt in range(retries):
        devices = list(rs.context().query_devices())
        if not devices:
            raise CameraError(
                "No RealSense device found. Check the cable is connected, and "
                "that it is a USB 3 cable plugged into a USB 3 port."
            )
        usb = devices[0].get_info(rs.camera_info.usb_type_descriptor)
        if not usb.startswith("2"):
            if attempt:
                print(f"[camera] Link trained to USB {usb} on attempt {attempt + 1} "
                      f"— the connection is intermittent, reseat the cable.")
            return usb
    return usb


def capture(width: int = DEFAULT_WIDTH,
            height: int = DEFAULT_HEIGHT,
            fps: int = DEFAULT_FPS,
            warmup: int = DEFAULT_WARMUP,
            allow_usb2: bool = False) -> Capture:
    """
    Grab one aligned colour+depth frame from the first connected D435.

    Raises CameraError with an actionable message rather than letting the
    librealsense exception through — every common failure here (no device, the
    Viewer holding the device, a USB 2 link starving the stream) otherwise
    surfaces as an opaque RuntimeError.

    On a USB 2.1 link the requested profile is refused rather than attempted,
    because it does not fail cleanly: the pipeline starts, reports success, and
    then no frame ever arrives. Pass allow_usb2=True to drop to a profile that
    does fit, at reduced resolution.
    """
    usb = _negotiated_usb()

    if usb.startswith("2"):
        if not allow_usb2:
            raise CameraError(
                f"Camera negotiated USB {usb}, not SuperSpeed.\n"
                f"  At this speed {width}x{height}@{fps} starts but delivers no frames.\n"
                f"  Fix the link: reseat the cable, use a USB 3 port (blue / 'SS'),\n"
                f"  and avoid passive extensions.\n"
                f"  To capture anyway at {USB2_WIDTH}x{USB2_HEIGHT}@{USB2_FPS}, "
                f"pass allow_usb2=True (CLI: --allow-usb2)."
            )
        width, height, fps = USB2_WIDTH, USB2_HEIGHT, USB2_FPS
        print(f"[camera] USB {usb}: falling back to {width}x{height}@{fps}. "
              f"Fewer depth samples per tool — expect coarser keypoint depth.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise CameraError(
            f"Could not start the camera pipeline: {exc}\n"
            f"  The usual cause is that the RealSense Viewer is still open — "
            f"only one process can stream from the device at a time."
        ) from exc

    try:
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # Align depth INTO the colour frame. Every downstream (u, v) comes from
        # the colour image, so depth must be resampled to match it.
        align = rs.align(rs.stream.color)

        try:
            for _ in range(warmup):
                pipeline.wait_for_frames()
            frames = align.process(pipeline.wait_for_frames())
        except RuntimeError as exc:
            # "Frame didn't arrive within 5000" — the stream was accepted but
            # nothing is being delivered. Always a link or ownership problem,
            # never a configuration one, since start() already validated it.
            raise CameraError(
                f"Camera accepted {width}x{height}@{fps} but delivered no frames "
                f"({exc}).\n"
                f"  Negotiated link: USB {usb}\n"
                f"  Check, in order: the RealSense Viewer is closed (it takes "
                f"exclusive ownership);\n"
                f"  the cable is seated at both ends; the port is USB 3."
            ) from exc

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise CameraError(
                "Pipeline started but delivered an incomplete frame pair. "
                "If this repeats, close the RealSense Viewer and retry."
            )

        intr_rs = depth_frame.profile.as_video_stream_profile().intrinsics
        intrinsics = Intrinsics(
            width=intr_rs.width, height=intr_rs.height,
            fx=intr_rs.fx, fy=intr_rs.fy,
            ppx=intr_rs.ppx, ppy=intr_rs.ppy,
            model=str(intr_rs.model), coeffs=list(intr_rs.coeffs),
            depth_scale=depth_scale,
        )

        return Capture(
            color=np.asanyarray(color_frame.get_data()).copy(),
            depth=np.asanyarray(depth_frame.get_data()).copy(),
            intrinsics=intrinsics,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    finally:
        pipeline.stop()


# ---------------------------------------------------------------------------
# Deprojection
# ---------------------------------------------------------------------------

def deproject_pixel(depth: np.ndarray,
                    u: int,
                    v: int,
                    intrinsics: Intrinsics,
                    patch: int = DEFAULT_PATCH) -> Optional[tuple]:
    """
    Convert a FULL-FRAME pixel (u, v) to metric (X, Y, Z) in camera coords.

    Implements the formula recorded in structures/affordance_structures.py:78
        X = (u - ppx) * Z / fx
        Y = (v - ppy) * Z / fy

    Depth is sampled as the MEDIAN of non-zero values in a patch x patch window
    rather than the single pixel: a lone pixel is routinely 0 (invalid) on tool
    edges and specular metal, and a single reading there silently becomes the
    origin. Returns None if the whole window is invalid — the caller must
    decide what an unmeasurable keypoint means, rather than receiving (0,0,0).

    (u, v) MUST be in full-frame coordinates. Keypoints coming out of HRE+SIF
    are in ROI space and have to have both crop offsets added first.
    """
    h, w = depth.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError(f"pixel ({u}, {v}) is outside the {w}x{h} depth frame")

    r = max(0, patch // 2)
    window = depth[max(0, v - r): v + r + 1, max(0, u - r): u + r + 1]
    valid = window[window > 0]
    if valid.size == 0:
        return None

    z = float(np.median(valid)) * intrinsics.depth_scale
    x = (u - intrinsics.ppx) * z / intrinsics.fx
    y = (v - intrinsics.ppy) * z / intrinsics.fy
    return (x, y, z)


def project_point(xyz: tuple, intrinsics: Intrinsics) -> tuple:
    """
    Inverse of deproject_pixel: metric (X, Y, Z) back to pixel (u, v).
    Used for the reprojection round-trip check.
    """
    x, y, z = xyz
    if z <= 0:
        raise ValueError(f"cannot project a point with Z={z}")
    u = x * intrinsics.fx / z + intrinsics.ppx
    v = y * intrinsics.fy / z + intrinsics.ppy
    return (u, v)


def distance_between(depth: np.ndarray,
                     p1: tuple,
                     p2: tuple,
                     intrinsics: Intrinsics,
                     patch: int = DEFAULT_PATCH) -> Optional[float]:
    """
    Metric distance in metres between two full-frame pixels.

    This is the strongest single verification available: measure a tool
    physically, then compare. A correct value confirms intrinsics, depth scale,
    alignment and crop offsets all at once. Returns None if either pixel has no
    valid depth.
    """
    a = deproject_pixel(depth, p1[0], p1[1], intrinsics, patch)
    b = deproject_pixel(depth, p2[0], p2[1], intrinsics, patch)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def colorize_depth(depth: np.ndarray,
                   depth_range: Optional[tuple] = None,
                   depth_scale: float = 0.001) -> np.ndarray:
    """
    Render depth for human inspection. Display only — depth.npy is the truth.

    Default is histogram equalisation, which is what the RealSense Viewer does
    and the only thing that works on a scene like a workbench. Linear scaling
    fails badly here: the frame spans 0.4 m (the board) to 20+ m (whatever is
    visible past it), so a linear map spends almost the entire colormap on
    empty far space and renders every tool the same shade. Equalisation
    allocates colour by pixel population instead, so the densely occupied
    working distance gets the resolution.

    Pass depth_range=(lo_m, hi_m) to force a linear map over a known band
    instead, when colours need to mean a fixed distance rather than maximise
    contrast.

    Invalid (zero) pixels render black, so holes stay distinct from near
    surfaces — under equalisation they would otherwise map to the same end of
    the colormap as the closest real geometry.
    """
    import cv2

    valid_mask = depth > 0
    if not valid_mask.any():
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)

    if depth_range is not None:
        lo = max(1.0, depth_range[0] / depth_scale)
        hi = max(lo + 1.0, depth_range[1] / depth_scale)
        norm = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)
    else:
        # CDF over valid depths only, used as a 16-bit -> 8-bit lookup table.
        hist = np.bincount(depth[valid_mask].ravel(), minlength=65536)
        cdf = np.cumsum(hist).astype(np.float64)
        cdf /= cdf[-1]
        gray = (cdf * 255).astype(np.uint8)[depth]

    vis = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    vis[~valid_mask] = (0, 0, 0)
    return vis


def save_capture(cap: Capture, out_dir: Path) -> Path:
    """Write the four-file capture bundle to out_dir. Returns out_dir."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "color.png"), cap.color)
    np.save(str(out_dir / "depth.npy"), cap.depth)
    cv2.imwrite(str(out_dir / "depth_vis.png"), colorize_depth(cap.depth))

    meta = cap.intrinsics.to_dict()
    meta["timestamp"] = cap.timestamp
    meta["valid_depth_fraction"] = round(cap.valid_depth_fraction, 4)
    with open(out_dir / "intrinsics.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def load_capture(in_dir: Path) -> Capture:
    """
    Read a capture bundle back. Lets every downstream step be re-run offline
    against the exact frame that produced a result, with no camera attached.
    """
    import cv2

    in_dir = Path(in_dir)
    with open(in_dir / "intrinsics.json") as f:
        meta = json.load(f)

    timestamp = meta.pop("timestamp", "")
    meta.pop("valid_depth_fraction", None)

    return Capture(
        color=cv2.imread(str(in_dir / "color.png")),
        depth=np.load(str(in_dir / "depth.npy")),
        intrinsics=Intrinsics.from_dict(meta),
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one aligned RealSense frame.")
    parser.add_argument("--preview", action="store_true",
                        help="show the captured frame before saving (any key to close)")
    parser.add_argument("--out", metavar="PATH",
                        help="also copy the colour frame to this path")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--allow-usb2", action="store_true",
                        help=f"capture at {USB2_WIDTH}x{USB2_HEIGHT}@{USB2_FPS} "
                             f"if the link is only USB 2.1")
    args = parser.parse_args()

    try:
        cap = capture(args.width, args.height, args.fps, args.warmup,
                      allow_usb2=args.allow_usb2)
    except CameraError as exc:
        print(f"[camera] ERROR: {exc}")
        return 1

    intr = cap.intrinsics
    print(f"[camera] Captured {intr.width}x{intr.height} at {cap.timestamp}")
    print(f"[camera] fx={intr.fx:.3f} fy={intr.fy:.3f} "
          f"ppx={intr.ppx:.3f} ppy={intr.ppy:.3f}")
    print(f"[camera] depth_scale={intr.depth_scale} m/unit")
    print(f"[camera] valid depth: {100 * cap.valid_depth_fraction:.1f}%")

    if cap.valid_depth_fraction < 0.5:
        print("[camera] WARNING: over half the frame has no depth. Check the "
              "tools are 0.3-3 m away and the surface is not strongly specular.")

    # Centre-pixel deprojection as an immediate smoke check on the bundle.
    cu, cv_ = intr.width // 2, intr.height // 2
    xyz = deproject_pixel(cap.depth, cu, cv_, intr)
    if xyz is None:
        print(f"[camera] centre pixel ({cu}, {cv_}): no valid depth")
    else:
        print(f"[camera] centre pixel ({cu}, {cv_}) -> "
              f"X={xyz[0]:+.4f} Y={xyz[1]:+.4f} Z={xyz[2]:+.4f} m")

    if args.preview:
        import cv2
        cv2.imshow("colour (any key to continue)", cap.color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    stamped = save_capture(cap, CAPTURES_DIR / cap.timestamp)
    print(f"[camera] Saved bundle -> {stamped}")

    # 'latest' is a plain copy, not a symlink: Windows needs elevation for those.
    if LATEST_DIR.exists():
        shutil.rmtree(LATEST_DIR)
    shutil.copytree(stamped, LATEST_DIR)
    print(f"[camera] Mirrored     -> {LATEST_DIR}")

    if args.out:
        import cv2
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cap.color)
        print(f"[camera] Colour frame -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
