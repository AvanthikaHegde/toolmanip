"""
Standalone RealSense D435 smoke test — NOT wired into the pipeline.

Verifies, in order:
  1. Device enumerates and reports USB 3.x (not the 2.1 fallback)
  2. Depth + color streams actually deliver frames
  3. Depth can be aligned into the color frame's pixel space
  4. Camera intrinsics (fx, fy, ppx, ppy) are readable
  5. The deprojection in structures/affordance_structures.py:78-81
     produces real metric XYZ from a (u, v) pixel

Run with the RealSense Viewer CLOSED — it holds an exclusive lock on the device.

    python test_realsense.py
"""

import sys
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

OUT_DIR = Path("results") / "camera_test"
WARMUP_FRAMES = 30      # let auto-exposure settle before trusting a frame
TEST_FRAMES = 30        # frames to grab for the throughput check
WIDTH, HEIGHT, FPS = 640, 480, 30


def check_device() -> bool:
    """Enumerate connected devices and report the negotiated USB speed."""
    ctx = rs.context()
    devices = list(ctx.query_devices())

    if not devices:
        print("[FAIL] No RealSense device found.")
        print("       Check the cable is plugged in and the Viewer is closed.")
        return False

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)

        print(f"[OK]   Device   : {name}")
        print(f"       Serial   : {serial}")
        print(f"       Firmware : {fw}")
        print(f"       USB type : {usb}")

        if usb.startswith("2"):
            print(f"[WARN] USB {usb} — this is the slow 2-lane fallback.")
            print("       Depth+color at 640x480@30 may still work, but higher")
            print("       profiles will return no frames. Check the cable/port.")
        else:
            print(f"[OK]   USB {usb} = SuperSpeed. Full bandwidth available.")

    return True


def main() -> int:
    print("=" * 62)
    print("RealSense D435 smoke test")
    print("=" * 62)

    if not check_device():
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- configure the two streams the pipeline will eventually need ----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print(f"\n[..]   Starting pipeline at {WIDTH}x{HEIGHT}@{FPS} ...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        print(f"[FAIL] Could not start pipeline: {exc}")
        print("       Most common cause: the RealSense Viewer is still open.")
        print("       Close it and re-run.")
        return 1

    try:
        # depth scale converts raw uint16 depth units -> metres
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"[OK]   Pipeline started. Depth scale = {depth_scale} m/unit")

        # align depth into the color frame so (u,v) means the same thing in both
        align = rs.align(rs.stream.color)

        print(f"[..]   Warming up ({WARMUP_FRAMES} frames, auto-exposure) ...")
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        print(f"[..]   Grabbing {TEST_FRAMES} frames ...")
        good = 0
        color_image = None
        depth_frame = None

        for _ in range(TEST_FRAMES):
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            good += 1
            color_image = np.asanyarray(color_frame.get_data())

        print(f"[{'OK' if good == TEST_FRAMES else 'WARN'}]   Received {good}/{TEST_FRAMES} complete frame pairs")
        if good == 0:
            print("[FAIL] No frames received — same failure the Viewer showed.")
            return 1

        depth_image = np.asanyarray(depth_frame.get_data())
        print(f"[OK]   Color array : {color_image.shape} {color_image.dtype}")
        print(f"[OK]   Depth array : {depth_image.shape} {depth_image.dtype}")

        # ---- intrinsics: exactly the fx/fy/cx/cy the pipeline TODO wants ----
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        print("\n--- INTRINSICS (color-aligned) ---")
        print(f"       width x height : {intr.width} x {intr.height}")
        print(f"       fx, fy         : {intr.fx:.3f}, {intr.fy:.3f}")
        print(f"       ppx, ppy (cx,cy): {intr.ppx:.3f}, {intr.ppy:.3f}")
        print(f"       model          : {intr.model}")
        print(f"       coeffs         : {[round(c, 5) for c in intr.coeffs]}")

        # ---- deprojection check against affordance_structures.py:78-81 ----
        u, v = intr.width // 2, intr.height // 2
        z = depth_frame.get_distance(u, v)   # metres, already scaled

        print(f"\n--- DEPROJECTION TEST at centre pixel ({u}, {v}) ---")
        if z == 0:
            print("[WARN] Depth at centre is 0 (invalid).")
            print("       Point the camera at a textured surface 0.3-3 m away.")
        else:
            # library implementation
            xyz_rs = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
            # the formula written in structures/affordance_structures.py
            x_manual = (u - intr.ppx) * z / intr.fx
            y_manual = (v - intr.ppy) * z / intr.fy

            print(f"       Z (depth)       : {z:.4f} m")
            print(f"       librealsense    : X={xyz_rs[0]:+.4f} Y={xyz_rs[1]:+.4f} Z={xyz_rs[2]:+.4f} (m)")
            print(f"       paper formula   : X={x_manual:+.4f} Y={y_manual:+.4f} Z={z:+.4f} (m)")
            agree = abs(x_manual - xyz_rs[0]) < 1e-6 and abs(y_manual - xyz_rs[1]) < 1e-6
            print(f"[{'OK' if agree else 'WARN'}]   Formulas agree  : {agree}")

        # ---- valid-depth coverage: how usable is this view ----
        valid = int(np.count_nonzero(depth_image))
        total = depth_image.size
        print(f"\n[OK]   Valid depth pixels: {valid}/{total} ({100.0 * valid / total:.1f}%)")
        nz = depth_image[depth_image > 0] * depth_scale
        if nz.size:
            print(f"       Depth range    : {nz.min():.3f} m -> {nz.max():.3f} m")

        # ---- save artifacts so the frames can be inspected afterwards ----
        import cv2
        color_path = OUT_DIR / "color.png"
        depth_raw_path = OUT_DIR / "depth_raw.npy"
        depth_vis_path = OUT_DIR / "depth_colorized.png"

        cv2.imwrite(str(color_path), color_image)
        np.save(str(depth_raw_path), depth_image)
        colorized = np.asanyarray(
            rs.colorizer().colorize(depth_frame).get_data()
        )
        cv2.imwrite(str(depth_vis_path), colorized)

        print(f"\n[OK]   Saved {color_path}")
        print(f"[OK]   Saved {depth_raw_path}  (uint16, multiply by {depth_scale} for metres)")
        print(f"[OK]   Saved {depth_vis_path}")

    finally:
        pipeline.stop()
        print("\n[OK]   Pipeline stopped cleanly.")

    print("=" * 62)
    print("PASS — camera is usable from Python.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
