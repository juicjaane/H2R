"""
Phase 1 validation: verify metric depth accuracy against a known physical distance.

HOW TO USE:
1. Place an object (e.g. a piece of paper, a ruler, a book) on your desk.
2. Measure the distance from the camera lens to that object with a tape measure.
3. Run this script; point the camera at the object; press SPACE to capture one reading.
4. The script prints the model's depth estimate. Compare to your tape measure.
5. Target: within ±10% of the true distance.

Controls:
    SPACE  — capture current reading and print result
    Q      — quit

Requires: metric checkpoint already downloaded (run download_metric_checkpoint.py first)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / ".."))

import cv2
import numpy as np

# Config sets up sys.path for DAV2 metric imports
import src.config as cfg
from src.tracking.depth_model import MetricDepthModel

WINDOW_RGB   = "Validation — RGB (point at your target, press SPACE)"
WINDOW_DEPTH = "Validation — Depth (meters)"


def run_validation():
    if not cfg.DEPTH_METRIC_CKPT.exists():
        print(f"ERROR: Metric checkpoint not found at {cfg.DEPTH_METRIC_CKPT}")
        print("Run:  python scripts/download_metric_checkpoint.py")
        sys.exit(1)

    print("Loading metric depth model...")
    model = MetricDepthModel()
    print("Model ready.\n")
    print("Instructions:")
    print("  1. Measure the distance from your camera to an object with a tape measure.")
    print("  2. Point the camera at that object.")
    print("  3. Press SPACE to capture a depth reading from the image centre.")
    print("  4. Compare the printed value to your tape measure.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    # Detect resolution for correct intrinsics
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        intr = cfg.get_intrinsics(w, h)
        print(f"Camera resolution: {w}x{h}  fx={intr['fx']} cx={intr['cx']}")
    except ValueError:
        print(f"Warning: no preset intrinsics for {w}x{h}; continuing anyway.")

    readings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Draw crosshair at centre
        cx, cy = w // 2, h // 2
        cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

        # HUD
        cv2.putText(display, "SPACE=capture  Q=quit", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if readings:
            last = readings[-1]
            cv2.putText(display, f"Last reading: {last:.3f} m", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(WINDOW_RGB, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord(' '):
            print("Running depth inference on current frame...")
            depth = model.infer_frame(frame)

            # Median of 11x11 patch at image centre (robust to edge noise)
            reading = model.sample_patch(depth, cx, cy, radius=5)
            readings.append(reading)

            # Depth visualisation
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            cv2.drawMarker(depth_vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.imshow(WINDOW_DEPTH, depth_vis)

            print(f"\n  Model depth at image centre: {reading:.4f} m  ({reading*100:.1f} cm)")
            print("  Compare this to your tape measure reading.")
            print("  Acceptable error: ± 10%")
            print()

    cap.release()
    cv2.destroyAllWindows()

    if readings:
        print(f"\nAll readings this session: {[f'{r:.3f}m' for r in readings]}")
        print(f"Mean: {np.mean(readings):.3f} m  |  Std: {np.std(readings):.3f} m")


if __name__ == "__main__":
    run_validation()
