"""
Guided calibration UI — walks the user through touching each table corner with
their index fingertip, captures depth, and saves the calibration file.

Run via:  python scripts/calibrate.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.config import get_intrinsics, DATA_DIR
from src.tracking.depth_model import MetricDepthModel
from src.calibration.surface import (
    pixel_to_3d,
    fit_plane,
    build_table_transform,
    save_calibration,
    CORNER_NAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
_GREEN  = (0, 220, 0)
_YELLOW = (0, 220, 220)
_RED    = (0, 60, 220)
_WHITE  = (255, 255, 255)
_CYAN   = (220, 220, 0)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )


def _get_index_tip(hand_results, w: int, h: int) -> Optional[tuple[int, int]]:
    """Return (px, py) of landmark 8 (index fingertip), or None."""
    if not hand_results.multi_hand_landmarks:
        return None
    lm = hand_results.multi_hand_landmarks[0].landmark[8]
    px = int(np.clip(lm.x * w, 0, w - 1))
    py = int(np.clip(lm.y * h, 0, h - 1))
    return px, py


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

def _put(img, text: str, y: int, color=_WHITE, scale: float = 0.75, thickness: int = 2):
    cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_corner_markers(img, captured: list[tuple[int, int]]):
    """Draw green dots for already-captured corners."""
    labels = ["TL", "TR", "BR", "BL"]
    for i, (px, py) in enumerate(captured):
        cv2.circle(img, (px, py), 12, _GREEN, cv2.FILLED)
        cv2.putText(img, labels[i], (px + 14, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration routine
# ─────────────────────────────────────────────────────────────────────────────

MAX_RESIDUAL_WARN_M = 0.010   # 10 mm — warn if plane fit is worse than this
CAPTURE_FRAMES     = 20       # depth frames to median per corner (reduces shot noise ~4.5×)


def run_calibration(
    output_path: Optional[Path] = None,
    camera_index: int = 0,
) -> Path:
    """
    Interactive calibration.

    Args:
        output_path: where to save calibration.json (default: data/calibration.json)
        camera_index: OpenCV camera index

    Returns:
        Path to saved calibration file.
    """
    if output_path is None:
        output_path = DATA_DIR / "calibration.json"

    depth_model = MetricDepthModel()
    hands = _make_hands()
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    intrinsics = get_intrinsics(w, h)
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    resolution_str = f"{w}x{h}"

    captured_3d: list[np.ndarray]  = []   # 3D points in camera frame
    captured_2d: list[tuple[int, int]] = []  # pixel positions for display

    corner_idx   = 0      # which corner we're currently capturing (0–3)
    status_msg   = ""
    status_color = _WHITE

    # Live depth hint: updated every N frames using a fast single-frame estimate
    _live_depth_val: float = 0.0
    _live_depth_frame_count: int = 0
    LIVE_DEPTH_INTERVAL = 8   # update live depth every 8 frames (~4fps at 30fps cam)

    WINDOW = "Surface Calibration"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    print("\n=== Surface Calibration ===")
    print("Touch your INDEX FINGERTIP to each table corner in order.")
    print("Order: TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT")
    print("Press SPACE to capture each corner.  Press Q to abort.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb)
        index_tip = _get_index_tip(hand_results, w, h)

        # Draw hand skeleton
        if hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(display, hand_results.multi_hand_landmarks[0],
                                   mp_hands.HAND_CONNECTIONS)

        # Highlight index tip
        if index_tip:
            cv2.circle(display, index_tip, 14, _CYAN, 3)
            cv2.circle(display, index_tip, 5,  _CYAN, cv2.FILLED)

        # Draw already-captured corners
        _draw_corner_markers(display, captured_2d)

        # ── Live depth update (every N frames) ────────────────────────────
        _live_depth_frame_count += 1
        if index_tip and _live_depth_frame_count >= LIVE_DEPTH_INTERVAL:
            _live_depth_frame_count = 0
            try:
                _d = depth_model.infer_frame(frame)
                _live_depth_val = depth_model.sample_patch(_d, index_tip[0], index_tip[1], radius=5)
            except Exception:
                pass

        # ── Instruction panel ──────────────────────────────────────────────
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        if corner_idx < 4:
            corner_name = CORNER_NAMES[corner_idx]
            hand_state  = "HAND DETECTED" if index_tip else "NO HAND"
            hand_col    = _GREEN if index_tip else _RED
            depth_hint  = f"  live depth: {_live_depth_val:.3f}m" if (_live_depth_val > 0 and index_tip) else ""
            _put(display, f"Step {corner_idx+1}/4  Touch index tip to: {corner_name}",
                 30, _YELLOW, 0.80)
            _put(display, f"Status: {hand_state}{depth_hint}   SPACE = capture   Q = abort",
                 62, hand_col, 0.62)
            _put(display, "TIP: Camera 0.8-1.2m from table gives best accuracy.",
                 92, (160, 160, 160), 0.55)
            if status_msg:
                _put(display, status_msg, 118, status_color, 0.52)
        else:
            _put(display, "All 4 corners captured! Computing plane...", 55, _GREEN, 0.85)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Calibration aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return None

        if key == ord(' ') and corner_idx < 4:
            if not index_tip:
                status_msg  = "No hand detected — move index finger into view first."
                status_color = _RED
                continue

            px, py = index_tip
            corner_name = CORNER_NAMES[corner_idx]
            print(f"Capturing corner {corner_idx+1} ({corner_name}) — collecting {CAPTURE_FRAMES} frames...",
                  end=" ", flush=True)

            # ── Multi-frame capture: hold finger still, median across frames ──
            depth_samples: list[float]          = []
            pixel_xs:      list[int]            = []
            pixel_ys:      list[int]            = []

            for frame_i in range(CAPTURE_FRAMES):
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                frame2 = cv2.flip(frame2, 1)

                # Track finger position in each capture frame
                rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                hr2  = hands.process(rgb2)
                tip2 = _get_index_tip(hr2, w, h)

                if tip2:
                    px2, py2 = tip2
                else:
                    px2, py2 = px, py   # fallback: use last known position

                depth2 = depth_model.infer_frame(frame2)
                d2     = depth_model.sample_patch(depth2, px2, py2, radius=5)

                if d2 > 0.05:
                    depth_samples.append(d2)
                    pixel_xs.append(px2)
                    pixel_ys.append(py2)

                # Live countdown overlay
                countdown_frame = frame2.copy()
                pct  = int((frame_i + 1) / CAPTURE_FRAMES * 100)
                msg  = f"Hold still... {frame_i+1}/{CAPTURE_FRAMES}"
                _put(countdown_frame, msg, 55, _YELLOW, 0.80)
                bar_w = int((w - 40) * (frame_i + 1) / CAPTURE_FRAMES)
                cv2.rectangle(countdown_frame, (20, 70), (20 + bar_w, 90), _GREEN, cv2.FILLED)
                cv2.rectangle(countdown_frame, (20, 70), (w - 20, 90), _WHITE, 2)
                if tip2:
                    cv2.circle(countdown_frame, tip2, 14, _CYAN, 3)
                cv2.imshow(WINDOW, countdown_frame)
                cv2.waitKey(1)

            if len(depth_samples) < CAPTURE_FRAMES // 2:
                status_msg  = f"Too many bad depth readings ({len(depth_samples)}/{CAPTURE_FRAMES}) — try again."
                status_color = _RED
                print("REJECTED")
                continue

            depth_val = float(np.median(depth_samples))
            px_med    = int(np.median(pixel_xs))
            py_med    = int(np.median(pixel_ys))

            point_3d = pixel_to_3d(px_med, py_med, depth_val, fx, fy, cx, cy)
            captured_3d.append(point_3d)
            captured_2d.append((px_med, py_med))
            corner_idx += 1

            depth_std  = float(np.std(depth_samples))
            status_msg  = (f"Corner {corner_idx} captured  "
                           f"depth={depth_val:.3f}m (std={depth_std*1000:.1f}mm)  "
                           f"3D=({point_3d[0]:.3f},{point_3d[1]:.3f},{point_3d[2]:.3f})")
            status_color = _GREEN
            print(f"OK  depth={depth_val:.3f}m  std={depth_std*1000:.1f}mm  "
                  f"n={len(depth_samples)}  point={point_3d.round(4)}")

            if corner_idx == 4:
                break   # all done

    cap.release()
    cv2.destroyAllWindows()

    if corner_idx < 4:
        print("Not enough corners captured. Aborting.")
        return None

    # ── Compute plane ──────────────────────────────────────────────────────
    corners_arr = np.array(captured_3d)   # (4, 3)
    plane = fit_plane(corners_arr)
    T = build_table_transform(corners_arr, plane)

    print(f"\n[Calibration] Plane normal: {plane['normal'].round(4)}")
    print(f"[Calibration] Residuals (m): {plane['residuals_m'].round(5)}")
    print(f"[Calibration] Max residual: {plane['max_residual_m']*1000:.2f} mm")

    if plane["max_residual_m"] > MAX_RESIDUAL_WARN_M:
        print(f"\n⚠  WARNING: max residual {plane['max_residual_m']*1000:.1f} mm > {MAX_RESIDUAL_WARN_M*1000:.0f} mm threshold.")
        print("   The captured corners do not lie on a very flat plane.")
        print("   This could be caused by depth noise or an uneven surface.")
        print("   Re-run calibration for better accuracy, or proceed if acceptable.\n")
    else:
        print(f"[Calibration] Plane fit quality: GOOD (max residual {plane['max_residual_m']*1000:.2f} mm)\n")

    # ── Save ──────────────────────────────────────────────────────────────
    save_calibration(corners_arr, plane, T, intrinsics, resolution_str, output_path)
    return output_path
