"""
Phase 8: Composite 4-panel video renderer.

Layout (each panel PANEL_W × PANEL_H):
  ┌──────────────────────┬──────────────────────┐
  │  RGB + hand overlay  │  Depth (inferno)     │
  ├──────────────────────┼──────────────────────┤
  │  Top-down table view │  MuJoCo sim          │
  └──────────────────────┴──────────────────────┘
"""

from __future__ import annotations

import numpy as np
import cv2

# Panel dimensions — each panel at half the 1280×720 source resolution
PANEL_W = 640
PANEL_H = 360

# Depth clipping range for inferno colormap
DEPTH_NEAR_M = 0.2
DEPTH_FAR_M  = 2.5

# Colours (BGR)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GREEN  = (0,   220, 80)
_ORANGE = (0,   140, 255)
_RED    = (0,   60,  220)
_GREY   = (80,  80,  80)
_LABEL_BG = (30, 30, 30)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_panel(bgr: np.ndarray) -> np.ndarray:
    """Resize any BGR frame to PANEL_W × PANEL_H."""
    if bgr.shape[1] != PANEL_W or bgr.shape[0] != PANEL_H:
        bgr = cv2.resize(bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    return bgr


def add_label(frame: np.ndarray, text: str) -> None:
    """Draw a dark-background label in the top-left corner (in-place)."""
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    cv2.rectangle(frame, (0, 0), (tw + pad * 2, th + pad * 2 + 2), _LABEL_BG, -1)
    cv2.putText(frame, text, (pad, th + pad), font, scale, _WHITE, thickness, cv2.LINE_AA)


def tile_2x2(tl: np.ndarray, tr: np.ndarray,
             bl: np.ndarray, br: np.ndarray) -> np.ndarray:
    """Stack four PANEL_W×PANEL_H BGR frames into a 2×2 grid."""
    top = np.concatenate([tl, tr], axis=1)
    bot = np.concatenate([bl, br], axis=1)
    return np.concatenate([top, bot], axis=0)


# ── Panel 1: RGB + hand overlay ───────────────────────────────────────────────

def _backproject(
    points_table: np.ndarray,           # (N, 3) table frame
    T_table_to_cam: np.ndarray,         # (4, 4)
    fx: float, fy: float,
    cx: float, cy: float,
    w: int, h: int,
) -> list[tuple[int, int] | None]:
    """Back-project table-frame 3D points to pixel coordinates."""
    results = []
    for p in points_table:
        ph = np.append(p, 1.0)
        pc = T_table_to_cam @ ph          # camera frame
        Xc, Yc, Zc = pc[:3]
        if Zc <= 0.01:
            results.append(None)
            continue
        u = int(fx * Xc / Zc + cx)
        v = int(fy * Yc / Zc + cy)
        if 0 <= u < w and 0 <= v < h:
            results.append((u, v))
        else:
            results.append(None)
    return results


def make_rgb_panel(
    bgr_frame: np.ndarray,
    frame_data: dict,                   # keys: thumb_tip, index_tip, wrist, gripper_center (table frame)
    T_table_to_cam: np.ndarray,
    intrinsics: dict,
    valid: bool,
) -> np.ndarray:
    """
    Draw key hand landmarks on the RGB frame.
    frame_data values are (3,) in table frame.
    """
    out = bgr_frame.copy()
    h, w = out.shape[:2]
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    if valid:
        pts_3d = np.array([
            frame_data["thumb_tip"],
            frame_data["index_tip"],
            frame_data["wrist"],
            frame_data["gripper_center"],
        ])
        pts_2d = _backproject(pts_3d, T_table_to_cam, fx, fy, cx, cy, w, h)
        thumb, index, wrist, grip = pts_2d

        # Draw skeleton lines
        for a, b in [(wrist, thumb), (wrist, index), (thumb, index)]:
            if a and b:
                cv2.line(out, a, b, _WHITE, 2, cv2.LINE_AA)

        # Draw key points
        for pt, colour, r in [
            (thumb, _ORANGE, 6),
            (index, _GREEN,  6),
            (wrist, _WHITE,  5),
        ]:
            if pt:
                cv2.circle(out, pt, r, colour, -1, cv2.LINE_AA)
                cv2.circle(out, pt, r, _BLACK,  1, cv2.LINE_AA)

        # Gripper centre
        if grip:
            cv2.drawMarker(out, grip, _GREEN, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    panel = _to_panel(out)
    add_label(panel, "RGB + hand")
    return panel


# ── Panel 2: Depth colormap ───────────────────────────────────────────────────

def make_depth_panel(depth_m: np.ndarray) -> np.ndarray:
    """Convert metric depth array to inferno BGR image."""
    clipped   = np.clip(depth_m, DEPTH_NEAR_M, DEPTH_FAR_M)
    normed    = ((clipped - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M) * 255).astype(np.uint8)
    coloured  = cv2.applyColorMap(normed, cv2.COLORMAP_INFERNO)
    panel     = _to_panel(coloured)
    add_label(panel, f"Depth  {DEPTH_NEAR_M:.1f}-{DEPTH_FAR_M:.1f} m")
    return panel


def make_depth_placeholder() -> np.ndarray:
    """Grey panel when depth model is not available."""
    panel = np.full((PANEL_H, PANEL_W, 3), 40, dtype=np.uint8)
    cv2.putText(panel, "Depth (skipped)", (160, PANEL_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _GREY, 1, cv2.LINE_AA)
    add_label(panel, "Depth")
    return panel


# ── Panel 3: Top-down trajectory view ────────────────────────────────────────

def _table_to_img(
    xy: np.ndarray,          # (N, 2) or (2,) in table frame
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    pw: int, ph: int,
    pad: int = 30,
) -> np.ndarray:
    """Scale table XY coords to image pixel coords. Y flipped so near=bottom."""
    xy    = np.atleast_2d(xy).astype(float)
    rx    = (x_max - x_min) or 1.0
    ry    = (y_max - y_min) or 1.0
    ux    = (xy[:, 0] - x_min) / rx * (pw - 2 * pad) + pad
    uy    = (y_max - xy[:, 1]) / ry * (ph - 2 * pad) + pad   # flip Y
    return np.column_stack([ux, uy]).astype(int)


def make_topdown_panel(
    gripper_traj: np.ndarray,   # (N, 3) table frame, all frames
    valid_mask: np.ndarray,     # (N,) bool
    current_idx: int,
    base_pos: np.ndarray,       # (3,) table frame
    corners_table: np.ndarray,  # (4, 3) table frame
) -> np.ndarray:
    pw, ph = PANEL_W, PANEL_H
    pad    = 30
    canvas = np.full((ph, pw, 3), 20, dtype=np.uint8)

    # Compute view bounds — encompass trajectory + base + corners + margin
    all_xy = np.vstack([
        gripper_traj[valid_mask, :2],
        base_pos[:2].reshape(1, 2),
        corners_table[:, :2],
    ])
    margin = 0.1
    x_min = all_xy[:, 0].min() - margin
    x_max = all_xy[:, 0].max() + margin
    y_min = all_xy[:, 1].min() - margin
    y_max = all_xy[:, 1].max() + margin

    def to_img(xy):
        return _table_to_img(xy, x_min, x_max, y_min, y_max, pw, ph, pad)

    # Draw table outline
    corner_px = to_img(corners_table[:, :2]).reshape((-1, 1, 2))
    cv2.polylines(canvas, [corner_px], isClosed=True,
                  color=(100, 80, 60), thickness=1, lineType=cv2.LINE_AA)

    # Draw reach / deadzone circles around robot base
    base_px = to_img(base_pos[:2])[0]
    px_per_m = (pw - 2 * pad) / max(x_max - x_min, y_max - y_min)
    for radius_m, color, thick in [
        (0.855, (0, 100, 180), 1),   # outer reach
        (0.170, (0, 60,  150), 1),   # inner dead zone
    ]:
        r_px = int(radius_m * px_per_m)
        cv2.circle(canvas, tuple(base_px), r_px, color, thick, cv2.LINE_AA)

    # Draw trajectory path coloured by time (blue → red)
    valid_pts = np.where(valid_mask)[0]
    if len(valid_pts) > 1:
        traj_px = to_img(gripper_traj[valid_pts, :2])
        n       = len(traj_px)
        for k in range(n - 1):
            t   = k / max(n - 1, 1)
            col = (int(220 * (1 - t)), 60, int(220 * t))   # blue → red
            cv2.line(canvas, tuple(traj_px[k]), tuple(traj_px[k + 1]),
                     col, 1, cv2.LINE_AA)

    # Draw robot base marker
    cv2.drawMarker(canvas, tuple(base_px), _ORANGE,
                   cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)

    # Draw current frame position
    if valid_mask[current_idx]:
        cur_px = to_img(gripper_traj[current_idx, :2])[0]
        cv2.circle(canvas, tuple(cur_px), 6, _GREEN, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(cur_px), 6, _WHITE,  1, cv2.LINE_AA)

    # Axis labels
    cv2.putText(canvas, "X", (pw - 20, ph // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, _GREY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Y", (pw // 2, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, _GREY, 1, cv2.LINE_AA)

    add_label(canvas, "Top-down  (blue=start  red=end)")
    return canvas


# ── Panel 4: Sim frame ────────────────────────────────────────────────────────

def make_sim_panel(sim_frame_bgr: np.ndarray) -> np.ndarray:
    panel = _to_panel(sim_frame_bgr)
    add_label(panel, "MuJoCo sim")
    return panel


# ── Frame counter overlay ─────────────────────────────────────────────────────

def add_frame_counter(composite: np.ndarray, frame_idx: int, total: int,
                      fps: float) -> None:
    """Stamp frame number + timestamp in bottom-right (in-place)."""
    t_sec = frame_idx / fps
    text  = f"frame {frame_idx+1}/{total}  {t_sec:.2f}s"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    h, w = composite.shape[:2]
    x    = w - tw - 8
    y    = h - 8
    cv2.rectangle(composite, (x - 4, y - th - 4), (w, h), _LABEL_BG, -1)
    cv2.putText(composite, text, (x, y), font, scale, _WHITE, thick, cv2.LINE_AA)
