"""
Phase 4: Offline trajectory smoothing.

Strategy:
  1. Clip physically impossible values (gripper width, Z floor).
  2. Linear-interpolate across detection gaps (fills invalid frames).
  3. Apply Savitzky-Golay over the full sequence for each spatial coordinate.
  4. Re-normalize unit vectors (grasp_axis, approach_vec) after smoothing.
  5. Record which frames were interpolated vs originally detected.

Savitzky-Golay is strictly better than per-frame EMA when the full sequence
is available: it is acausal, has no lag, and preserves peak shapes.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from src.tracking.trajectory import TrajectoryData


# ─────────────────────────────────────────────────────────────────────────────
# Physical bounds
# ─────────────────────────────────────────────────────────────────────────────

GRIPPER_WIDTH_MAX_M  = 0.15    # max realistic human hand opening (15 cm)
GRIPPER_WIDTH_MIN_M  = 0.0
Z_FLOOR_M            = -0.06   # allow up to 6 cm below table (calibration error margin)

# Large-gap threshold: don't interpolate across gaps longer than this.
# Gaps above this are left as the nearest valid boundary value (hold-last).
MAX_INTERPOLATE_FRAMES = 30    # 1 second at 30 fps


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interp_gaps(arr: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Fill invalid frames in arr (N, ...) by linear interpolation.
    Gaps longer than max_gap frames are filled by hold-last (nearest boundary).

    Args:
        arr:     (N,) or (N, D) float array
        valid:   (N,) bool mask — True = originally detected
        max_gap: gaps longer than this use boundary fill instead of lerp

    Returns:
        filled: same shape as arr, no NaN/invalid values
    """
    out = arr.copy().astype(np.float64)
    N = len(valid)
    i = 0
    while i < N:
        if valid[i]:
            i += 1
            continue

        # Find run of invalid frames
        j = i
        while j < N and not valid[j]:
            j += 1

        gap_len = j - i
        left_i  = i - 1   # last valid before gap (-1 if gap starts at 0)
        right_i = j        # first valid after gap  (N if gap ends at N)

        if left_i < 0 and right_i >= N:
            # Entire array invalid — leave zeros
            pass
        elif left_i < 0:
            # Gap at start — replicate first valid value
            out[i:j] = out[right_i]
        elif right_i >= N:
            # Gap at end — replicate last valid value
            out[i:j] = out[left_i]
        elif gap_len <= max_gap:
            # Short gap — linear interpolate
            t = np.linspace(0, 1, gap_len + 2)[1:-1]    # (gap_len,) in (0,1)
            if arr.ndim == 1:
                out[i:j] = out[left_i] + t * (out[right_i] - out[left_i])
            else:
                out[i:j] = (out[left_i][None, :] +
                            t[:, None] * (out[right_i] - out[left_i])[None, :])
        else:
            # Long gap — split at midpoint: left half holds left, right half holds right
            mid = i + gap_len // 2
            out[i:mid] = out[left_i]
            out[mid:j] = out[right_i]

        i = j

    return out


def _savgol(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Apply Savitzky-Golay to (N,) or (N, D) array. Window must be odd."""
    window = window if window % 2 == 1 else window + 1
    # Clamp window to sequence length
    window = min(window, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    window = max(window, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)

    if arr.ndim == 1:
        return savgol_filter(arr, window, poly).astype(np.float32)
    else:
        return np.stack(
            [savgol_filter(arr[:, d], window, poly) for d in range(arr.shape[1])],
            axis=1,
        ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main smoother
# ─────────────────────────────────────────────────────────────────────────────

def smooth_trajectory(
    raw: TrajectoryData,
    window: int = 15,
    polyorder: int = 3,
) -> TrajectoryData:
    """
    Smooth a raw trajectory.

    Args:
        raw:       raw TrajectoryData from extract_trajectory()
        window:    Savitzky-Golay window length (odd; ~0.5s at 30fps → 15)
        polyorder: polynomial order (3 is a good default)

    Returns:
        New TrajectoryData with smoothed values. The 'valid' mask is preserved
        (smoothed-only frames are still marked valid=False in the original sense,
        but all arrays are now fully populated via interpolation).
    """
    N     = raw.n_frames
    valid = raw.valid.copy()

    print(f"[Smoother] {N} frames  |  {valid.sum()} valid  |  window={window}  poly={polyorder}")

    # ── Step 1: clip physically impossible values ─────────────────────────────
    gw = raw.gripper_width.copy()
    gw = np.clip(gw, GRIPPER_WIDTH_MIN_M, GRIPPER_WIDTH_MAX_M)

    gc = raw.gripper_center.copy()
    gc[:, 2] = np.maximum(gc[:, 2], Z_FLOOR_M)   # Z floor

    tt = raw.thumb_tip.copy()
    it = raw.index_tip.copy()
    wr = raw.wrist.copy()
    ga = raw.grasp_axis.copy()
    av = raw.approach_vec.copy()

    # ── Step 2: interpolate across gaps ───────────────────────────────────────
    gc_i  = _interp_gaps(gc,  valid, MAX_INTERPOLATE_FRAMES)
    gw_i  = _interp_gaps(gw,  valid, MAX_INTERPOLATE_FRAMES)
    ga_i  = _interp_gaps(ga,  valid, MAX_INTERPOLATE_FRAMES)
    av_i  = _interp_gaps(av,  valid, MAX_INTERPOLATE_FRAMES)
    tt_i  = _interp_gaps(tt,  valid, MAX_INTERPOLATE_FRAMES)
    it_i  = _interp_gaps(it,  valid, MAX_INTERPOLATE_FRAMES)
    wr_i  = _interp_gaps(wr,  valid, MAX_INTERPOLATE_FRAMES)

    # ── Step 3: Savitzky-Golay smoothing ─────────────────────────────────────
    gc_s  = _savgol(gc_i,  window, polyorder)
    gw_s  = _savgol(gw_i,  window, polyorder).clip(0, GRIPPER_WIDTH_MAX_M)
    ga_s  = _savgol(ga_i,  window, polyorder)
    av_s  = _savgol(av_i,  window, polyorder)
    tt_s  = _savgol(tt_i,  window, polyorder)
    it_s  = _savgol(it_i,  window, polyorder)
    wr_s  = _savgol(wr_i,  window, polyorder)

    # ── Step 4: renormalize unit vectors ──────────────────────────────────────
    def _renorm(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (v / norms).astype(np.float32)

    ga_s = _renorm(ga_s)
    av_s = _renorm(av_s)

    # ── Step 5: Z floor re-apply after smoothing ──────────────────────────────
    gc_s[:, 2] = np.maximum(gc_s[:, 2], Z_FLOOR_M)

    gap_count = int((~valid).sum())
    interp_count = gap_count  # all invalid frames got interpolated
    print(f"[Smoother] Interpolated {interp_count} gap frames  |  "
          f"Longest gap: {_longest_gap(valid)} frames")

    return TrajectoryData({
        "gripper_center": gc_s,
        "gripper_width":  gw_s,
        "grasp_axis":     ga_s,
        "approach_vec":   av_s,
        "thumb_tip":      tt_s,
        "index_tip":      it_s,
        "wrist":          wr_s,
        "valid":          valid,          # preserve original detection mask
        "frame_indices":  raw.frame_indices,
        "fps":            raw.fps,
        "resolution":     raw.resolution,
        "video_path":     raw.video_path,
    })


def _longest_gap(valid: np.ndarray) -> int:
    max_gap = cur = 0
    for v in valid:
        cur = 0 if v else cur + 1
        max_gap = max(max_gap, cur)
    return max_gap
