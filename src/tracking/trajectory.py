"""
Offline trajectory extraction from a recorded video.

For each frame:
  1. Run MediaPipe to get 2D landmark pixel positions.
  2. Run metric DAV2 to get per-pixel depth in metres.
  3. Back-project each landmark to 3D using camera intrinsics.
  4. Transform 3D positions from camera frame to table frame.
  5. Compute hand state: gripper centre, width, grasp axis, approach vector.

Output: a dict saved as .npy (compressed) with keys documented in TrajectoryData.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import get_intrinsics
from src.calibration.surface import (
    pixel_to_3d,
    cam_to_table,
    load_calibration,
)
from src.tracking.depth_model import MetricDepthModel
from src.tracking.hand_tracker import (
    HandTracker,
    LM_WRIST, LM_THUMB_TIP, LM_INDEX_MCP, LM_INDEX_TIP, LM_PINKY_MCP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame hand state (table frame)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _compute_hand_state(
    landmarks_3d_cam: dict[int, np.ndarray],
    T_cam_to_table: np.ndarray,
) -> dict:
    """
    Given a dict of landmark 3D positions in CAMERA frame, compute the full
    hand state expressed in TABLE frame.
    """
    def t(key) -> np.ndarray:
        return cam_to_table(landmarks_3d_cam[key], T_cam_to_table)

    thumb_tip   = t(LM_THUMB_TIP)
    index_tip   = t(LM_INDEX_TIP)
    index_mcp   = t(LM_INDEX_MCP)
    pinky_mcp   = t(LM_PINKY_MCP)
    wrist       = t(LM_WRIST)

    gripper_center = (thumb_tip + index_tip) * 0.5
    gripper_width  = float(np.linalg.norm(index_tip - thumb_tip))

    # Grasp axis: thumb → index (maps to gripper X-axis)
    grasp_axis = _normalize(index_tip - thumb_tip)

    # Palm normal: cross(index_MCP−wrist, pinky_MCP−wrist)
    palm_a   = index_mcp - wrist
    palm_b   = pinky_mcp - wrist
    approach = _normalize(np.cross(palm_a, palm_b))

    return {
        "gripper_center": gripper_center,
        "gripper_width":  gripper_width,
        "grasp_axis":     grasp_axis,
        "approach_vec":   approach,
        "thumb_tip":      thumb_tip,
        "index_tip":      index_tip,
        "wrist":          wrist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory data container
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryData:
    """
    Holds per-frame hand state extracted from a video.

    All positions in TABLE frame (metres).
    Shape of each array: (N_frames, ...) where N_frames is total frame count.

    Arrays:
        gripper_center  (N, 3)   midpoint of thumb+index tips
        gripper_width   (N,)     metres, 0–~0.1
        grasp_axis      (N, 3)   unit vector thumb→index
        approach_vec    (N, 3)   palm normal (unit vector)
        thumb_tip       (N, 3)
        index_tip       (N, 3)
        wrist           (N, 3)
        valid           (N,)     bool — False means no hand detected this frame
        frame_indices   (N,)     int  — original frame number in the video
    """

    def __init__(self, data: dict):
        self.gripper_center = data["gripper_center"]  # (N,3)
        self.gripper_width  = data["gripper_width"]   # (N,)
        self.grasp_axis     = data["grasp_axis"]      # (N,3)
        self.approach_vec   = data["approach_vec"]    # (N,3)
        self.thumb_tip      = data["thumb_tip"]       # (N,3)
        self.index_tip      = data["index_tip"]       # (N,3)
        self.wrist          = data["wrist"]           # (N,3)
        self.valid          = data["valid"]           # (N,)  bool
        self.frame_indices  = data["frame_indices"]   # (N,)  int
        self.fps            = float(data["fps"])
        self.resolution     = str(data["resolution"])
        self.video_path     = str(data["video_path"])

    @property
    def n_frames(self) -> int:
        return len(self.valid)

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            gripper_center=self.gripper_center,
            gripper_width=self.gripper_width,
            grasp_axis=self.grasp_axis,
            approach_vec=self.approach_vec,
            thumb_tip=self.thumb_tip,
            index_tip=self.index_tip,
            wrist=self.wrist,
            valid=self.valid,
            frame_indices=self.frame_indices,
            fps=np.array([self.fps]),
            resolution=np.array([self.resolution]),
            video_path=np.array([self.video_path]),
        )
        print(f"[Trajectory] Saved {self.n_frames} frames ({self.n_valid} valid) to {path}")

    @staticmethod
    def load(path: Path) -> "TrajectoryData":
        d = np.load(path, allow_pickle=True)
        return TrajectoryData({
            "gripper_center": d["gripper_center"],
            "gripper_width":  d["gripper_width"],
            "grasp_axis":     d["grasp_axis"],
            "approach_vec":   d["approach_vec"],
            "thumb_tip":      d["thumb_tip"],
            "index_tip":      d["index_tip"],
            "wrist":          d["wrist"],
            "valid":          d["valid"],
            "frame_indices":  d["frame_indices"],
            "fps":            float(d["fps"][0]),
            "resolution":     str(d["resolution"][0]),
            "video_path":     str(d["video_path"][0]),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_LANDMARKS = [LM_WRIST, LM_THUMB_TIP, LM_INDEX_MCP, LM_INDEX_TIP, LM_PINKY_MCP]


def extract_trajectory(
    video_path: Path,
    calibration_path: Path,
    output_path: Optional[Path] = None,
    depth_model: Optional[MetricDepthModel] = None,
    show_preview: bool = False,
) -> TrajectoryData:
    """
    Process a recorded video and extract the full hand trajectory.

    Args:
        video_path:       path to the recorded .mp4
        calibration_path: path to calibration.json (from Phase 2)
        output_path:      where to save .npz (None = don't save)
        depth_model:      pre-loaded MetricDepthModel (loads fresh if None)
        show_preview:     display tracking overlay while processing

    Returns:
        TrajectoryData
    """
    # ── Load resources ────────────────────────────────────────────────────────
    calib = load_calibration(calibration_path)
    T     = calib["T_cam_to_table"]

    if depth_model is None:
        depth_model = MetricDepthModel()

    tracker = HandTracker()

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resolution_str = f"{w}x{h}"

    intr = get_intrinsics(w, h)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    print(f"[Trajectory] Video: {video_path.name}  {w}x{h}  {fps:.1f}fps  {total_frames} frames")
    print(f"[Trajectory] Calibration: {calibration_path.name}  "
          f"max_residual={calib['max_residual_m']*1000:.1f}mm")

    # ── Pre-allocate output buffers ───────────────────────────────────────────
    gc  = np.zeros((total_frames, 3), dtype=np.float32)   # gripper_center
    gw  = np.zeros( total_frames,    dtype=np.float32)    # gripper_width
    ga  = np.zeros((total_frames, 3), dtype=np.float32)   # grasp_axis
    av  = np.zeros((total_frames, 3), dtype=np.float32)   # approach_vec
    tt  = np.zeros((total_frames, 3), dtype=np.float32)   # thumb_tip
    it  = np.zeros((total_frames, 3), dtype=np.float32)   # index_tip
    wr  = np.zeros((total_frames, 3), dtype=np.float32)   # wrist
    vld = np.zeros( total_frames,    dtype=bool)
    fi  = np.arange(total_frames,    dtype=np.int32)

    frame_num    = 0
    detected     = 0
    gap_start    = None
    max_gap      = 0

    print(f"[Trajectory] Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= total_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Hand detection ────────────────────────────────────────────────────
        landmarks_2d = tracker.process(rgb)

        if landmarks_2d is None or not all(k in landmarks_2d for k in REQUIRED_LANDMARKS):
            # No valid detection this frame
            vld[frame_num] = False
            if gap_start is None:
                gap_start = frame_num
            frame_num += 1

            if show_preview:
                cv2.putText(frame, "NO HAND", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)
                cv2.imshow("Trajectory Extraction", frame)
                cv2.waitKey(1)
            continue

        # Track longest gap
        if gap_start is not None:
            max_gap = max(max_gap, frame_num - gap_start)
            gap_start = None

        # ── Depth inference ───────────────────────────────────────────────────
        depth = depth_model.infer_frame(frame)

        # ── Back-project landmarks to 3D (camera frame) ───────────────────────
        landmarks_3d_cam: dict[int, np.ndarray] = {}
        for lm_id in REQUIRED_LANDMARKS:
            px, py = landmarks_2d[lm_id]
            d = depth_model.sample_patch(depth, px, py, radius=2)
            if d <= 0.05:
                d = 0.5   # fallback: 0.5m if depth is invalid
            landmarks_3d_cam[lm_id] = pixel_to_3d(px, py, d, fx, fy, cx, cy)

        # ── Compute hand state in table frame ─────────────────────────────────
        state = _compute_hand_state(landmarks_3d_cam, T)

        gc[frame_num]  = state["gripper_center"]
        gw[frame_num]  = state["gripper_width"]
        ga[frame_num]  = state["grasp_axis"]
        av[frame_num]  = state["approach_vec"]
        tt[frame_num]  = state["thumb_tip"]
        it[frame_num]  = state["index_tip"]
        wr[frame_num]  = state["wrist"]
        vld[frame_num] = True
        detected += 1

        # ── Preview ───────────────────────────────────────────────────────────
        if show_preview:
            # Draw fingertip markers
            t_px, t_py = landmarks_2d[LM_THUMB_TIP]
            i_px, i_py = landmarks_2d[LM_INDEX_TIP]
            cv2.circle(frame, (t_px, t_py), 10, (255, 80, 0), cv2.FILLED)
            cv2.circle(frame, (i_px, i_py), 10, (0, 200, 0), cv2.FILLED)
            cv2.line(frame, (t_px, t_py), (i_px, i_py), (0, 220, 220), 2)

            pos = state["gripper_center"]
            cv2.putText(frame,
                        f"GC=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})m  W={state['gripper_width']*100:.1f}cm",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            pct = int(frame_num / max(total_frames - 1, 1) * 100)
            cv2.putText(frame, f"Frame {frame_num}/{total_frames} ({pct}%)",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Trajectory Extraction", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Trajectory] Preview closed early by user.")
                break

        frame_num += 1

        # Progress print every 5%
        if total_frames > 0 and frame_num % max(1, total_frames // 20) == 0:
            pct = frame_num / total_frames * 100
            print(f"  {pct:5.1f}%  frame {frame_num}/{total_frames}  "
                  f"detected={detected}  gaps={total_frames - detected}", end="\r")

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Trim to actual frames processed
    n = frame_num
    traj = TrajectoryData({
        "gripper_center": gc[:n],
        "gripper_width":  gw[:n],
        "grasp_axis":     ga[:n],
        "approach_vec":   av[:n],
        "thumb_tip":      tt[:n],
        "index_tip":      it[:n],
        "wrist":          wr[:n],
        "valid":          vld[:n],
        "frame_indices":  fi[:n],
        "fps":            fps,
        "resolution":     resolution_str,
        "video_path":     str(video_path),
    })

    detection_rate = detected / n * 100 if n > 0 else 0
    print(f"\n[Trajectory] Done. {n} frames, {detected} detected ({detection_rate:.1f}%), "
          f"longest gap={max_gap} frames ({max_gap/fps:.2f}s)")

    if output_path:
        traj.save(output_path)

    return traj
