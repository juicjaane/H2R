"""
Solve IK for every frame of a smoothed trajectory.

Improvements over v1.py per-frame greedy IK:
  1. Warmstart from previous frame's solution (not from home pose every frame).
  2. Multi-iteration convergence (IK_ITERATIONS per frame).
  3. Joint velocity clamping between consecutive frames.
  4. IK failure handling: carry forward last valid solution + flag the frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import (
    PANDA_HOME_QPOS,
    PANDA_NUM_JOINTS,
    PANDA_GRIPPER_MAX_M,
    PANDA_MAX_JOINT_VEL,
    ROBOT_XML,
)
from src.tracking.trajectory import TrajectoryData
from src.ik.solver import IKSolver, build_grasp_rotation
from src.ik.workspace import table_to_robot


# ─────────────────────────────────────────────────────────────────────────────
# Joint velocity limit per frame
# ─────────────────────────────────────────────────────────────────────────────

def _max_dq_per_frame(fps: float) -> float:
    """Max joint angle change per frame based on Panda's 2.175 rad/s limit."""
    return PANDA_MAX_JOINT_VEL / fps


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_trajectory(
    smoothed: TrajectoryData,
    base_pos_table: np.ndarray,
    model_path: str | None = None,
) -> dict:
    """
    Compute joint trajectory for all frames of a smoothed hand trajectory.

    Args:
        smoothed:        TrajectoryData (from Phase 4)
        base_pos_table:  (3,) robot base position in table frame (from Phase 5)
        model_path:      path to panda.xml (default: src.config.ROBOT_XML)

    Returns:
        dict with keys:
            joint_angles   (N, 7)   float32 — joint positions q[0:7]
            gripper        (N,)     float32 — gripper opening in metres [0, 0.08]
            ik_success     (N,)     bool    — True if IK converged for this frame
            base_pos_table (3,)     float64 — robot base used
    """
    if model_path is None:
        model_path = str(ROBOT_XML)

    N   = smoothed.n_frames
    fps = smoothed.fps
    max_dq = _max_dq_per_frame(fps)

    solver = IKSolver(model_path)

    joint_angles = np.zeros((N, PANDA_NUM_JOINTS), dtype=np.float32)
    gripper      = np.zeros(N,                     dtype=np.float32)
    ik_success   = np.zeros(N,                     dtype=bool)

    prev_q   = PANDA_HOME_QPOS.copy().astype(np.float64)
    n_ok     = 0
    n_fail   = 0
    n_vel    = 0   # frames where velocity was clamped

    print(f"[IK] Solving {N} frames  fps={fps:.1f}  max_dq/frame={max_dq:.4f} rad")
    report_every = max(1, N // 20)

    for i in range(N):
        # ── Target pose ────────────────────────────────────────────────────────
        target_pos = table_to_robot(
            smoothed.gripper_center[i].astype(np.float64),
            base_pos_table,
        )
        target_rot = build_grasp_rotation(
            smoothed.grasp_axis[i].astype(np.float64),
            smoothed.approach_vec[i].astype(np.float64),
        )

        # ── IK solve (warmstarted from prev frame) ────────────────────────────
        q, ok = solver.solve(target_pos, target_rot, warmstart_q=prev_q)

        if not ok:
            # Carry forward last valid q — keeps motion continuous
            q    = prev_q.copy()
            n_fail += 1
        else:
            n_ok += 1

        # ── Joint velocity clamping ───────────────────────────────────────────
        dq   = q - prev_q
        dq_max = np.max(np.abs(dq))
        if dq_max > max_dq:
            q = prev_q + dq * (max_dq / dq_max)
            n_vel += 1

        # ── Gripper ───────────────────────────────────────────────────────────
        grip = float(np.clip(smoothed.gripper_width[i], 0.0, PANDA_GRIPPER_MAX_M))

        joint_angles[i] = q.astype(np.float32)
        gripper[i]      = grip
        ik_success[i]   = ok
        prev_q          = q.copy()

        if (i + 1) % report_every == 0:
            pct = (i + 1) / N * 100
            print(f"  {pct:5.1f}%  frame {i+1}/{N}  "
                  f"ok={n_ok}  fail={n_fail}  vel_clamped={n_vel}", end="\r")

    print(f"\n[IK] Done.  ok={n_ok}  fail={n_fail}  vel_clamped={n_vel}  "
          f"success_rate={n_ok/N*100:.1f}%")

    return {
        "joint_angles":    joint_angles,
        "gripper":         gripper,
        "ik_success":      ik_success,
        "base_pos_table":  base_pos_table,
        "fps":             fps,
        "n_frames":        N,
    }


def save_joint_trajectory(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_angles=result["joint_angles"],
        gripper=result["gripper"],
        ik_success=result["ik_success"],
        base_pos_table=result["base_pos_table"],
        fps=np.array([result["fps"]]),
        n_frames=np.array([result["n_frames"]]),
    )
    n_ok = int(result["ik_success"].sum())
    N    = result["n_frames"]
    print(f"[IK] Saved {N} frames ({n_ok}/{N} successful) to {path}")


def load_joint_trajectory(path: Path) -> dict:
    d = np.load(path)
    return {
        "joint_angles":   d["joint_angles"],
        "gripper":        d["gripper"],
        "ik_success":     d["ik_success"],
        "base_pos_table": d["base_pos_table"],
        "fps":            float(d["fps"][0]),
        "n_frames":       int(d["n_frames"][0]),
    }
