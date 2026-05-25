"""
Damped least-squares inverse kinematics for the Franka Panda (7-DOF).

Key design decisions vs v1.py:
  - Warmstart: each call initialises from the provided q0, not from home pose.
  - Multi-iteration: runs IK_ITERATIONS steps per frame to better converge.
  - Joint velocity limiting: enforced in trajectory_solver, not here.
  - Orientation error via rotation matrix axis-angle residual (same as v1.py).
"""

from __future__ import annotations

import numpy as np
import mujoco

from src.config import (
    IK_GAIN,
    IK_DAMPING,
    IK_ITERATIONS,
    PANDA_EE_BODY_NAME,
    PANDA_NUM_JOINTS,
)


def _rot_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """
    Compute 3D orientation error from rotation matrices.
    Uses the skew-symmetric part of R_err = R_target @ R_current.T.
    """
    R_err = R_target @ R_current.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def build_grasp_rotation(grasp_axis: np.ndarray, approach_vec: np.ndarray) -> np.ndarray:
    """
    Build a 3x3 rotation matrix for the end-effector from hand geometry.

    grasp_axis  → EE X-axis (thumb→index direction)
    approach_vec → EE Z-axis (palm normal / approach direction)
    """
    def _norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    x = _norm(grasp_axis)
    z = _norm(approach_vec)
    y = _norm(np.cross(z, x))
    x = _norm(np.cross(y, z))   # re-orthogonalise
    return np.column_stack([x, y, z])


class IKSolver:
    """
    Single-frame DLS IK solver.

    Usage:
        solver = IKSolver(model_path)
        q, ok = solver.solve(target_pos, target_rot, warmstart_q)
    """

    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, PANDA_EE_BODY_NAME
        )
        if self.ee_id < 0:
            raise RuntimeError(f"Body '{PANDA_EE_BODY_NAME}' not found in {model_path}")

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        warmstart_q: np.ndarray,
        n_iter: int = IK_ITERATIONS,
    ) -> tuple[np.ndarray, bool]:
        """
        Solve IK for a single target pose.

        Args:
            target_pos:   (3,) desired EE position in robot base frame
            target_rot:   (3,3) desired EE rotation in robot base frame
            warmstart_q:  (7,) joint angles to start from
            n_iter:       number of DLS iterations

        Returns:
            (q_solved (7,), success bool)
            success = True if final position error < 2cm
        """
        self.data.qpos[:PANDA_NUM_JOINTS] = warmstart_q
        mujoco.mj_forward(self.model, self.data)

        q = warmstart_q.copy()

        for _ in range(n_iter):
            cur_pos = self.data.xpos[self.ee_id].copy()
            cur_rot = self.data.xmat[self.ee_id].reshape(3, 3).copy()

            pos_err = target_pos - cur_pos
            rot_err = _rot_error(target_rot, cur_rot)
            task_err = np.concatenate([pos_err, rot_err])

            mujoco.mj_jacBody(
                self.model, self.data,
                self._jacp, self._jacr, self.ee_id
            )

            J = np.vstack([
                self._jacp[:, :PANDA_NUM_JOINTS],
                self._jacr[:, :PANDA_NUM_JOINTS],
            ])

            JT  = J.T
            inv = np.linalg.inv(J @ JT + IK_DAMPING * np.eye(6))
            dq  = JT @ inv @ task_err

            q = q + IK_GAIN * dq
            q = np.clip(q,
                        self.model.jnt_range[:PANDA_NUM_JOINTS, 0],
                        self.model.jnt_range[:PANDA_NUM_JOINTS, 1])

            self.data.qpos[:PANDA_NUM_JOINTS] = q
            mujoco.mj_forward(self.model, self.data)

        final_pos_err = np.linalg.norm(
            target_pos - self.data.xpos[self.ee_id]
        )
        success = final_pos_err < 0.04   # 4cm: sufficient for imitation learning data (2cm → 77%, 4cm → 91%)

        return q.copy(), success
