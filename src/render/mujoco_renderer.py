"""
Headless MuJoCo renderer for Franka Panda joint trajectories.

Renders each frame by setting joint positions + gripper, running mj_forward,
then capturing via mujoco.Renderer (no display window required).
"""

from __future__ import annotations

import numpy as np
import mujoco

from src.config import PANDA_NUM_JOINTS, PANDA_GRIPPER_MAX_M


# ── Default camera presets ────────────────────────────────────────────────────

CAMERA_PRESETS: dict[str, dict] = {
    "side": {
        "azimuth":   135.0,
        "elevation": -25.0,
        "distance":  2.0,
        "lookat":    [0.4, 0.0, 0.35],
    },
    "top": {
        "azimuth":   0.0,
        "elevation": -90.0,
        "distance":  2.0,
        "lookat":    [0.4, 0.0, 0.0],
    },
    "front": {
        "azimuth":   180.0,
        "elevation": -20.0,
        "distance":  2.0,
        "lookat":    [0.4, 0.0, 0.35],
    },
}


def _apply_camera(cam: mujoco.MjvCamera, preset: dict) -> None:
    cam.azimuth   = preset["azimuth"]
    cam.elevation = preset["elevation"]
    cam.distance  = preset["distance"]
    cam.lookat[:] = preset["lookat"]


def _set_pose(model: mujoco.MjModel, data: mujoco.MjData,
              joint_q: np.ndarray, gripper_m: float) -> None:
    """Set arm joints + gripper fingers, then run forward kinematics."""
    data.qpos[:PANDA_NUM_JOINTS] = joint_q.astype(np.float64)
    finger = float(np.clip(gripper_m, 0.0, PANDA_GRIPPER_MAX_M)) / 2.0
    data.qpos[PANDA_NUM_JOINTS]     = finger   # left_finger
    data.qpos[PANDA_NUM_JOINTS + 1] = finger   # right_finger
    mujoco.mj_forward(model, data)


# ── Public API ────────────────────────────────────────────────────────────────

def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_q: np.ndarray,
    gripper_m: float,
    camera: str | dict = "side",
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """
    Render a single frame to a BGR numpy array (H×W×3 uint8).

    Args:
        model:     Loaded MjModel
        data:      MjData (will be mutated)
        joint_q:   (7,) joint angles in radians
        gripper_m: gripper opening in metres [0, 0.08]
        camera:    preset name ("side", "top", "front") or a dict with keys
                   azimuth/elevation/distance/lookat
        width:     output image width in pixels
        height:    output image height in pixels
    """
    _set_pose(model, data, joint_q, gripper_m)

    preset = CAMERA_PRESETS[camera] if isinstance(camera, str) else camera

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        cam = mujoco.MjvCamera()
        _apply_camera(cam, preset)
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()   # H×W×3 uint8 RGB

    return rgb[:, :, ::-1].copy()   # RGB → BGR


def render_trajectory(
    joint_angles: np.ndarray,
    gripper: np.ndarray,
    model_path: str,
    camera: str | dict = "side",
    width: int = 1280,
    height: int = 720,
    progress: bool = True,
) -> list[np.ndarray]:
    """
    Render every frame of a joint trajectory.

    Args:
        joint_angles:  (N, 7) float32
        gripper:       (N,) float32 in metres
        model_path:    path to panda.xml
        camera:        preset name or dict (see CAMERA_PRESETS)
        width/height:  output resolution
        progress:      print progress to stdout

    Returns:
        List of N BGR frames (H×W×3 uint8).
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    N = len(joint_angles)
    preset = CAMERA_PRESETS[camera] if isinstance(camera, str) else camera
    frames: list[np.ndarray] = []

    report_every = max(1, N // 20)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        cam = mujoco.MjvCamera()
        _apply_camera(cam, preset)

        for i in range(N):
            _set_pose(model, data, joint_angles[i], gripper[i])
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            frames.append(rgb[:, :, ::-1].copy())

            if progress and (i + 1) % report_every == 0:
                pct = (i + 1) / N * 100
                print(f"  [render] {pct:5.1f}%  frame {i+1}/{N}", end="\r")

    if progress:
        print(f"\n[render] Done. {N} frames rendered at {width}×{height}.")

    return frames
