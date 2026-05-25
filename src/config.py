"""
Central configuration for the hand-to-robot pipeline.
Import this at the top of any script that needs paths or constants.
"""

import sys
import os
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Make Depth-Anything-V2 importable ─────────────────────────────────────────
# Standard (relative) model: depth_anything_v2.dpt.DepthAnythingV2
DAV2_ROOT = PROJECT_ROOT / "Depth-Anything-V2"
if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

# Metric model: import from metric_depth sub-package
DAV2_METRIC_ROOT = DAV2_ROOT / "metric_depth"
if str(DAV2_METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_METRIC_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR  = PROJECT_ROOT / "checkpoints"
ROBOT_XML        = PROJECT_ROOT / "robot" / "panda.xml"
DATA_DIR         = PROJECT_ROOT / "data"
OUTPUTS_DIR      = PROJECT_ROOT / "outputs"

# Checkpoint filenames
DEPTH_RELATIVE_CKPT = CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"
DEPTH_METRIC_CKPT   = CHECKPOINTS_DIR / "depth_anything_v2_metric_hypersim_vits.pth"

# ── Camera intrinsics: Logitech Brio 100, 58° diagonal FOV ───────────────────
# Derived from: f = sqrt(w²+h²) / (2 * tan(29°))
# Principal point at image center. No distortion correction needed.
BRIO100_INTRINSICS = {
    "1920x1080": {"fx": 1986.0, "fy": 1986.0, "cx": 960.0,  "cy": 540.0},
    "1280x720":  {"fx": 1324.0, "fy": 1324.0, "cx": 640.0,  "cy": 360.0},
    "640x480":   {"fx": 722.0,  "fy": 722.0,  "cx": 320.0,  "cy": 240.0},
}

def get_intrinsics(width: int, height: int) -> dict:
    """Return intrinsics dict for a given resolution. Raises if unknown."""
    key = f"{width}x{height}"
    if key not in BRIO100_INTRINSICS:
        raise ValueError(
            f"No Brio 100 intrinsics for {key}. "
            f"Supported: {list(BRIO100_INTRINSICS.keys())}"
        )
    return BRIO100_INTRINSICS[key]

# ── Depth model config ────────────────────────────────────────────────────────
DEPTH_MODEL_CONFIG = {
    "encoder":      "vits",
    "features":     64,
    "out_channels": [48, 96, 192, 384],
}

# Metric model requires max_depth; Hypersim (indoor) trained up to 20m.
DEPTH_METRIC_MAX_DEPTH = 20.0

# ── Robot constants ───────────────────────────────────────────────────────────
PANDA_EE_BODY_NAME   = "hand"
PANDA_NUM_JOINTS     = 7
PANDA_REACH_M        = 0.855    # outer reachability sphere radius
PANDA_DEADZONE_M     = 0.170    # inner dead zone radius
PANDA_GRIPPER_MAX_M  = 0.08     # maximum gripper opening in meters
PANDA_MAX_JOINT_VEL  = 2.175    # rad/s per joint (from spec sheet)

# ── IK parameters ─────────────────────────────────────────────────────────────
IK_GAIN        = 0.3   # 0.3 beats 0.5 — smaller steps reduce vel-clamp cascade on warmstart chain
IK_DAMPING     = 1e-4
IK_ITERATIONS  = 50   # 50 is optimal — more iterations hurt due to vel_clamp+warmstart interaction

# ── Initial Panda joint pose (known-good home position) ──────────────────────
import numpy as np
PANDA_HOME_QPOS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
