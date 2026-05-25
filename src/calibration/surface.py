"""
Surface calibration: fit a plane from 4 table corners, build table coordinate frame.

All geometry uses OpenCV camera frame convention: X=right, Y=down, Z=into_scene.
The table frame uses: X=table-width (TL→TR), Y=table-depth (TL→BL), Z=up (toward camera).
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_3d(
    px: float, py: float, depth_m: float,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """
    Back-project a pixel at metric depth to a 3D point in camera frame.
    Camera frame: X=right, Y=down, Z=into_scene.
    """
    return np.array([
        (px - cx) * depth_m / fx,
        (py - cy) * depth_m / fy,
        depth_m,
    ], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Plane fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_plane(points: np.ndarray) -> dict:
    """
    Fit a plane to N ≥ 3 3D points using SVD (least-squares).

    Args:
        points: (N, 3) float array in camera frame

    Returns dict:
        normal  — unit normal (3,), sign chosen so Z component < 0 (points toward camera)
        d       — scalar such that normal · p = d for all plane points
        origin  — centroid of input points (3,)
        residuals_m — per-point signed distance from fitted plane (N,)
        max_residual_m — max |residual|
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    centroid = points.mean(axis=0)
    centered = points - centroid

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # eigenvector for smallest singular value = plane normal

    # Normalise (SVD already gives unit vectors, but be defensive)
    normal = normal / np.linalg.norm(normal)

    # Convention: Z component of normal in camera frame is negative
    # (the table normal points back toward the camera, i.e. in -Z direction
    # since camera Z points away from camera into the scene)
    if normal[2] > 0:
        normal = -normal

    d = float(np.dot(normal, centroid))
    residuals = (centered @ normal).astype(np.float64)

    return {
        "normal": normal,
        "d": d,
        "origin": centroid,
        "residuals_m": residuals,
        "max_residual_m": float(np.max(np.abs(residuals))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table coordinate frame
# ─────────────────────────────────────────────────────────────────────────────

def build_table_transform(corners: np.ndarray, plane: dict) -> np.ndarray:
    """
    Build 4×4 homogeneous transform: camera frame → table frame.

    Table frame definition:
        Origin  — centroid of 4 corners
        X-axis  — TL→TR direction, projected onto plane (table width)
        Y-axis  — Z × X  (table depth, TL→BL direction)
        Z-axis  — plane normal pointing toward camera (table "up")

    Args:
        corners: (4, 3) array, order [TL, TR, BR, BL], camera frame
        plane:   output of fit_plane()

    Returns:
        T_cam_to_table: (4, 4) float64 homogeneous transform
    """
    normal = plane["normal"]   # unit vector, points toward camera
    origin = plane["origin"]   # centroid in camera frame

    # X-axis: TL → TR, projected onto table plane
    tl, tr = corners[0], corners[1]
    x_raw = tr - tl
    x_axis = x_raw - np.dot(x_raw, normal) * normal
    norm_x = np.linalg.norm(x_axis)
    if norm_x < 1e-6:
        raise ValueError("TL and TR corners are too close — cannot define X-axis.")
    x_axis = x_axis / norm_x

    # Z-axis is the plane normal (toward camera)
    z_axis = normal

    # Y-axis: right-hand rule (table depth)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Re-orthogonalise X (Gram-Schmidt, removes any floating-point skew)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # R_c2t: rotates camera-frame vectors into table-frame coordinates
    # Columns of R_t2c are the table axes in camera coords  → R_c2t = R_t2c.T
    R_t2c = np.column_stack([x_axis, y_axis, z_axis])
    R_c2t = R_t2c.T

    t_c2t = -R_c2t @ origin

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_c2t
    T[:3, 3] = t_c2t

    return T


def cam_to_table(point_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform a 3D point from camera frame to table frame using T_cam_to_table."""
    p = np.ones(4)
    p[:3] = point_cam
    return (T @ p)[:3]


def table_to_cam(point_table: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Inverse transform: table frame → camera frame."""
    T_inv = np.linalg.inv(T)
    p = np.ones(4)
    p[:3] = point_table
    return (T_inv @ p)[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Calibration save / load
# ─────────────────────────────────────────────────────────────────────────────

CORNER_NAMES = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]


def save_calibration(
    corners_cam: np.ndarray,
    plane: dict,
    T_cam_to_table: np.ndarray,
    intrinsics: dict,
    resolution: str,
    save_path: Path,
) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "resolution": resolution,
        "intrinsics": intrinsics,
        "corners_cam": corners_cam.tolist(),
        "plane_normal": plane["normal"].tolist(),
        "plane_d": plane["d"],
        "origin_cam": plane["origin"].tolist(),
        "T_cam_to_table": T_cam_to_table.tolist(),
        "residuals_m": plane["residuals_m"].tolist(),
        "max_residual_m": plane["max_residual_m"],
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Calibration] Saved to {save_path}")


def load_calibration(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    data["corners_cam"]    = np.array(data["corners_cam"])
    data["plane_normal"]   = np.array(data["plane_normal"])
    data["origin_cam"]     = np.array(data["origin_cam"])
    data["T_cam_to_table"] = np.array(data["T_cam_to_table"])
    data["residuals_m"]    = np.array(data["residuals_m"])
    return data
