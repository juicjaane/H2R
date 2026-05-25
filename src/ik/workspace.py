"""
Phase 5: Workspace analysis and optimal robot base placement.

Given a smoothed hand trajectory (in table frame), find the 2D position of the
Panda robot base (on the table plane, Z=0) that maximises the fraction of
waypoints inside the robot's reachable workspace.

Panda workspace model used here:
  - Outer sphere: 0.855m radius (all joints extended)
  - Inner dead zone: 0.170m radius (too close, singular)
  - Elevation: no constraint applied (spherical approximation)

Known limitation: orientation-aware reachability is not modelled. A point inside
the sphere may still fail IK due to joint limits + required EE orientation.
This is documented in SKILL.md L4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    PANDA_REACH_M,
    PANDA_DEADZONE_M,
)
from src.tracking.trajectory import TrajectoryData
from src.calibration.surface import load_calibration, cam_to_table


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkspaceStats:
    """Summary statistics of the hand trajectory in table frame."""
    centroid:      np.ndarray    # (3,) mean gripper centre
    min_bound:     np.ndarray    # (3,) component-wise minimum
    max_bound:     np.ndarray    # (3,) component-wise maximum
    extent:        np.ndarray    # (3,) max - min
    max_xy_radius: float         # max distance from centroid in XY plane
    n_waypoints:   int           # number of valid frames used


@dataclass
class PlacementResult:
    """Result of the robot placement optimisation."""
    base_pos:       np.ndarray   # (3,) in table frame — Z is always 0
    coverage:       float        # fraction of waypoints in reachable zone [0,1]
    n_reachable:    int
    n_total:        int
    workspace_stats: WorkspaceStats
    search_grid:    Optional[np.ndarray] = None   # (M, M) coverage heatmap
    grid_extent:    Optional[tuple]      = None   # (x_min, x_max, y_min, y_max)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_workspace(traj: TrajectoryData) -> WorkspaceStats:
    """Compute bounding stats of the valid gripper centre positions."""
    gc = traj.gripper_center[traj.valid]
    if len(gc) == 0:
        raise ValueError("No valid frames in trajectory.")

    centroid      = gc.mean(axis=0)
    min_bound     = gc.min(axis=0)
    max_bound     = gc.max(axis=0)
    extent        = max_bound - min_bound
    xy_dists      = np.linalg.norm(gc[:, :2] - centroid[:2], axis=1)
    max_xy_radius = float(xy_dists.max())

    return WorkspaceStats(
        centroid=centroid,
        min_bound=min_bound,
        max_bound=max_bound,
        extent=extent,
        max_xy_radius=max_xy_radius,
        n_waypoints=len(gc),
    )


def _coverage_score(
    base_xy: np.ndarray,
    waypoints: np.ndarray,
    base_z: float = 0.0,
    outer: float = PANDA_REACH_M,
    inner: float = PANDA_DEADZONE_M,
) -> float:
    """
    Fraction of waypoints (N, 3) that fall inside the annular workspace.

    base_z is the robot base height in TABLE frame:
      -  0.0  → robot base is at table surface (table-mounted arm)
      - ~-0.75 → robot base is on the floor below the table (floor-mounted)
    All three axes (X, Y, Z) contribute to the 3D distance.
    """
    base_3d = np.array([base_xy[0], base_xy[1], base_z])
    dists   = np.linalg.norm(waypoints - base_3d, axis=1)
    in_zone = (dists > inner) & (dists < outer)
    return float(in_zone.sum()) / len(waypoints)


def find_robot_placement(
    traj: TrajectoryData,
    stats: WorkspaceStats,
    grid_resolution: int = 40,
    search_margin: float = 0.30,
    outer_reach: float = PANDA_REACH_M,
    inner_dead:  float = PANDA_DEADZONE_M,
    base_z: float = 0.0,
) -> PlacementResult:
    """
    base_z: robot base height in table frame.
        0.0   → table-mounted robot (base at table surface level)
        -0.75 → floor-mounted robot (base ~75cm below table surface)
    This directly affects which XY positions give good 3D reach coverage.
    """
    """
    Grid-search for the robot base position that maximises workspace coverage.

    The search grid is centred on the trajectory centroid XY, extended by
    search_margin in every direction.

    Args:
        traj:             smoothed TrajectoryData
        stats:            output of analyze_workspace()
        grid_resolution:  number of grid points per axis (40 → 1600 candidates)
        search_margin:    extra search radius beyond trajectory bounding box (m)
        outer_reach:      Panda outer sphere radius (m)
        inner_dead:       Panda inner dead zone radius (m)

    Returns:
        PlacementResult with best base_pos and coverage heatmap for plotting
    """
    gc = traj.gripper_center[traj.valid]

    cx, cy = stats.centroid[:2]
    half_x = max(stats.extent[0] * 0.5, 0.1) + search_margin
    half_y = max(stats.extent[1] * 0.5, 0.1) + search_margin

    x_vals = np.linspace(cx - half_x, cx + half_x, grid_resolution)
    y_vals = np.linspace(cy - half_y, cy + half_y, grid_resolution)

    best_score = -1.0
    best_xy    = np.array([cx, cy])
    heatmap    = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

    for i, bx in enumerate(x_vals):
        for j, by in enumerate(y_vals):
            score = _coverage_score(np.array([bx, by]), gc, base_z, outer_reach, inner_dead)
            heatmap[j, i] = score   # j=row (y), i=col (x) for imshow
            if score > best_score:
                best_score = score
                best_xy    = np.array([bx, by])

    base_pos = np.array([best_xy[0], best_xy[1], base_z])
    n_reach  = int(round(best_score * len(gc)))

    print(f"[Placement] Best base: ({base_pos[0]:.3f}, {base_pos[1]:.3f}) m  "
          f"coverage={best_score*100:.1f}%  ({n_reach}/{len(gc)} waypoints)")

    if best_score < 0.95:
        print(f"[Placement] WARNING: coverage {best_score*100:.1f}% < 95%. "
              "Some waypoints will be outside the reachable workspace.")

    return PlacementResult(
        base_pos=base_pos,
        coverage=best_score,
        n_reachable=n_reach,
        n_total=len(gc),
        workspace_stats=stats,
        search_grid=heatmap,
        grid_extent=(
            float(x_vals[0]), float(x_vals[-1]),
            float(y_vals[0]), float(y_vals[-1]),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transform helper
# ─────────────────────────────────────────────────────────────────────────────

def table_to_robot(
    point_table: np.ndarray,
    base_pos_table: np.ndarray,
) -> np.ndarray:
    """
    Convert a point from table frame to robot base frame.

    The robot base frame shares orientation with the table frame
    (X=width, Y=depth, Z=up) but is translated to base_pos_table.

    Args:
        point_table:    (3,) or (N, 3) in table frame
        base_pos_table: (3,) robot base position in table frame

    Returns:
        (3,) or (N, 3) in robot base frame
    """
    return point_table - base_pos_table


def load_table_corners_in_table_frame(calibration_path: Path) -> np.ndarray:
    """
    Load the 4 calibration corners expressed in table frame for visualisation.
    Returns (4, 3) array.
    """
    calib = load_calibration(calibration_path)
    T     = calib["T_cam_to_table"]
    corners_cam = calib["corners_cam"]
    return np.array([cam_to_table(c, T) for c in corners_cam])
