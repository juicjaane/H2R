"""
Visualise raw vs smoothed trajectory for verification.

Usage:
    python scripts/plot_trajectory.py --raw data/take1_raw.npz --smoothed data/take1_smoothed.npz
    python scripts/plot_trajectory.py --raw data/take1_raw.npz   # raw only
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.tracking.trajectory import TrajectoryData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",      type=Path, required=True)
    parser.add_argument("--smoothed", type=Path, default=None)
    args = parser.parse_args()

    raw = TrajectoryData.load(args.raw)
    smth = TrajectoryData.load(args.smoothed) if (args.smoothed and args.smoothed.exists()) else None

    t = np.arange(raw.n_frames) / raw.fps

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Trajectory: {Path(raw.video_path).name}", fontsize=13)

    labels  = ["X (table width, m)", "Y (table depth, m)", "Z (height above table, m)"]
    colours = ["#e74c3c", "#2ecc71", "#3498db"]

    valid_mask = raw.valid

    for ax, col, lbl, dim in zip(axes[:3], colours, labels, range(3)):
        raw_vals = raw.gripper_center[:, dim]

        # Raw — dim grey for invalid frames, colour for valid
        ax.scatter(t[~valid_mask], raw_vals[~valid_mask],
                   s=4, c="lightgrey", label="invalid (gap)", zorder=1)
        ax.scatter(t[valid_mask],  raw_vals[valid_mask],
                   s=4, c=col, alpha=0.4, label="raw", zorder=2)

        if smth is not None:
            ax.plot(t, smth.gripper_center[:, dim],
                    color=col, linewidth=1.8, label="smoothed", zorder=3)

        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylabel(lbl, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, markerscale=2)
        ax.grid(alpha=0.3)

    # Gripper width
    ax = axes[3]
    gw_raw = raw.gripper_width * 100   # convert to cm
    ax.scatter(t[~valid_mask], gw_raw[~valid_mask],
               s=4, c="lightgrey", label="invalid", zorder=1)
    ax.scatter(t[valid_mask],  gw_raw[valid_mask],
               s=4, c="#e67e22", alpha=0.4, label="raw", zorder=2)

    if smth is not None:
        ax.plot(t, smth.gripper_width * 100,
                color="#e67e22", linewidth=1.8, label="smoothed (clipped)", zorder=3)

    ax.axhline(15, color="red", linewidth=0.8, linestyle=":", alpha=0.6, label="15cm clip")
    ax.set_ylabel("Gripper width (cm)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # ── 3D trajectory ──────────────────────────────────────────────────────────
    fig2 = plt.figure(figsize=(9, 7))
    ax3d = fig2.add_subplot(111, projection="3d")
    ax3d.set_title("Gripper centre — table frame (valid frames only)", fontsize=11)

    gc_v = raw.gripper_center[valid_mask]
    sc = ax3d.scatter(gc_v[:, 0], gc_v[:, 1], gc_v[:, 2],
                      c=np.where(valid_mask)[0][: len(gc_v)],
                      cmap="plasma", s=8, alpha=0.5, label="raw")

    if smth is not None:
        gc_s = smth.gripper_center
        ax3d.plot(gc_s[:, 0], gc_s[:, 1], gc_s[:, 2],
                  color="cyan", linewidth=1.2, label="smoothed", alpha=0.8)

    # Draw table plane at Z=0
    xs = np.linspace(gc_v[:, 0].min() - 0.05, gc_v[:, 0].max() + 0.05, 2)
    ys = np.linspace(gc_v[:, 1].min() - 0.05, gc_v[:, 1].max() + 0.05, 2)
    Xg, Yg = np.meshgrid(xs, ys)
    ax3d.plot_surface(Xg, Yg, np.zeros_like(Xg),
                      alpha=0.12, color="grey", label="table (Z=0)")

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(fontsize=8)
    fig2.colorbar(sc, ax=ax3d, label="frame index", shrink=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
