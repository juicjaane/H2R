"""
Phase 5: Analyse workspace and find optimal Panda base placement.

Usage:
    python scripts/analyze_workspace.py --smoothed data/take1_smoothed.npz
    python scripts/analyze_workspace.py --smoothed data/take1_smoothed.npz --calibration data/calibration.json
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

import src.config as cfg
from src.tracking.trajectory import TrajectoryData
from src.ik.workspace import (
    analyze_workspace,
    find_robot_placement,
    load_table_corners_in_table_frame,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoothed",    type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        default=cfg.DATA_DIR / "calibration.json")
    parser.add_argument("--output",      type=Path, default=None,
                        help="Save placement JSON (default: data/<stem>_placement.json)")
    parser.add_argument("--base-z",      type=float, default=0.0,
                        help=(
                            "Robot base height in TABLE frame (metres). "
                            "0.0 = table-mounted (base at table surface). "
                            "-0.75 = floor-mounted (base ~75cm below table). "
                            "Default: 0.0"
                        ))
    args = parser.parse_args()

    if not args.smoothed.exists():
        print(f"ERROR: {args.smoothed} not found.")
        sys.exit(1)

    if args.output is None:
        stem = args.smoothed.stem.replace("_smoothed", "")
        args.output = cfg.DATA_DIR / f"{stem}_placement.json"

    # ── Load data ─────────────────────────────────────────────────────────────
    traj  = TrajectoryData.load(args.smoothed)
    stats = analyze_workspace(traj)

    print(f"\n[Workspace] Gripper centre stats (table frame, metres):")
    print(f"  Centroid:     ({stats.centroid[0]:.3f}, {stats.centroid[1]:.3f}, {stats.centroid[2]:.3f})")
    print(f"  Bounding box: X [{stats.min_bound[0]:.3f}, {stats.max_bound[0]:.3f}]  "
          f"Y [{stats.min_bound[1]:.3f}, {stats.max_bound[1]:.3f}]  "
          f"Z [{stats.min_bound[2]:.3f}, {stats.max_bound[2]:.3f}]")
    print(f"  XY spread (max radius from centroid): {stats.max_xy_radius:.3f} m")
    print(f"  Min dist for full reach: {stats.max_xy_radius:.3f} m  "
          f"(must be < {cfg.PANDA_REACH_M} m)")

    # ── Find placement ────────────────────────────────────────────────────────
    base_z = args.base_z
    print(f"\n[Placement] Robot base Z in table frame: {base_z:.3f} m  "
          f"({'table-mounted' if base_z == 0 else f'floor-mounted, {abs(base_z):.2f}m below table'})")
    result = find_robot_placement(traj, stats, base_z=base_z)

    # ── Save placement JSON ───────────────────────────────────────────────────
    placement_data = {
        "base_pos_table_frame": result.base_pos.tolist(),   # [x, y, base_z]
        "base_z_note":          ("table-mounted" if base_z == 0
                                 else f"floor-mounted, {abs(base_z):.2f}m below table"),
        "coverage":             result.coverage,
        "n_reachable":          result.n_reachable,
        "n_total":              result.n_total,
        "panda_reach_m":        cfg.PANDA_REACH_M,
        "panda_deadzone_m":     cfg.PANDA_DEADZONE_M,
    }
    with open(args.output, "w") as f:
        json.dump(placement_data, f, indent=2)
    print(f"\n[Placement] Saved to {args.output}")

    # ── Load table corners for visualisation ──────────────────────────────────
    table_corners = None
    if args.calibration.exists():
        try:
            table_corners = load_table_corners_in_table_frame(args.calibration)
        except Exception as e:
            print(f"[Vis] Could not load table corners: {e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Workspace Analysis & Robot Placement", fontsize=13)

    # Left: top-down view
    ax = axes[0]
    ax.set_title("Top-down (table XY plane)", fontsize=10)

    # Coverage heatmap
    xe0, xe1, ye0, ye1 = result.grid_extent
    im = ax.imshow(result.search_grid,
                   extent=[xe0, xe1, ye0, ye1],
                   origin="lower", cmap="YlGn",
                   vmin=0, vmax=1, aspect="equal", alpha=0.6)
    fig.colorbar(im, ax=ax, label="Coverage fraction", shrink=0.8)

    # Trajectory path
    gc_v = traj.gripper_center[traj.valid]
    sc   = ax.scatter(gc_v[:, 0], gc_v[:, 1],
                      c=np.arange(len(gc_v)), cmap="plasma",
                      s=6, alpha=0.6, label="gripper path", zorder=4)

    # Table outline
    if table_corners is not None:
        corners_xy = np.vstack([table_corners[:, :2], table_corners[0, :2]])
        ax.plot(corners_xy[:, 0], corners_xy[:, 1],
                "b-", linewidth=2, label="table outline", zorder=3)
        labels = ["TL", "TR", "BR", "BL"]
        for i, (label, c) in enumerate(zip(labels, table_corners)):
            ax.annotate(label, (c[0], c[1]), fontsize=8, color="blue",
                        xytext=(4, 4), textcoords="offset points")

    # Robot base
    bx, by = result.base_pos[:2]
    ax.plot(bx, by, "r*", markersize=18, label=f"robot base ({bx:.2f},{by:.2f})", zorder=6)

    # Reach circles
    for r, ls, lbl in [
        (cfg.PANDA_REACH_M,    "-",  f"reach {cfg.PANDA_REACH_M}m"),
        (cfg.PANDA_DEADZONE_M, "--", f"dead zone {cfg.PANDA_DEADZONE_M}m"),
    ]:
        circ = plt.Circle((bx, by), r, fill=False, color="red",
                           linestyle=ls, linewidth=1.4, alpha=0.7, label=lbl)
        ax.add_patch(circ)

    # Table origin marker
    ax.plot(0, 0, "b+", markersize=12, markeredgewidth=2, label="table origin")

    ax.set_xlabel("X (table width, m)")
    ax.set_ylabel("Y (table depth, m)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # Right: coverage vs candidate positions (1D slice through best base)
    ax2 = axes[1]
    ax2.set_title(f"Coverage heatmap (search grid)\nBest: ({bx:.3f}, {by:.3f}) m  "
                  f"→ {result.coverage*100:.1f}% coverage", fontsize=10)
    ax2.imshow(result.search_grid,
               extent=[xe0, xe1, ye0, ye1],
               origin="lower", cmap="RdYlGn",
               vmin=0, vmax=1, aspect="equal")
    ax2.plot(bx, by, "w*", markersize=16, label="best position")
    ax2.plot(stats.centroid[0], stats.centroid[1], "wx", markersize=12,
             markeredgewidth=2, label="traj centroid")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*50}")
    print(f"Robot base position (table frame): "
          f"({result.base_pos[0]:.4f}, {result.base_pos[1]:.4f}, 0.000) m")
    print(f"Workspace coverage:  {result.coverage*100:.1f}%")
    print(f"{'='*50}")
    if result.coverage < 0.95:
        print("ACTION NEEDED: Coverage < 95%.")
        print("  Options: (1) adjust camera/table position and re-record,")
        print("           (2) zoom out so hand motion covers smaller area,")
        print("           (3) proceed anyway (unreachable frames will use IK fallback).")


if __name__ == "__main__":
    main()
