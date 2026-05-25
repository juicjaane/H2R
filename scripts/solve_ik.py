"""
Phase 6: Solve IK for every frame of the smoothed trajectory.

Usage:
    python scripts/solve_ik.py --smoothed data/take1_smoothed.npz
    python scripts/solve_ik.py --smoothed data/take1_smoothed.npz --placement data/take1_placement.json
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import src.config as cfg
from src.tracking.trajectory import TrajectoryData
from src.ik.trajectory_solver import solve_trajectory, save_joint_trajectory


def main():
    parser = argparse.ArgumentParser(description="Solve offline IK for trajectory.")
    parser.add_argument("--smoothed",   type=Path, required=True)
    parser.add_argument("--placement",  type=Path, default=None,
                        help="Placement JSON from analyze_workspace.py "
                             "(default: data/<stem>_placement.json)")
    parser.add_argument("--output",     type=Path, default=None,
                        help="Output .npz (default: data/<stem>_joints.npz)")
    args = parser.parse_args()

    stem = args.smoothed.stem.replace("_smoothed", "")

    if args.placement is None:
        args.placement = cfg.DATA_DIR / f"{stem}_placement.json"
    if args.output is None:
        args.output = cfg.DATA_DIR / f"{stem}_joints.npz"

    for p, name in [(args.smoothed, "smoothed trajectory"),
                    (args.placement, "placement JSON")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            sys.exit(1)

    with open(args.placement) as f:
        placement = json.load(f)
    base_pos = np.array(placement["base_pos_table_frame"])

    print(f"Robot base (table frame): ({base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}) m")
    print(f"Coverage from placement:  {placement['coverage']*100:.1f}%")
    print()

    smoothed = TrajectoryData.load(args.smoothed)
    result   = solve_trajectory(smoothed, base_pos)
    save_joint_trajectory(result, args.output)

    # ── Report ────────────────────────────────────────────────────────────────
    ja = result["joint_angles"]
    print(f"\nJoint angle ranges (rad):")
    for j in range(cfg.PANDA_NUM_JOINTS):
        print(f"  J{j+1}: [{ja[:,j].min():.3f}, {ja[:,j].max():.3f}]")
    print(f"Gripper range: [{result['gripper'].min()*100:.1f}, "
          f"{result['gripper'].max()*100:.1f}] cm")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
