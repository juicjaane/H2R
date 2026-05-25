"""
Smooth a raw trajectory using Savitzky-Golay.

Usage:
    python scripts/smooth_trajectory.py --raw data/take1_raw.npz
    python scripts/smooth_trajectory.py --raw data/take1_raw.npz --window 21 --poly 3
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import src.config as cfg
from src.tracking.trajectory import TrajectoryData
from src.tracking.smoother   import smooth_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",    type=Path, required=True,
                        help="Raw trajectory .npz from extract_trajectory.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: <stem>_smoothed.npz)")
    parser.add_argument("--window", type=int, default=15,
                        help="Savitzky-Golay window length (odd, default 15)")
    parser.add_argument("--poly",   type=int, default=3,
                        help="Polynomial order (default 3)")
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"ERROR: {args.raw} not found.")
        sys.exit(1)

    if args.output is None:
        args.output = args.raw.parent / (args.raw.stem.replace("_raw", "") + "_smoothed.npz")

    raw  = TrajectoryData.load(args.raw)
    smth = smooth_trajectory(raw, window=args.window, polyorder=args.poly)
    smth.save(args.output)

    v = smth.valid
    gc_raw  = raw.gripper_center[v]
    gc_smth = smth.gripper_center[v]
    pos_diff = np.linalg.norm(gc_smth - gc_raw, axis=1)

    print(f"\nSmoothing effect on valid frames:")
    print(f"  Mean displacement raw→smooth: {pos_diff.mean()*100:.1f}cm")
    print(f"  Max  displacement raw→smooth: {pos_diff.max()*100:.1f}cm")

    gw_raw  = raw.gripper_width[v]
    gw_smth = smth.gripper_width[v]
    print(f"  Gripper width before clip+smooth: {gw_raw.min()*100:.1f}–{gw_raw.max()*100:.1f}cm")
    print(f"  Gripper width after  clip+smooth: {gw_smth.min()*100:.1f}–{gw_smth.max()*100:.1f}cm")

    print(f"\nSmoothed trajectory: {args.output}")


if __name__ == "__main__":
    main()
