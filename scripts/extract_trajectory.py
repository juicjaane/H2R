"""
Extract hand trajectory from a recorded video.

Usage:
    python scripts/extract_trajectory.py --video data/take1.mp4
    python scripts/extract_trajectory.py --video data/take1.mp4 --preview
    python scripts/extract_trajectory.py --video data/take1.mp4 --calibration data/calibration.json
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import src.config as cfg
from src.tracking.trajectory import extract_trajectory


def main():
    parser = argparse.ArgumentParser(description="Extract hand trajectory from video.")
    parser.add_argument("--video",       type=Path, required=True,
                        help="Path to recorded .mp4")
    parser.add_argument("--calibration", type=Path,
                        default=cfg.DATA_DIR / "calibration.json",
                        help="Path to calibration.json (default: data/calibration.json)")
    parser.add_argument("--output",      type=Path, default=None,
                        help="Output .npz path (default: data/<video_stem>_raw.npz)")
    parser.add_argument("--preview",     action="store_true",
                        help="Show live tracking overlay while processing")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    if not args.calibration.exists():
        print(f"ERROR: Calibration file not found: {args.calibration}")
        print("Run:  python scripts/calibrate.py")
        sys.exit(1)

    if args.output is None:
        args.output = cfg.DATA_DIR / f"{args.video.stem}_raw.npz"

    traj = extract_trajectory(
        video_path=args.video,
        calibration_path=args.calibration,
        output_path=args.output,
        show_preview=args.preview,
    )

    print(f"\nSummary:")
    print(f"  Frames:         {traj.n_frames}")
    print(f"  Valid frames:   {traj.n_valid}  ({traj.n_valid/traj.n_frames*100:.1f}%)")
    print(f"  FPS:            {traj.fps:.1f}")
    print(f"  Duration:       {traj.n_frames/traj.fps:.1f}s")
    if traj.n_valid > 0:
        v = traj.valid
        gc = traj.gripper_center[v]
        print(f"  Gripper centre range (table frame, metres):")
        print(f"    X: {gc[:,0].min():.3f} to {gc[:,0].max():.3f}")
        print(f"    Y: {gc[:,1].min():.3f} to {gc[:,1].max():.3f}")
        print(f"    Z: {gc[:,2].min():.3f} to {gc[:,2].max():.3f}")
        gw = traj.gripper_width[v]
        print(f"  Gripper width:  {gw.min()*100:.1f}cm to {gw.max()*100:.1f}cm")
    print(f"\nRaw trajectory saved to: {args.output}")


if __name__ == "__main__":
    main()
