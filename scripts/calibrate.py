"""
Run surface calibration — touch index fingertip to 4 table corners.

Usage:
    python scripts/calibrate.py
    python scripts/calibrate.py --output data/my_session.json
    python scripts/calibrate.py --camera 1   # if not default camera
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.calibration.ui import run_calibration


def main():
    parser = argparse.ArgumentParser(description="Surface calibration for hand-to-robot pipeline.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to save calibration.json (default: data/calibration.json)")
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    args = parser.parse_args()

    path = run_calibration(output_path=args.output, camera_index=args.camera)
    if path:
        print(f"\nCalibration complete. File saved to: {path}")
        print("You can now record a session. The pipeline will use this calibration file.")
    else:
        print("\nCalibration did not complete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
