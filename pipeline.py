"""
Hand-to-Robot Simulation Pipeline — Main Entrypoint

Subcommands:
  record      Interactive calibration + video recording (requires camera)
  process     Extract and smooth hand trajectory from a recorded video
  simulate    Solve IK and render simulation video(s)
  run         Full pipeline: process + simulate (calibration must exist)

Examples:
  python pipeline.py record
  python pipeline.py process data/take1.mp4
  python pipeline.py simulate data/take1.mp4
  python pipeline.py run data/take1.mp4
  python pipeline.py run data/take1.mp4 --skip-composite --camera top
"""

import sys
import subprocess
import argparse
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

SCRIPTS = Path(__file__).resolve().parent / "scripts"


def _banner(title: str) -> None:
    width = 60
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def _run(args: list[str | Path], step_name: str) -> None:
    """Run a script via subprocess, forwarding stdout/stderr. Exit on failure."""
    cmd = [sys.executable] + [str(a) for a in args]
    print(f"\n[pipeline] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[pipeline] ERROR: '{step_name}' failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def _stem(video: Path) -> str:
    return video.stem


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_record(args: argparse.Namespace) -> None:
    """Interactive: record a video session and calibrate the table surface."""
    _banner("RECORD — Camera recording + table calibration")

    calib_out = Path(args.data_dir) / "calibration.json"
    video_out  = Path(args.data_dir) / f"{args.name}.mp4"

    print("Step 1/2  Table surface calibration")
    _run([SCRIPTS / "calibrate.py",
          "--output", calib_out,
          "--camera", str(args.camera_index)],
         "calibrate")

    print("\nStep 2/2  Video recording")
    _run([SCRIPTS / "record.py",
          "--output", video_out,
          "--camera", str(args.camera_index)],
         "record")

    print(f"\n[pipeline] Done. Video: {video_out}  Calibration: {calib_out}")
    print(f"  Next: python pipeline.py process {video_out}")


def cmd_process(args: argparse.Namespace) -> None:
    """Extract trajectory, smooth it, and find the optimal robot placement."""
    video  = Path(args.video)
    stem   = _stem(video)
    ddir   = Path(args.data_dir)
    calib  = Path(args.calibration)

    raw_out       = ddir / f"{stem}_raw.npz"
    smoothed_out  = ddir / f"{stem}_smoothed.npz"
    placement_out = ddir / f"{stem}_placement.json"

    _banner(f"PROCESS — {video.name}")

    if not video.exists():
        print(f"ERROR: video not found: {video}"); sys.exit(1)
    if not calib.exists():
        print(f"ERROR: calibration not found: {calib}")
        print("Run:  python pipeline.py record"); sys.exit(1)

    print("Step 1/3  Extract trajectory")
    _run([SCRIPTS / "extract_trajectory.py",
          "--video",       video,
          "--calibration", calib,
          "--output",      raw_out],
         "extract_trajectory")

    print("Step 2/3  Smooth trajectory")
    _run([SCRIPTS / "smooth_trajectory.py",
          "--raw",    raw_out,
          "--output", smoothed_out],
         "smooth_trajectory")

    print("Step 3/3  Workspace analysis & robot placement")
    _run([SCRIPTS / "analyze_workspace.py",
          "--smoothed", smoothed_out,
          "--output",   placement_out],
         "analyze_workspace")

    print(f"\n[pipeline] Process complete.")
    print(f"  Smoothed: {smoothed_out}")
    print(f"  Placement: {placement_out}")
    print(f"  Next: python pipeline.py simulate {video}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Solve IK and render simulation + optional composite video."""
    video  = Path(args.video)
    stem   = _stem(video)
    ddir   = Path(args.data_dir)
    odir   = Path(args.output_dir)
    calib  = Path(args.calibration)

    smoothed_out  = ddir / f"{stem}_smoothed.npz"
    placement_out = ddir / f"{stem}_placement.json"
    joints_out    = ddir / f"{stem}_joints.npz"
    sim_out       = odir / f"{stem}_{args.camera}.mp4"
    composite_out = odir / f"{stem}_composite.mp4"

    _banner(f"SIMULATE — {video.name}")

    for p, name in [(smoothed_out, "smoothed trajectory"),
                    (placement_out, "robot placement")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            print(f"Run:  python pipeline.py process {video}"); sys.exit(1)

    total_steps = 3 if args.skip_composite else 3
    step = 1

    print(f"Step {step}/{total_steps}  Solve IK")
    _run([SCRIPTS / "solve_ik.py",
          "--smoothed",  smoothed_out,
          "--placement", placement_out,
          "--output",    joints_out],
         "solve_ik")
    step += 1

    print(f"Step {step}/{total_steps}  Render simulation")
    _run([SCRIPTS / "render_sim.py",
          "--joints", joints_out,
          "--camera", args.camera,
          "--output", sim_out],
         "render_sim")
    step += 1

    if not args.skip_composite:
        print(f"Step {step}/{total_steps}  Render composite video")
        _run([SCRIPTS / "render_composite.py",
              "--video",     video,
              "--traj",      smoothed_out,
              "--joints",    joints_out,
              "--calib",     calib,
              "--placement", placement_out,
              "--output",    composite_out,
              "--camera",    args.camera],
             "render_composite")

    print(f"\n[pipeline] Simulate complete.")
    print(f"  Simulation: {sim_out}")
    if not args.skip_composite:
        print(f"  Composite:  {composite_out}")


def cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline: process + simulate."""
    _banner(f"RUN (full pipeline) — {Path(args.video).name}")
    cmd_process(args)
    cmd_simulate(args)
    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hand-to-Robot simulation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared optional flags (added per subcommand below)
    def add_common(p):
        p.add_argument("--data-dir",     default="data",    metavar="DIR",
                       help="Directory for intermediate files (default: data/)")
        p.add_argument("--output-dir",   default="outputs", metavar="DIR",
                       help="Directory for output videos (default: outputs/)")
        p.add_argument("--calibration",  default="data/calibration.json", metavar="PATH",
                       help="Path to calibration.json (default: data/calibration.json)")
        p.add_argument("--camera",       default="side", choices=["side", "top", "front"],
                       help="Simulation camera preset (default: side)")

    # record
    p_rec = sub.add_parser("record", help="Calibrate table + record video")
    p_rec.add_argument("--name",          default="take1",  help="Base name for output files")
    p_rec.add_argument("--camera-index",  type=int, default=0, help="OpenCV camera index")
    p_rec.add_argument("--data-dir",      default="data")
    p_rec.set_defaults(func=cmd_record)

    # process
    p_proc = sub.add_parser("process", help="Extract + smooth trajectory from video")
    p_proc.add_argument("video", type=str, help="Path to recorded .mp4")
    add_common(p_proc)
    p_proc.set_defaults(func=cmd_process)

    # simulate
    p_sim = sub.add_parser("simulate", help="Solve IK + render simulation")
    p_sim.add_argument("video", type=str, help="Path to recorded .mp4")
    p_sim.add_argument("--skip-composite", action="store_true",
                       help="Skip the 4-panel composite video (faster)")
    add_common(p_sim)
    p_sim.set_defaults(func=cmd_simulate)

    # run
    p_run = sub.add_parser("run", help="Full pipeline: process + simulate")
    p_run.add_argument("video", type=str, help="Path to recorded .mp4")
    p_run.add_argument("--skip-composite", action="store_true")
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    return parser


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
