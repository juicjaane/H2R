"""
Phase 7: Render a MuJoCo simulation video from a joint trajectory.

Usage:
    python scripts/render_sim.py --joints data/take1_joints.npz
    python scripts/render_sim.py --joints data/take1_joints.npz --camera top --output outputs/take1_top.mp4
    python scripts/render_sim.py --joints data/take1_joints.npz --smoke-test
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

import src.config as cfg
from src.ik.trajectory_solver import load_joint_trajectory
from src.render.mujoco_renderer import render_frame, render_trajectory, CAMERA_PRESETS


def write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"[render] Wrote {len(frames)} frames to {path}")


def main():
    parser = argparse.ArgumentParser(description="Render Panda simulation from joint trajectory.")
    parser.add_argument("--joints",     type=Path, required=True,
                        help="Joint trajectory .npz from solve_ik.py")
    parser.add_argument("--camera",     default="side",
                        choices=list(CAMERA_PRESETS.keys()),
                        help="Camera preset (default: side)")
    parser.add_argument("--width",      type=int, default=1280)
    parser.add_argument("--height",     type=int, default=720)
    parser.add_argument("--output",     type=Path, default=None,
                        help="Output .mp4 (default: outputs/<stem>_<camera>.mp4)")
    parser.add_argument("--model",      type=Path, default=cfg.ROBOT_XML,
                        help="Path to panda.xml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Render only frame 0, save as .png, then exit")
    args = parser.parse_args()

    if not args.joints.exists():
        print(f"ERROR: joints file not found: {args.joints}")
        sys.exit(1)
    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    traj = load_joint_trajectory(args.joints)
    ja   = traj["joint_angles"]    # (N, 7)
    grip = traj["gripper"]         # (N,)
    fps  = traj["fps"]
    N    = traj["n_frames"]
    n_ok = int(traj["ik_success"].sum())
    print(f"Loaded {N} frames  ({n_ok}/{N} IK success)  fps={fps:.1f}")
    print(f"Camera: {args.camera}  Resolution: {args.width}×{args.height}")

    # ── Smoke test ────────────────────────────────────────────────────────────
    if args.smoke_test:
        import mujoco
        stem   = args.joints.stem.replace("_joints", "")
        out_png = cfg.OUTPUTS_DIR / f"{stem}_{args.camera}_frame0.png"
        cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        model = mujoco.MjModel.from_xml_path(str(args.model))
        data  = mujoco.MjData(model)
        frame = render_frame(model, data,
                             joint_q=ja[0], gripper_m=grip[0],
                             camera=args.camera,
                             width=args.width, height=args.height)
        cv2.imwrite(str(out_png), frame)
        print(f"[smoke-test] Saved frame 0 -> {out_png}")
        return

    # ── Full render ───────────────────────────────────────────────────────────
    stem = args.joints.stem.replace("_joints", "")
    if args.output is None:
        args.output = cfg.OUTPUTS_DIR / f"{stem}_{args.camera}.mp4"

    frames = render_trajectory(
        joint_angles=ja,
        gripper=grip,
        model_path=str(args.model),
        camera=args.camera,
        width=args.width,
        height=args.height,
    )
    write_mp4(frames, args.output, fps)


if __name__ == "__main__":
    main()
