"""
Export animated GIF demo assets for the GitHub README.

Generates GIFs for every pipeline stage from existing outputs.

Usage:
    python scripts/export_demo_assets.py
    python scripts/export_demo_assets.py --video data/take1.mp4 --n-frames 60
    python scripts/export_demo_assets.py --start-frame 30 --n-frames 90 --every-n 3

Outputs written to media/:
    00_raw_input.gif          — raw webcam video loop
    01_hand_tracking.gif      — MediaPipe landmarks animated
    02_depth_map.gif          — metric depth (inferno) animated
    03_trajectory_3d.gif      — 3D trajectory building up frame by frame
    04_workspace_heatmap.gif  — trajectory trace appearing on coverage heatmap
    05_ik_success.gif         — IK success bar filling left-to-right
    06_simulation.gif         — MuJoCo simulation playback
    07_composite.gif          — full 4-panel composite (large, high quality)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import json
import imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.tracking.trajectory import TrajectoryData

MEDIA = Path(__file__).resolve().parent.parent / "media"
DATA  = Path(__file__).resolve().parent.parent / "data"
OUT   = Path(__file__).resolve().parent.parent / "outputs"
ROBOT = Path(__file__).resolve().parent.parent / "robot" / "panda.xml"

W, H = 640, 360      # per-panel GIF resolution
COMP_W, COMP_H = 800, 450    # composite GIF resolution (balance: visible 4-panel + <25 MB)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _resize(img: np.ndarray, w: int = W, h: int = H) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _inferno_frame(depth_m: np.ndarray, vmin: float = 0.2, vmax: float = 2.5) -> np.ndarray:
    norm = np.clip((depth_m - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap("inferno")
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb   # already RGB


def _save_gif(frames_rgb: list, name: str, fps: float = 15.0) -> None:
    path = MEDIA / name
    imageio.mimsave(str(path), frames_rgb, fps=fps, loop=0)
    mb = path.stat().st_size / 1e6
    print(f"  [saved] media/{name}  ({len(frames_rgb)} frames, {mb:.1f} MB)")


def _extract_raw_frames(video: Path, start: int, n: int, every: int) -> list:
    """Return a list of BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, count = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if count % every == 0:
            frames.append(f)
        count += 1
    cap.release()
    return frames


# ─── GIF generators ───────────────────────────────────────────────────────────

def gif_raw_input(frames_bgr: list, fps: float) -> None:
    print("\n[1/8] Raw input GIF")
    out = [_bgr_to_rgb(_resize(f)) for f in frames_bgr]
    _save_gif(out, "00_raw_input.gif", fps)


def gif_hand_tracking(frames_bgr: list, fps: float) -> None:
    print("\n[2/8] Hand tracking GIF")
    try:
        import mediapipe as mp
    except ImportError:
        print("  [skip] mediapipe not installed"); return

    hands  = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        model_complexity=1, min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )
    draw   = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    out = []
    for i, bgr in enumerate(frames_bgr):
        if i % 10 == 0:
            print(f"    tracking frame {i}/{len(frames_bgr)}", end="\r")
        h0, w0 = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        vis = bgr.copy()
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                draw.draw_landmarks(vis, lm,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    styles.get_default_hand_landmarks_style(),
                    styles.get_default_hand_connections_style())
            for idx, colour in [(4, (0, 140, 255)), (8, (0, 255, 80))]:
                pt = res.multi_hand_landmarks[0].landmark[idx]
                px, py = int(pt.x * w0), int(pt.y * h0)
                cv2.circle(vis, (px, py), 9, colour, -1)
                cv2.circle(vis, (px, py), 11, (255, 255, 255), 2)
        out.append(_bgr_to_rgb(_resize(vis)))
    print()
    hands.close()
    _save_gif(out, "01_hand_tracking.gif", fps)


def gif_depth_map(frames_bgr: list, fps: float) -> None:
    print("\n[3/8] Metric depth GIF")
    try:
        from src.tracking.depth_model import MetricDepthModel
    except Exception as e:
        print(f"  [skip] {e}"); return

    model = MetricDepthModel()
    out   = []
    for i, bgr in enumerate(frames_bgr):
        if i % 5 == 0:
            print(f"    depth frame {i}/{len(frames_bgr)}", end="\r")
        depth = model.infer_frame(bgr)
        inf   = _inferno_frame(depth)
        # Convert depth frame to same size as others
        h0, w0 = bgr.shape[:2]
        inf_resized = cv2.resize(inf, (w0, h0), interpolation=cv2.INTER_AREA)
        # Add min/max labels
        cv2.putText(inf_resized, "0.2 m", (8, h0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(inf_resized, "2.5 m", (w0 - 65, h0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        out.append(_resize(inf_resized))
    print()
    del model
    _save_gif(out, "02_depth_map.gif", fps)


def gif_trajectory_3d(traj_path: Path) -> None:
    print("\n[4/8] 3D trajectory build-up GIF")
    traj = TrajectoryData.load(traj_path)
    gc   = traj.gripper_center[traj.valid]
    n    = len(gc)

    fig = plt.figure(figsize=(7, 5), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")

    colours = plt.cm.plasma(np.linspace(0, 1, n))
    # Fixed axis limits
    pad = 0.05
    xl = (gc[:, 0].min() - pad, gc[:, 0].max() + pad)
    yl = (gc[:, 1].min() - pad, gc[:, 1].max() + pad)
    zl = (gc[:, 2].min() - pad, gc[:, 2].max() + pad)

    def _style_ax():
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
        ax.set_xlabel("X (m)", color="white", labelpad=4, fontsize=8)
        ax.set_ylabel("Y (m)", color="white", labelpad=4, fontsize=8)
        ax.set_zlabel("Z (m)", color="white", labelpad=4, fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.set_title("Gripper Trajectory — Table Frame", color="white", fontsize=10, pad=8)

    # Keyframes: step through in chunks so total ≤ 60 animation frames
    step       = max(1, n // 50)
    keyframes  = list(range(0, n, step)) + [n - 1]
    gif_fps    = 12

    frames_out = []
    for end_i in keyframes:
        ax.cla()
        _style_ax()
        # Azimuth rotates slowly
        angle = 30 + (end_i / n) * 60
        ax.view_init(elev=20, azim=angle)
        if end_i > 0:
            for i in range(end_i):
                ax.plot(gc[i:i+2, 0], gc[i:i+2, 1], gc[i:i+2, 2],
                        color=colours[i], linewidth=1.5, alpha=0.8)
        ax.scatter(*gc[0], color="lime", s=40, zorder=5)
        if end_i > 0:
            ax.scatter(*gc[end_i], color="red", s=40, zorder=5)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames_out.append(_resize(buf))

    plt.close(fig)
    _save_gif(frames_out, "03_trajectory_3d.gif", gif_fps)


def gif_workspace_heatmap(placement_path: Path, traj_path: Path) -> None:
    print("\n[5/8] Workspace heatmap GIF")
    traj   = TrajectoryData.load(traj_path)
    gc     = traj.gripper_center[traj.valid]

    from src.ik.workspace import analyze_workspace, find_robot_placement
    stats  = analyze_workspace(traj)
    result = find_robot_placement(traj, stats)
    heatmap, extent = result.search_grid, result.grid_extent
    base_xy = result.base_pos[:2]

    n      = len(gc)
    step   = max(1, n // 50)
    keys   = list(range(0, n, step)) + [n - 1]
    gif_fps = 12

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    frames_out = []
    for end_i in keys:
        ax.cla()
        ax.set_facecolor("#1a1a2e")
        im = ax.imshow(heatmap, origin="lower", cmap="viridis",
                       extent=extent, aspect="auto", vmin=0, vmax=1)
        # Trajectory trace up to end_i
        if end_i > 0:
            t_norm = np.linspace(0, 1, end_i)
            sc = ax.scatter(gc[:end_i, 0], gc[:end_i, 1],
                            c=t_norm, cmap="cool", s=4, alpha=0.55, zorder=3)
        # Robot marker + reach circles
        ax.scatter(*base_xy, c="red", s=120, zorder=5, marker="x", linewidths=2.5)
        for r, col, ls in [(0.855, "lime", "--"), (0.170, "orange", "--")]:
            circle = plt.Circle(base_xy, r, color=col, fill=False,
                                linewidth=1.5, linestyle=ls)
            ax.add_patch(circle)
        ax.set_xlabel("X — table width (m)", color="white", fontsize=8)
        ax.set_ylabel("Y — table depth (m)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        pct = end_i / (n - 1) * 100 if n > 1 else 100
        ax.set_title(
            f"Workspace Coverage Search  |  Best: {result.coverage*100:.1f}%",
            color="white", fontsize=9, pad=6
        )
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames_out.append(_resize(buf))

    plt.close(fig)
    _save_gif(frames_out, "04_workspace_heatmap.gif", gif_fps)


def gif_ik_success(joints_path: Path) -> None:
    print("\n[6/8] IK success timeline GIF")
    data    = np.load(joints_path)
    success = data["ik_success"].astype(bool)
    fps_vid = float(data.get("fps", 30))
    n       = len(success)
    t       = np.arange(n) / fps_vid

    step    = max(1, n // 60)
    keys    = list(range(0, n, step)) + [n - 1]
    gif_fps = 12

    fig, ax = plt.subplots(figsize=(10, 2.2), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    frames_out = []
    for end_i in keys:
        ax.cla(); ax.set_facecolor("#1a1a2e")
        for i in range(min(end_i + 1, n)):
            c = "#00c853" if success[i] else "#d50000"
            ax.axvspan(t[i], t[i] + 1 / fps_vid, color=c, alpha=0.75)
        rate = success[:end_i + 1].mean() * 100 if end_i >= 0 else 0
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_xlabel("Time (s)", color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.set_title(
            f"IK Success per Frame  |  {rate:.1f}% solved  "
            f"|  green = <4 cm error   red = carry-forward",
            color="white", fontsize=9, pad=6
        )
        for spine in ax.spines.values():
            spine.set_color("#444")
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames_out.append(_resize(buf, W, 200))

    plt.close(fig)
    _save_gif(frames_out, "05_ik_success.gif", gif_fps)


def gif_simulation(joints_path: Path, model_path: Path, camera: str, fps: float) -> None:
    print("\n[7/8] Simulation GIF")
    import mujoco
    from src.render.mujoco_renderer import render_frame

    data         = np.load(joints_path)
    joint_angles = data["joint_angles"]
    gripper      = data["gripper"]
    n            = len(joint_angles)

    model  = mujoco.MjModel.from_xml_path(str(model_path))
    mjdata = mujoco.MjData(model)

    step = max(1, n // 60)
    frames_out = []
    for i in range(0, n, step):
        if i % 20 == 0:
            print(f"    sim frame {i}/{n}", end="\r")
        img = render_frame(model, mjdata, joint_angles[i], gripper[i],
                           camera=camera, width=W, height=H)
        frames_out.append(_bgr_to_rgb(img))
    print()
    _save_gif(frames_out, "06_simulation.gif", fps)


def gif_composite(composite_video: Path) -> None:
    print("\n[8/8] Composite GIF (full size)")
    if not composite_video.exists():
        print(f"  [skip] {composite_video} not found"); return

    cap   = cv2.VideoCapture(str(composite_video))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample every 5th frame — keeps GIF under ~25 MB even for long composites
    every = 5
    frames_out = []
    count = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if count % every == 0:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            # Keep composite at full 1280x720 for maximum impact
            rgb = cv2.resize(rgb, (COMP_W, COMP_H), interpolation=cv2.INTER_AREA)
            frames_out.append(rgb)
        count += 1
    cap.release()
    _save_gif(frames_out, "07_composite.gif", fps / every)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export all README demo GIFs to media/")
    parser.add_argument("--video",     type=Path, default=DATA / "take1.mp4")
    parser.add_argument("--smoothed",  type=Path, default=DATA / "take1_smoothed.npz")
    parser.add_argument("--joints",    type=Path, default=DATA / "take1_joints.npz")
    parser.add_argument("--placement", type=Path, default=DATA / "take1_placement.json")
    parser.add_argument("--composite", type=Path, default=OUT  / "take1_composite.mp4")
    parser.add_argument("--model",     type=Path, default=ROBOT)
    parser.add_argument("--camera",    type=str,  default="side",
                        choices=["side", "top", "front"])
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start frame in video for raw/tracking/depth GIFs (default 0)")
    parser.add_argument("--n-frames", type=int, default=90,
                        help="Number of source video frames to process (default 90 = 3s @ 30fps)")
    parser.add_argument("--every-n",  type=int, default=2,
                        help="Use every Nth source frame in output GIFs (default 2 → 15fps from 30fps video)")
    parser.add_argument("--skip-depth", action="store_true",
                        help="Skip depth GIF (slowest step — requires GPU + DAV2)")
    args = parser.parse_args()

    MEDIA.mkdir(exist_ok=True)
    print(f"Output directory: {MEDIA}")

    missing = [p for p in [args.video, args.smoothed, args.joints, args.placement]
               if not p.exists()]
    if missing:
        for p in missing:
            print(f"  [ERROR] not found: {p}")
        print("\nRun the full pipeline first:")
        print("  python pipeline.py process data/take1.mp4")
        print("  python pipeline.py simulate data/take1.mp4")
        raise SystemExit(1)

    # Determine source FPS
    cap = cv2.VideoCapture(str(args.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    gif_fps = src_fps / args.every_n

    print(f"Source FPS: {src_fps:.0f}   every-n: {args.every_n}   "
          f"GIF FPS: {gif_fps:.1f}   frames to process: {args.n_frames}")

    # Extract raw frames once — reuse for all three video-based GIFs
    print(f"\nExtracting {args.n_frames} frames starting at frame {args.start_frame}...")
    frames_bgr = _extract_raw_frames(args.video, args.start_frame,
                                     args.n_frames, args.every_n)
    print(f"  -> {len(frames_bgr)} frames extracted")

    gif_raw_input(frames_bgr, gif_fps)
    gif_hand_tracking(frames_bgr, gif_fps)
    if not args.skip_depth:
        gif_depth_map(frames_bgr, gif_fps)
    else:
        print("\n[3/8] Depth GIF — skipped (--skip-depth)")

    gif_trajectory_3d(args.smoothed)
    gif_workspace_heatmap(args.placement, args.smoothed)
    gif_ik_success(args.joints)
    gif_simulation(args.joints, args.model, args.camera, gif_fps)
    gif_composite(args.composite)

    print(f"\nAll GIFs written to {MEDIA}")
    print("Next: git add media/ && git commit -m 'add demo GIFs' && git push")


if __name__ == "__main__":
    main()
