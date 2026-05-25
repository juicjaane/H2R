"""
Record a manipulation video from the webcam.

Usage:
    python scripts/record.py --output data/take1.mp4
    python scripts/record.py --output data/take1.mp4 --camera 1
    python scripts/record.py --output data/take1.mp4 --resolution 1280x720

A tkinter control panel lets you start/stop recording.
Metadata (resolution, fps, intrinsics) is saved alongside the video.
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import tkinter as tk

import src.config as cfg
from src.config import DATA_DIR


SUPPORTED_RESOLUTIONS = {
    "1920x1080": (1920, 1080),
    "1280x720":  (1280, 720),
    "640x480":   (640,  480),
}


def record(
    output_path: Path,
    camera_index: int = 0,
    resolution: str = "1280x720",
):
    if resolution not in SUPPORTED_RESOLUTIONS:
        print(f"Unsupported resolution '{resolution}'. Choose from: {list(SUPPORTED_RESOLUTIONS)}")
        sys.exit(1)

    rw, rh = SUPPORTED_RESOLUTIONS[resolution]
    intr = cfg.get_intrinsics(rw, rh)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  rw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_res = f"{actual_w}x{actual_h}"

    if actual_res != resolution:
        print(f"Warning: requested {resolution} but camera gave {actual_res}. "
              "Intrinsics will be derived from actual resolution.")
        try:
            intr = cfg.get_intrinsics(actual_w, actual_h)
        except ValueError:
            print(f"No preset intrinsics for {actual_res}. Using {resolution} intrinsics with incorrect cx/cy.")

    fps_target = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_target, (actual_w, actual_h))

    # ── Tkinter control panel ─────────────────────────────────────────────────
    recording = False
    running   = True
    frame_count = [0]

    def start():
        nonlocal recording
        recording = True
        start_btn.config(state=tk.DISABLED)
        stop_btn.config(state=tk.NORMAL)
        print("[Record] Recording started.")

    def stop():
        nonlocal running
        running = False
        print("[Record] Stopped.")

    root = tk.Tk()
    root.title("Recording Control")
    root.geometry("260x130")
    start_btn = tk.Button(root, text="Start Recording", command=start,
                          font=("Arial", 12), bg="lightgreen")
    start_btn.pack(pady=8, fill=tk.X, padx=20)
    stop_btn  = tk.Button(root, text="Stop Recording", command=stop,
                          font=("Arial", 12), bg="salmon", state=tk.DISABLED)
    stop_btn.pack(pady=4, fill=tk.X, padx=20)
    status_lbl = tk.Label(root, text="Press 'Start' when ready.", font=("Arial", 9))
    status_lbl.pack(pady=4)

    print(f"[Record] Camera {actual_w}x{actual_h}  |  Output: {output_path}")
    print("  Click 'Start Recording' in the control panel. Press Q in the preview to stop.")

    while running:
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break

        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if recording:
            writer.write(frame)
            frame_count[0] += 1
            cv2.circle(display, (30, 30), 14, (0, 0, 220), cv2.FILLED)
            cv2.putText(display, f"REC  {frame_count[0]}f", (52, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
            status_lbl.config(text=f"Recording... {frame_count[0]} frames")
        else:
            cv2.putText(display, "PREVIEW — click Start to record", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 0), 2)

        cv2.imshow("Recording Preview (Q to stop)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    try:
        root.destroy()
    except tk.TclError:
        pass

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if frame_count[0] == 0:
        print("[Record] No frames recorded.")
        output_path.unlink(missing_ok=True)
        return

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "video_path":    str(output_path),
        "resolution":    actual_res,
        "fps":           fps_target,
        "frame_count":   frame_count[0],
        "intrinsics":    intr,
        "camera_index":  camera_index,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Record] Saved {frame_count[0]} frames to {output_path}")
    print(f"[Record] Metadata saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Record a manipulation video.")
    parser.add_argument("--output", type=Path,
                        default=DATA_DIR / "take1.mp4",
                        help="Output .mp4 path (default: data/take1.mp4)")
    parser.add_argument("--camera",     type=int, default=0)
    parser.add_argument("--resolution", type=str, default="1280x720",
                        choices=list(SUPPORTED_RESOLUTIONS))
    args = parser.parse_args()
    record(args.output, args.camera, args.resolution)


if __name__ == "__main__":
    main()
