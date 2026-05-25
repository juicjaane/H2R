from __future__ import annotations

import cv2
import numpy as np


def write_video(frames: list[np.ndarray], output_path: str, fps: float) -> None:
    """
    Save a list of BGR frames to an MP4 video file.
    
    Args:
        frames: List of HxWx3 uint8 numpy arrays (all must be same shape).
        output_path: Path to the output .mp4 file.
        fps: Frames per second.
    """
    if not frames:
        print(f"[writer] Warning: No frames provided for {output_path}")
        return

    h, w = frames[0].shape[:2]
    
    # mp4v is broadly compatible
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    try:
        report_every = max(1, len(frames) // 10)
        for i, frame in enumerate(frames):
            writer.write(frame)
            if (i + 1) % report_every == 0:
                print(f"  [writer] {((i + 1) / len(frames) * 100):5.1f}%  written {i+1}/{len(frames)}", end="\r")
        print(f"\n[writer] Saved {len(frames)} frames to {output_path}")
    finally:
        writer.release()
