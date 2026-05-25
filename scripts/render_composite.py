import argparse
import sys
import json
import numpy as np
import cv2
from pathlib import Path

# Fix sys.path for importing modules
sys.path.append(str(Path(__file__).parent.parent))

from src.render.composite import (
    make_rgb_panel,
    make_depth_panel,
    make_topdown_panel,
    make_sim_panel,
    tile_2x2,
    add_frame_counter,
)
from src.render.writer import write_video
from src.render.mujoco_renderer import render_trajectory
from src.tracking.depth_model import MetricDepthModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/take1.mp4")
    parser.add_argument("--traj", type=str, default="data/take1_smoothed.npz")
    parser.add_argument("--joints", type=str, default="data/take1_joints.npz")
    parser.add_argument("--calib", type=str, default="data/calibration.json")
    parser.add_argument("--placement", type=str, default="data/take1_placement.json")
    parser.add_argument("--output", type=str, default="outputs/composite.mp4")
    parser.add_argument("--model-xml", type=str, default="robot/panda.xml")
    parser.add_argument("--camera", type=str, default="side")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of frames to render (for quick testing)")
    
    args = parser.parse_args()
    
    # 1. Load configuration and metadata
    print("[composite] Loading metadata...")
    with open(args.calib, "r") as f:
        calib = json.load(f)
        
    T_cam_to_table = np.array(calib["T_cam_to_table"])
    T_table_to_cam = np.linalg.inv(T_cam_to_table)
    intrinsics = calib["intrinsics"]
    
    corners_cam_h = np.column_stack([np.array(calib["corners_cam"]), np.ones(4)])
    corners_table = (T_cam_to_table @ corners_cam_h.T).T[:, :3]
    
    # 2. Load trajectories
    traj_data = np.load(args.traj)
    gripper_traj = traj_data["gripper_center"]
    valid_mask = traj_data["valid"]
    fps = float(traj_data["fps"].item() if traj_data["fps"].ndim > 0 else traj_data["fps"])
    
    # 3. Load joints
    joints_data = np.load(args.joints)
    joint_angles = joints_data["joint_angles"]
    gripper = joints_data["gripper"]
    
    with open(args.placement, "r") as f:
         base_pos = np.array(json.load(f)["base_pos_table_frame"])

    # 4. Render Sim frames (Panel 4)
    print(f"[composite] Rendering MuJoCo '{args.camera}' view frames...")
    sim_frames = render_trajectory(
        joint_angles=joint_angles,
        gripper=gripper,
        model_path=args.model_xml,
        camera=args.camera,
        width=1280,  # Full resolution, will be scale down in make_sim_panel if needed. Wait, we want it PANEL_W, PANEL_H maybe?
        height=720,
        progress=True
    )
    
    # 5. Process RGB Video and generate composite
    print("[composite] Processing RGB frames & Depth estimation...")
    depth_model = MetricDepthModel()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
        
    composite_frames = []
    
    N = len(sim_frames)
    if args.limit > 0:
        N = min(N, args.limit)
    
    for current_idx in range(N):
        ret, bgr_frame = cap.read()
        if not ret:
            break
            
        is_valid = bool(valid_mask[current_idx])
        frame_data = {
            "thumb_tip": traj_data["thumb_tip"][current_idx],
            "index_tip": traj_data["index_tip"][current_idx],
            "wrist": traj_data["wrist"][current_idx],
            "gripper_center": traj_data["gripper_center"][current_idx],
        }
        
        # P1: RGB
        p1 = make_rgb_panel(bgr_frame, frame_data, T_table_to_cam, intrinsics, is_valid)
        
        # P2: Depth
        depth_m = depth_model.infer_frame(bgr_frame)
        p2 = make_depth_panel(depth_m)
        
        # P3: Top down
        p3 = make_topdown_panel(gripper_traj, valid_mask, current_idx, base_pos, corners_table)
        
        # P4: Sim
        p4 = make_sim_panel(sim_frames[current_idx])
        
        comp = tile_2x2(p1, p2, p3, p4)
        add_frame_counter(comp, current_idx, N, fps)
        
        composite_frames.append(comp)
        
        report_every = max(1, N // 20)
        if (current_idx + 1) % report_every == 0:
            print(f"  [composite] {((current_idx + 1) / N * 100):5.1f}%  frame {current_idx+1}/{N}", end="\r")
            
    print("\n[composite] Composite frames assembled.")
    
    cap.release()
    
    # 6. Write final video
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_video(composite_frames, args.output, fps)

if __name__ == "__main__":
    main()
