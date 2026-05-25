# H2R — Human to Robot

> A simple experiment to generate robot imitation learning data by bypassing teleoperation.

Convert a monocular video of a human hand performing a table-top manipulation task into a MuJoCo simulation of a Franka Panda robot performing the same motion — no teleoperation hardware required. The output is compatible with standard imitation learning frameworks (ACT, Diffusion Policy) as a drop-in replacement for teleoperated demonstrations.

> **Theory & design rationale** → [THEORY.md](THEORY.md)

```
Real-world video  →  Hand tracking  →  3D trajectory  →  IK  →  Panda simulation
```

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Quick Start](#quick-start)
4. [Pipeline Reference](#pipeline-reference)
5. [Component Guide](#component-guide)
   - [src/config.py](#srcconfigpy)
   - [src/calibration/](#srccalibration)
   - [src/tracking/](#srctracking)
   - [src/ik/](#srcik)
   - [src/render/](#srcrender)
6. [Data Files Reference](#data-files-reference)
7. [Configuration Tuning](#configuration-tuning)
8. [Known Limitations](#known-limitations)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What it does

1. **Calibrate** — the user touches four corners of a table with their index fingertip to establish a 3D coordinate frame for the working surface.
2. **Record** — capture a short video of a hand manipulation task (pinch grasp, move, place).
3. **Track** — for each frame, detect 21 hand landmarks via MediaPipe and lift them to metric 3D using Depth Anything V2 (indoor checkpoint).
4. **Smooth** — apply Savitzky-Golay filtering across the full sequence to remove depth noise.
5. **Place** — find the optimal Franka Panda base position (grid search over table XY) so the robot can reach all trajectory waypoints.
6. **IK** — solve inverse kinematics for every frame using damped least-squares, with warmstarting across frames and joint velocity clamping.
7. **Render** — replay the joint trajectory in a headless MuJoCo simulation and export `.mp4`.
8. **Composite** — combine original RGB, depth map, top-down trajectory view, and simulation into a single 2×2 panel video.

### Architecture

```
YUH/
├── pipeline.py                  ← main CLI entrypoint
├── src/
│   ├── config.py                ← all constants and paths
│   ├── calibration/
│   │   ├── surface.py           ← plane fitting, coordinate transforms
│   │   └── ui.py                ← interactive 4-corner calibration UI
│   ├── tracking/
│   │   ├── depth_model.py       ← Metric DAV2 wrapper
│   │   ├── hand_tracker.py      ← MediaPipe Hands wrapper
│   │   ├── trajectory.py        ← per-frame 3D extraction
│   │   └── smoother.py          ← Savitzky-Golay smoothing
│   ├── ik/
│   │   ├── solver.py            ← single-frame DLS IK
│   │   ├── trajectory_solver.py ← warmstarted per-frame IK + velocity clamp
│   │   └── workspace.py         ← workspace analysis, robot placement search
│   └── render/
│       ├── mujoco_renderer.py   ← headless MuJoCo render
│       ├── composite.py         ← 4-panel frame assembly
│       └── writer.py            ← cv2.VideoWriter wrapper
├── scripts/                     ← one script per pipeline phase
├── robot/
│   └── panda.xml                ← Franka Panda MuJoCo model + table geom
├── Depth-Anything-V2/           ← model code (read-only dependency)
├── checkpoints/                 ← model weights
├── data/                        ← videos, calibration, intermediate .npz files
└── outputs/                     ← final rendered videos
```

### Coordinate frames

| Frame | Convention | Used for |
|---|---|---|
| **Camera** | X=right, Y=down, Z=into scene (OpenCV) | MediaPipe 2D → 3D back-projection |
| **Table** | X=table width (TL→TR), Y=table depth, Z=up | All trajectory analysis |
| **Robot base** | Same axes as table, origin at robot base | IK target positions |
| **MuJoCo** | X=right, Y=forward, Z=up | Simulation rendering |

Transform chain: `camera → table → robot base`. Never skip a step.

---

## Prerequisites & Installation

### Hardware

- **GPU** — CUDA-capable (required for Depth Anything V2). Tested on NVIDIA.
- **Camera** — Logitech Brio 100 recommended (intrinsics pre-computed). Other cameras require adding intrinsics to `src/config.py`.
- **OS** — Windows 10/11 (tested), Linux should work with minor path adjustments.

### Software dependencies

```
Python 3.10+
torch >= 2.0 (CUDA build)
mujoco >= 3.0
mediapipe >= 0.10
opencv-python
numpy
scipy
matplotlib
```

### Installation

```bash
# 1. Clone this repo
git clone https://github.com/Janeshvar/H2R.git
cd H2R

# 2. Install Python dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mujoco mediapipe opencv-python numpy scipy matplotlib

# 3. Clone Depth-Anything-V2 (read-only dependency)
git clone https://github.com/DepthAnything/Depth-Anything-V2 Depth-Anything-V2
pip install -r Depth-Anything-V2/requirements.txt

# 4. Download the metric depth checkpoint (Hypersim indoor, ~100MB)
python scripts/download_metric_checkpoint.py
```

The checkpoint will be saved to `checkpoints/depth_anything_v2_metric_hypersim_vits.pth`.

---

## Quick Start

### New recording

```bash
# Step 1 — calibrate the table surface and record a video
python pipeline.py record

# Step 2 — process the video (extract trajectory, smooth, find robot placement)
python pipeline.py process data/take1.mp4

# Step 3 — solve IK and render all outputs
python pipeline.py simulate data/take1.mp4
```

### Full pipeline in one command

```bash
# Assumes calibration already exists at data/calibration.json
python pipeline.py run data/take1.mp4
```

### Options

```bash
# Use top-down camera view for simulation
python pipeline.py simulate data/take1.mp4 --camera top

# Skip the 4-panel composite video (faster)
python pipeline.py simulate data/take1.mp4 --skip-composite

# Custom output directory
python pipeline.py run data/take1.mp4 --output-dir my_outputs/
```

---

## Pipeline Reference

Each subcommand maps to a set of individual scripts that can also be run standalone.

### `pipeline.py record`

| Script | What it does | Output |
|---|---|---|
| `scripts/calibrate.py` | Interactive: touch index fingertip to 4 table corners | `data/calibration.json` |
| `scripts/record.py` | Camera recording UI with live hand overlay | `data/<name>.mp4` |

### `pipeline.py process <video>`

| Script | What it does | Output |
|---|---|---|
| `scripts/extract_trajectory.py` | Run MediaPipe + DAV2 per frame, back-project to 3D, transform to table frame | `data/<stem>_raw.npz` |
| `scripts/smooth_trajectory.py` | Apply Savitzky-Golay to all trajectory dimensions | `data/<stem>_smoothed.npz` |
| `scripts/analyze_workspace.py` | Grid-search for optimal robot base placement | `data/<stem>_placement.json` |

### `pipeline.py simulate <video>`

| Script | What it does | Output |
|---|---|---|
| `scripts/solve_ik.py` | Warmstarted DLS IK for all frames | `data/<stem>_joints.npz` |
| `scripts/render_sim.py` | Headless MuJoCo render | `outputs/<stem>_<camera>.mp4` |
| `scripts/render_composite.py` | 4-panel composite video | `outputs/<stem>_composite.mp4` |

### Standalone scripts

```bash
# Validate depth model accuracy against a ruler
python scripts/validate_depth.py

# Plot 3D trajectory in matplotlib (useful for debugging)
python scripts/plot_trajectory.py --smoothed data/take1_smoothed.npz

# Render a single frame to PNG (fast smoke test)
python scripts/render_sim.py --joints data/take1_joints.npz --smoke-test

# Re-render with a different camera
python scripts/render_sim.py --joints data/take1_joints.npz --camera top
```

---

## Component Guide

### `src/config.py`

Central configuration imported by every module. Change constants here, not in individual files.

**Key constants:**

| Constant | Value | Meaning |
|---|---|---|
| `IK_GAIN` | 0.3 | DLS step scale. Lower = more stable, slower convergence. 0.3 beats 0.5 because smaller steps reduce velocity-clamping cascade. |
| `IK_DAMPING` | 1e-4 | λ in `Δq = Jᵀ(JJᵀ + λI)⁻¹ · err`. Prevents large steps near singularities. |
| `IK_ITERATIONS` | 50 | DLS iterations per frame. More iterations paradoxically hurt success rate because the solver moves further from the previous frame, increasing velocity clamping. |
| `PANDA_REACH_M` | 0.855 | Outer reachability sphere radius (metres). |
| `PANDA_DEADZONE_M` | 0.170 | Inner dead zone — too close to reach. |
| `PANDA_GRIPPER_MAX_M` | 0.08 | Maximum Panda gripper opening in metres. |
| `PANDA_MAX_JOINT_VEL` | 2.175 | Joint velocity limit in rad/s (from spec sheet). |
| `BRIO100_INTRINSICS` | dict | Pre-computed `fx, fy, cx, cy` for Logitech Brio 100 at three resolutions. Formula: `f = sqrt(w²+h²) / (2·tan(29°))`. |

**Camera intrinsics table** (Logitech Brio 100, 58° diagonal FOV):

| Resolution | fx = fy | cx | cy |
|---|---|---|---|
| 1920×1080 | 1986 | 960 | 540 |
| 1280×720 | 1324 | 640 | 360 |
| 640×480 | 722 | 320 | 240 |

> Never use `fx=600`. That was a wrong guess in the original prototype.

---

### `src/calibration/`

#### `surface.py` — Plane fitting and coordinate transforms

**`pixel_to_3d(px, py, depth_m, fx, fy, cx, cy)`**

Back-projects a pixel at known metric depth to a 3D point in camera frame using the pinhole model:
```
X = (px - cx) * depth / fx
Y = (py - cy) * depth / fy
Z = depth
```

**`fit_plane(points)`**

Fits a plane to N ≥ 3 points using SVD (least-squares). Returns the plane normal, origin (centroid), and per-point residuals. Residuals > 5mm trigger a warning — this usually means depth noise at the fingertip or a non-flat surface.

**`build_table_frame(corners)`**

Constructs the 4×4 homogeneous transform `T_cam_to_table` from four 3D corner points (TL, TR, BR, BL):
- **Origin** — centroid of four corners
- **X-axis** — TL → TR (table width direction)
- **Y-axis** — TL → BL (table depth direction)
- **Z-axis** — plane normal pointing toward camera (up)

**`cam_to_table(point_cam, T)`** / **`table_to_cam(point_table, T)`**

Simple homogeneous-coordinate transform wrappers.

#### `ui.py` — Interactive calibration

Walks the user through touching each of the four table corners with their index fingertip. For each corner, captures 10 frames and takes the **median depth** from a 5×5 patch to reduce depth noise at the fingertip silhouette. Saves the result to `data/calibration.json`.

---

### `src/tracking/`

#### `depth_model.py` — Metric depth estimation

Wraps **Depth Anything V2** (ViT-S, Hypersim indoor checkpoint). Returns depth in **metres** — not a disparity map. This is the critical difference from the original `v1.py` which used the relative model and hid the scale error with magic constants.

```python
model = MetricDepthModel()          # loads checkpoint once, ~100MB
depth = model.infer_frame(bgr)      # HxW float32, metres
z     = model.sample_patch(depth, px, py, radius=2)  # 5x5 median patch
```

> Always use `sample_patch()` for landmark depth — not `depth[py, px]`. Single-pixel sampling picks up background depth at fingertip silhouette boundaries.

**Temporal instability:** DAV2 re-estimates the entire scene from each frame independently. Observed standard deviation of 30–130mm at the same physical point across 20 frames. This is expected and is handled by Savitzky-Golay smoothing, not per-frame EMA.

#### `hand_tracker.py` — 2D landmark detection

Wraps **MediaPipe Hands** (model complexity 1, single hand mode). Returns a dict mapping landmark index → `(px, py)` pixel coordinates.

Key landmark indices used by this pipeline:

| ID | Name | Role |
|---|---|---|
| 0 | Wrist | Palm anchor |
| 4 | Thumb tip | Gripper finger 1 |
| 5 | Index MCP | Knuckle (palm normal calculation) |
| 8 | Index tip | Gripper finger 2 |
| 17 | Pinky MCP | Knuckle (palm normal calculation) |

MediaPipe is wrapped in `try/except ImportError` so scripts that only need to load saved trajectory data don't require MediaPipe to be installed.

#### `trajectory.py` — Per-frame 3D extraction

For each video frame:
1. Run MediaPipe → get 2D pixel positions
2. Run DAV2 → get depth map in metres
3. For each key landmark: sample 5×5 median patch depth → `pixel_to_3d` → 3D point in camera frame
4. Transform all points from camera frame to table frame via `T_cam_to_table`
5. Compute hand state:

| Field | Computation |
|---|---|
| `gripper_center` | Midpoint of thumb_tip and index_tip |
| `gripper_width` | 3D Euclidean distance between thumb_tip and index_tip, clipped to [0, 0.15m] |
| `grasp_axis` | Unit vector from thumb_tip to index_tip |
| `approach_vec` | Palm normal = `cross(index_MCP − wrist, pinky_MCP − wrist)` |

Saves raw trajectory to `.npz` including `valid` boolean mask (True when MediaPipe detected a hand).

#### `smoother.py` — Savitzky-Golay smoothing

Applies `scipy.signal.savgol_filter` across the full sequence (offline). Before smoothing, linearly interpolates across detection gaps so the filter doesn't produce artifacts at boundaries. Default: window=15 frames, polyorder=3.

> Use Savitzky-Golay for offline processing, never EMA. EMA introduces lag proportional to the smoothing window. SG has no lag because it uses all frames.

---

### `src/ik/`

#### `solver.py` — Single-frame damped least-squares IK

**Damped Least-Squares (DLS) IK:**

```
Δq = Jᵀ(JJᵀ + λI)⁻¹ · error
```

Where:
- `J` — 6×7 Jacobian (3 position rows + 3 orientation rows, 7 joint columns)
- `λ` — damping factor (`IK_DAMPING = 1e-4`), prevents large updates near singularities
- `error` — 6D task error: `[pos_target − pos_current, rot_error]`

Orientation error uses the skew-symmetric part of `R_target · R_current^T`, which gives a 3D axis-angle residual.

**`build_grasp_rotation(grasp_axis, approach_vec)`**

Constructs the target end-effector rotation matrix from the hand geometry:
- `grasp_axis` → EE X-axis (thumb-to-index direction)
- `approach_vec` → EE Z-axis (palm normal / approach direction)
- Y-axis computed as `Z × X`, then X is re-orthogonalised

**Success threshold:** A frame is marked successful if final position error < **4cm**. This was tuned experimentally — 2cm gave 77% success, 4cm gives 91% on the test trajectory.

#### `trajectory_solver.py` — Full trajectory IK

Solves IK for every frame with three key improvements over a naive per-frame approach:

1. **Warmstart** — each frame initialises from the previous frame's solution, not from the home pose. This is mandatory for smooth trajectories.

2. **Joint velocity clamping** — after IK, the joint change `Δq` is scaled down if it exceeds `PANDA_MAX_JOINT_VEL / fps` per joint. This enforces hardware speed limits.

3. **Carry-forward on failure** — if IK fails for a frame (error > 4cm), the previous frame's joint configuration is reused. This keeps the trajectory continuous even across unreachable regions.

**Why more iterations can hurt:** With 50 iterations, the DLS solver moves a moderate distance in joint space. With 150 iterations, it finds a more optimal solution that requires larger joint changes, which then gets velocity-clamped to an intermediate position. This intermediate position is a worse warmstart for the next frame, cascading into more failures.

#### `workspace.py` — Robot placement optimisation

**`analyze_workspace(traj)`** — computes bounding box, centroid, and max XY radius of the trajectory in table frame.

**`find_robot_placement(traj, stats)`** — grid search (40×40 = 1600 candidates) over the table XY plane. For each candidate base position, scores it as the fraction of trajectory waypoints that fall within the annular workspace (`inner_dead < dist < outer_reach`). Returns the position with the highest coverage score.

```
base_z = 0.0  →  robot base is at table surface height (table-mounted)
base_z = -0.75 →  floor-mounted robot (severely limits horizontal reach)
```

**`table_to_robot(point_table, base_pos_table)`** — translates a point from table frame to robot base frame. Since both frames share the same orientation (Z=up, X=width, Y=depth), this is a simple vector subtraction.

---

### `src/render/`

#### `mujoco_renderer.py` — Headless simulation rendering

Uses `mujoco.Renderer` (no display window required) with a programmatic `MjvCamera` instead of XML-defined cameras. This avoids needing to compute camera `xyaxes` vectors manually.

**Camera presets:**

| Name | Azimuth | Elevation | Distance | Lookat |
|---|---|---|---|---|
| `side` | 135° | −25° | 2.0m | [0.4, 0, 0.35] |
| `top` | 0° | −90° | 2.0m | [0.4, 0, 0] |
| `front` | 180° | −20° | 2.0m | [0.4, 0, 0.35] |

**Gripper rendering:** Gripper fingers are set directly via `qpos[7] = qpos[8] = gripper_m / 2`, then `mj_forward` computes the resulting geometry. No simulation is run — this is forward kinematics only.

**Offscreen framebuffer:** `robot/panda.xml` includes `<global offwidth="1920" offheight="1080"/>` to allow rendering at up to 1080p. Without this, MuJoCo defaults to a 640px buffer and will raise a `ValueError` for larger renders.

#### `composite.py` — 4-panel frame assembly

Assembles the 2×2 composite from four 640×360 BGR panels:

| Panel | Content | Key functions |
|---|---|---|
| Top-left | Original RGB + hand skeleton | `make_rgb_panel()` — back-projects 3D table-frame landmarks to pixel coords |
| Top-right | Metric depth (inferno colormap, 0.2–2.5m) | `make_depth_panel()` |
| Bottom-left | Top-down table view | `make_topdown_panel()` — trajectory path (blue→red), robot base, reach circles |
| Bottom-right | MuJoCo simulation | `make_sim_panel()` |

**Hand overlay back-projection** (`make_rgb_panel`): inverts `T_cam_to_table` to get `T_table_to_cam`, then applies the pinhole projection formula to draw thumb tip, index tip, wrist, and gripper centre directly onto the video frame.

**Top-down view:** Renders in table frame XY. Y axis is flipped so the near side of the table (small Y, close to the camera) appears at the bottom of the image. Trajectory points are coloured blue (early frames) → red (late frames). The robot base marker uses an `×` glyph; reach and deadzone circles are drawn around it.

#### `writer.py` — Video export

Thin wrapper around `cv2.VideoWriter` using the `mp4v` codec. Accepts a list of BGR numpy arrays and writes them to an `.mp4` file at the specified fps.

---

## Data Files Reference

All intermediate files are saved to `data/` by default.

| File | Format | Contents |
|---|---|---|
| `calibration.json` | JSON | `T_cam_to_table` (4×4), intrinsics, 4 corner positions in camera frame, plane residuals |
| `<stem>.meta.json` | JSON | Video resolution, fps, frame count, camera intrinsics used |
| `<stem>_raw.npz` | NumPy | Per-frame: `thumb_tip, index_tip, wrist, gripper_center, gripper_width, grasp_axis, approach_vec, valid` — all in table frame (metres) |
| `<stem>_smoothed.npz` | NumPy | Same structure as raw but Savitzky-Golay filtered; interpolated across detection gaps |
| `<stem>_placement.json` | JSON | `base_pos_table_frame` (3,), coverage fraction, n_reachable / n_total |
| `<stem>_joints.npz` | NumPy | `joint_angles` (N×7, rad), `gripper` (N, metres), `ik_success` (N, bool), `fps`, `n_frames` |

---

## Configuration Tuning

### IK success rate

The current configuration achieves **91.6% success** on the test trajectory. The remaining 8.4% are frames where the hand reaches the edge of the robot's physical workspace.

| Parameter | Location | Effect |
|---|---|---|
| `IK_GAIN` | `src/config.py` | Lower = more stable but slower per iteration. 0.3 is optimal; 0.5 works but increases velocity clamping cascade |
| `IK_ITERATIONS` | `src/config.py` | 50 is optimal. Counter-intuitively, more iterations reduce success rate |
| Success threshold | `src/ik/solver.py:135` | Currently 4cm. Lower = more precise but fewer accepted frames |

### Depth quality

The depth model has ±10% accuracy at typical operating distances (0.5–2m). For better depth quality:
- Ensure good lighting (avoid shadows directly on the hand)
- Keep the hand within 0.4–1.8m of the camera
- The 5×5 median patch in `depth_model.sample_patch()` handles fingertip silhouette noise

### Smoothing

To adjust trajectory smoothness, modify `scripts/smooth_trajectory.py` flags:

```bash
# Smoother (loses fast details)
python scripts/smooth_trajectory.py --raw data/take1_raw.npz --window 25

# Less smooth (preserves fast motions)
python scripts/smooth_trajectory.py --raw data/take1_raw.npz --window 9
```

The window must be odd and > polyorder (default 3).

---

## Known Limitations

1. **Workspace boundary failures** — 8.4% of frames are unreachable by the robot. These correspond to the portion of the trajectory where the hand is at maximum extension relative to the robot base. Carry-forward is used to fill these gaps.

2. **Orientation-aware reachability not modelled** — the placement search uses a sphere approximation. A point inside the sphere may still fail IK due to joint limits combined with the required end-effector orientation (palm facing down, fingertips forward). This is a known simplification.

3. **Velocity clamping artifacts** — 70%+ of frames are velocity-clamped to respect the Panda's 2.175 rad/s joint limit. This can produce slightly jerky motion in the simulation compared to the smooth hand trajectory. Noticeable in the first ~5 seconds of the trajectory.

4. **DAV2 temporal instability** — the depth model re-estimates the full scene per frame with no temporal memory. Depth at a fixed point has 30–130mm standard deviation across frames. Savitzky-Golay mitigates this but cannot fully eliminate it.

5. **Single-hand only** — MediaPipe is configured for `max_num_hands=1`. Multi-hand or bimanual tasks are out of scope.

6. **No object detection** — the pipeline does not detect or track objects being manipulated. Only the hand trajectory is captured.

---

## Troubleshooting

**`ValueError: Image width N > framebuffer width 640`**

The MuJoCo offscreen buffer is too small. `robot/panda.xml` should contain:
```xml
<visual>
  <global offwidth="1920" offheight="1080"/>
</visual>
```

**`ModuleNotFoundError: No module named 'mediapipe'`**

MediaPipe is only required for live tracking (calibration, recording, extraction). If you only need to re-run IK or rendering on existing `.npz` files, install it or ignore the warning — it is wrapped in a `try/except` and only fails when actually called.

**`ModuleNotFoundError: No module named 'depth_anything_v2'`**

The `Depth-Anything-V2` directory is missing or not cloned. Run:
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2 Depth-Anything-V2
```

**IK success rate is 0%**

Check `IK_GAIN` in `src/config.py`. A gain of 2.0 causes divergence when the initial error is > ~0.3m (the DLS updates oscillate). Set `IK_GAIN = 0.3` and `IK_ITERATIONS = 50`.

**Depth values are clearly wrong (all near 0 or > 10m)**

You may be using the standard (relative) DAV2 model instead of the metric Hypersim checkpoint. Verify the checkpoint path in `src/config.py`:
```python
DEPTH_METRIC_CKPT = CHECKPOINTS_DIR / "depth_anything_v2_metric_hypersim_vits.pth"
```
And that the model is loaded via `metric_depth/depth_anything_v2/dpt.py` with `max_depth=20`.

**Rendered simulation looks frozen for several seconds**

This is the carry-forward region — frames where IK failed (unreachable workspace positions). The robot holds its last valid joint configuration. This is expected for the first ~5 seconds of `take1`, which covers the peak-reach portion of the trajectory.

**Camera not found during recording**

Try a different camera index: `python pipeline.py record --camera-index 1`
