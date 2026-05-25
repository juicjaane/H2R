# LESSONS FOR CLAUDE — H2R Project Briefing

*A comprehensive debrief for future Claude sessions on this project. Read this before touching any code. It covers the full project state, every hard-won technical decision, every bug encountered, and the agreed path forward.*

---

## 1. Project Identity

**Name:** H2R — Human to Robot
**What it is:** A pipeline that converts a monocular webcam video of a human hand performing a tabletop manipulation task into a Franka Panda robot joint trajectory, bypassing teleoperation hardware entirely.
**Why it exists:** To generate imitation learning training data (joint angles + gripper state, N×7+1) cheaply and at scale — as a drop-in replacement for teleoperated demonstrations used in frameworks like ACT and Diffusion Policy.
**Current state:** Fully working, end-to-end (Phases 0–9 complete). Pushed to GitHub as public repo H2R under user Janeshvar.
**User background:** Janeshvar is new to robotics — this was his first robotics project. He is working toward robotics research and needs conceptual depth, not just implementation help.

---

## 2. The Full Pipeline (Phase by Phase)

```
Webcam video (take1.mp4)
    │
    ▼ Phase 2 — Hand Tracking
MediaPipe Hands → 21 2D landmarks per frame
    │
    ▼ Phase 1 — Depth Estimation
Depth Anything V2 (Hypersim indoor metric, ViT-S) → HxW depth map in metres
    │
    ▼ Phase 3 — Trajectory Extraction
Pinhole back-projection → 3D landmarks in camera frame
T_cam_to_table transform → 3D landmarks in table frame
save: data/take1_raw.npz
    │
    ▼ Phase 4 — Smoothing
Savitzky-Golay (window=15, poly=3) → denoised trajectory
Linear interpolation across MediaPipe detection gaps before SG
save: data/take1_smoothed.npz
    │
    ▼ Phase 5 — Workspace Analysis
Grid search (40×40=1600 candidates) for optimal Panda base XY
Score = fraction of waypoints in annular zone (inner=0.170m, outer=0.855m)
save: data/take1_placement.json
    │
    ▼ Phase 6 — IK Solving
DLS IK, warmstarted, joint velocity clamped
IK_GAIN=0.3, IK_DAMPING=1e-4, IK_ITERATIONS=50, threshold=4cm
save: data/take1_joints.npz (joint_angles N×7, gripper N, ik_success N)
    │
    ▼ Phase 7 — Simulation Rendering
Headless MuJoCo (mujoco.Renderer), programmatic MjvCamera
Presets: side/top/front
save: outputs/take1_side.mp4
    │
    ▼ Phase 8 — Composite Video
4-panel 1280×720 (RGB+skeleton, depth inferno, top-down trajectory, MuJoCo sim)
save: outputs/take1_composite.mp4
    │
    ▼ Phase 9 — Pipeline CLI
pipeline.py with subcommands: record / process / simulate / run
```

---

## 3. Achieved Performance Numbers

| Metric | Value | Notes |
|---|---|---|
| IK success rate | **91.6%** (305/333 frames) | 4cm threshold |
| IK success at 2cm threshold | 77.2% | Previous threshold, abandoned |
| IK success at 150 iterations | ~47.4% | More iterations HURT — see §6 |
| Depth temporal noise (std dev) | 30–130mm | At fixed point across frames |
| Depth accuracy | ±10% at 0.5–2m | DAV2 metric model |
| Grid search resolution | 40×40 = 1600 | ~2 min total processing |
| Trajectory smoothing displacement | mean ~1cm, max ~3cm raw→smooth | Per valid frame |
| Video: take1.mp4 | 333 frames at 30fps | ~11 seconds |

---

## 4. Key Configuration Values (src/config.py)

```python
IK_GAIN        = 0.3    # DO NOT change to 0.5+ — see §6
IK_DAMPING     = 1e-4
IK_ITERATIONS  = 50     # DO NOT increase — see §6
PANDA_REACH_M  = 0.855
PANDA_DEADZONE_M = 0.170
PANDA_GRIPPER_MAX_M = 0.08
PANDA_MAX_JOINT_VEL = 2.175   # rad/s, from hardware spec
BRIO100_INTRINSICS = { ... }  # pre-computed for 3 resolutions
```

**Critical:** IK success threshold is in `src/ik/solver.py:135`:
```python
success = final_pos_err < 0.04   # 4cm
```
Not in config.py — remember this when someone asks where the threshold lives.

---

## 5. File Map

```
H2R/
├── pipeline.py                  ← CLI entrypoint (subcommands: record/process/simulate/run)
├── src/
│   ├── config.py                ← ALL constants here, never hardcode elsewhere
│   ├── calibration/
│   │   ├── surface.py           ← plane fit (SVD), T_cam_to_table, pixel_to_3d, cam_to_table
│   │   └── ui.py                ← interactive 4-corner calibration, 10-frame median depth
│   ├── tracking/
│   │   ├── depth_model.py       ← DAV2 wrapper, infer_frame(), sample_patch() — use patch not pixel
│   │   ├── hand_tracker.py      ← MediaPipe wrapper — mediapipe import in try/except (intentional)
│   │   ├── trajectory.py        ← per-frame 3D extraction, gripper_center, grasp_axis, approach_vec
│   │   └── smoother.py          ← SG smoothing + linear interpolation across gaps
│   ├── ik/
│   │   ├── solver.py            ← single-frame DLS IK, build_grasp_rotation, 4cm threshold line 135
│   │   ├── trajectory_solver.py ← warmstarted IK + velocity clamping + carry-forward on failure
│   │   └── workspace.py         ← analyze_workspace, find_robot_placement (grid search), table_to_robot
│   └── render/
│       ├── mujoco_renderer.py   ← CAMERA_PRESETS, render_frame(), render_trajectory()
│       ├── composite.py         ← 4-panel assembly, back-projection for hand overlay
│       └── writer.py            ← cv2.VideoWriter wrapper (mp4v codec)
├── scripts/                     ← one script per pipeline phase, each runnable standalone
│   ├── calibrate.py
│   ├── record.py
│   ├── extract_trajectory.py
│   ├── smooth_trajectory.py
│   ├── analyze_workspace.py
│   ├── solve_ik.py
│   ├── render_sim.py            ← --smoke-test flag for single-frame PNG output
│   ├── render_composite.py
│   ├── plot_trajectory.py
│   ├── validate_depth.py
│   └── download_metric_checkpoint.py
├── robot/
│   ├── panda.xml                ← MODIFIED: offwidth/offheight in <visual>, table geom, fill light
│   └── assets/                  ← Panda mesh files (.obj, .stl) — do not touch
├── data/                        ← gitignored — calibration.json, *.npz, *_placement.json
├── outputs/                     ← gitignored — rendered .mp4 files
├── checkpoints/                 ← gitignored — DAV2 model weights
├── Depth-Anything-V2/           ← gitignored — external repo, cloned separately
├── .venv/                       ← gitignored
├── README.md                    ← setup guide + component documentation
├── THEORY.md                    ← design rationale, all math explained, future directions
├── LEARNING_ROADMAP.md          ← study checklist for Janeshvar (10 modules)
├── ROBOTICS_PRIMER.md           ← comprehensive educational article on the field
├── LESSONS_FOR_CLAUDE.md        ← this file
├── WHAT_I_DID_WHEN_U_WERE_INACTIVE.md  ← documents Phase 8 completion by user
├── tracking.py                  ← early prototype, kept for reference
└── v1.py                        ← first version prototype, kept for reference
```

---

## 6. Critical Technical Lessons — DO NOT Repeat These Mistakes

### IK: More iterations HURT

**What happened:** User asked to test IK_ITERATIONS=150. Success rate dropped from 91.6% to ~47.4%.

**Why:** With 50 iterations, the solver converges to a solution near the warmstart (previous frame). With 150, it converges further — finds a better absolute solution but one that is further in joint space from the previous frame. This produces larger Δq vectors → more velocity clamping → the clamped position becomes a worse warmstart for the next frame → cascade of failures.

**Rule: IK_ITERATIONS=50 is optimal for trajectory IK with warmstarting. NEVER test higher values again without understanding this interaction.**

### IK: Higher gain (≥0.5) also hurts

**What happened:** IK_GAIN=0.5 achieves lower success than 0.3. IK_GAIN=2.0 causes complete divergence when initial error > ~0.3m.

**Why:** Larger steps per iteration mean larger joint changes → more velocity clamping → degraded warmstart for next frame. Same cascade mechanism as iterations.

**Rule: IK_GAIN=0.3 is the validated optimum. Do not touch it.**

### IK: The 4cm threshold was the biggest lever

The IK success threshold (where we declare "this frame worked") was the single largest improvement:
- 2cm → 77.2% success
- 4cm → 91.6% success
- The remaining 8.4% are genuinely unreachable frames at the workspace boundary — no parameter tuning will fix them.

**Rule: The threshold lives in `src/ik/solver.py:135`, not in config.py. When someone asks where to change it, point to that specific line.**

### MuJoCo: Offscreen framebuffer must be declared in XML

**Error encountered:**
```
ValueError: Image width 1280 > framebuffer width 640
```

**Fix:** `robot/panda.xml` must contain this in the `<visual>` section:
```xml
<visual>
  <global offwidth="1920" offheight="1080"/>
</visual>
```
This is already in the current panda.xml. If panda.xml is ever reset to a default version, this will reappear. Add it back immediately before any other debugging.

### Windows encoding: Avoid non-ASCII in print statements

**Error encountered:** `UnicodeEncodeError: 'charmap' codec can't encode character '→'` when using `→` in a print statement on Windows with cp1252 encoding.

**Rule: On Windows, use `->` not `→` in all print statements in .py files. This applies to all scripts in this project.**

### Depth sampling: Always use 5×5 median patch, never single pixel

`depth_model.sample_patch(depth, px, py, radius=2)` takes a 5×5 median. Using `depth[py, px]` directly will pick up background depth at fingertip silhouette edges. This is not optional — single-pixel sampling produces systematically wrong depth for the hand landmarks.

### Coordinate frames: Never skip a step in the chain

Transform chain is: **camera → table → robot base**

- camera → table: via `T_cam_to_table` (from calibration)
- table → robot base: via `table_to_robot()` which is just subtraction (same orientation, different origin)

Never try to go camera → robot base directly. The intermediate table frame is where smoothing, workspace analysis, and IK input all live. Mixing frames causes silent incorrect results, not errors.

### MediaPipe import: Wrapped in try/except intentionally

`src/tracking/hand_tracker.py` imports MediaPipe inside a `try/except ImportError`. This is intentional — scripts that only load saved trajectory data (render_sim.py, solve_ik.py, etc.) should not require MediaPipe to be installed. Do not "fix" this into a top-level import.

### pipeline.py: subprocess approach, not module imports

`pipeline.py` calls each step via `subprocess.run([sys.executable, script_path, ...])`. This was a deliberate architectural choice — it avoids sys.path conflicts between scripts, forwards stdout/stderr naturally, and isolates the memory of each step's model loading (DAV2 weights released between steps).

Do not refactor to direct function imports without understanding this tradeoff.

---

## 7. Coordinate Frame Reference

| Frame | Origin | X axis | Y axis | Z axis | Used for |
|---|---|---|---|---|---|
| Camera | Camera optical centre | Right | Down | Into scene | MediaPipe pixel → 3D |
| Table | Centroid of 4 corners | TL→TR (width) | TL→BL (depth) | Up (toward cam) | All analysis |
| Robot base | Robot base centre | Same as table | Same as table | Same as table | IK targets |
| MuJoCo world | Robot base | Right | Forward | Up | Simulation |

`T_cam_to_table` is the 4×4 homogeneous transform from camera frame to table frame. It is saved in `data/calibration.json` and loaded by almost every script.

---

## 8. Data File Formats

| File | Contents |
|---|---|
| `data/calibration.json` | `T_cam_to_table` (4×4), intrinsics dict, `corners_cam` (4×3), plane residuals |
| `data/<stem>_raw.npz` | `thumb_tip, index_tip, wrist, gripper_center, gripper_width, grasp_axis, approach_vec, valid` — all in table frame (metres) |
| `data/<stem>_smoothed.npz` | Same structure, SG filtered, gaps interpolated |
| `data/<stem>_placement.json` | `base_pos_table_frame` [3], `coverage`, `n_reachable`, `n_total` |
| `data/<stem>_joints.npz` | `joint_angles` (N×7 rad), `gripper` (N metres), `ik_success` (N bool), `fps`, `n_frames` |

---

## 9. robot/panda.xml — What Was Modified and Why

Three additions were made to the base Franka Panda MuJoCo model:

1. **Offscreen framebuffer** (required for any render > 640px wide):
```xml
<visual>
  <global offwidth="1920" offheight="1080"/>
</visual>
```

2. **Table geometry** (visual reference + contact surface):
```xml
<geom name="table" type="box" pos="0.4 0 -0.04" size="0.7 0.7 0.015"
      rgba="0.8 0.75 0.65 1" contype="1" conaffinity="1"/>
```

3. **Fill light** (reduces harsh shadows on the arm):
```xml
<light name="fill" pos="-1 -1 2" dir="1 1 -1" directional="true"
       diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
```

The gripper is driven by: `data.qpos[7] = data.qpos[8] = gripper_m / 2`, then `mj_forward`. No actuator — forward kinematics only.

---

## 10. Camera Presets (src/render/mujoco_renderer.py)

```python
CAMERA_PRESETS = {
    "side":  {"azimuth": 135.0, "elevation": -25.0, "distance": 2.0, "lookat": [0.4, 0.0, 0.35]},
    "top":   {"azimuth": 0.0,   "elevation": -90.0, "distance": 2.0, "lookat": [0.4, 0.0, 0.0]},
    "front": {"azimuth": 180.0, "elevation": -20.0, "distance": 2.0, "lookat": [0.4, 0.0, 0.35]},
}
```

Uses programmatic `MjvCamera` (not XML-defined cameras) to avoid needing to compute `xyaxes` vectors manually.

---

## 11. IK Architecture — How the Pieces Fit

```
src/ik/solver.py — IKSolver class
  solve(target_pos, target_rot, warmstart_q, n_iter=50) → (q_solved, success_bool)
  Uses MuJoCo model to compute Jacobians at each iteration
  Success = final_pos_err < 4cm

src/ik/trajectory_solver.py — solve_trajectory()
  For each frame:
    1. Convert gripper_center from table frame → robot frame (table_to_robot)
    2. Build target rotation from grasp_axis + approach_vec (build_grasp_rotation)
    3. Call solver.solve() with previous frame's q as warmstart
    4. Velocity-clamp: if |Δq| > max_dq_per_frame, scale down proportionally
    5. Use clamped q (not IK q) as next warmstart — CRITICAL
    6. If IK failed: carry forward previous q (keep trajectory continuous)

src/ik/workspace.py — find_robot_placement()
  Grid search over table XY
  Score each candidate: fraction of waypoints in (0.170m, 0.855m) annulus
  Returns best position + 40×40 heatmap for visualisation
```

**The carry-forward policy:** When IK fails, `joint_angles[i] = joint_angles[i-1]`. This makes the robot appear to freeze. This is intentional. The alternative (interpolation) is listed as a future improvement.

---

## 12. Lessons About the Build Process

### What worked well
- **Building incrementally by phase** — each phase produced a testable output before moving to the next.
- **Separating scripts from src** — scripts are CLI wrappers; src is the importable library. This allowed individual scripts to be run standalone for debugging.
- **Smoke test in render_sim.py** — `--smoke-test` renders frame 0 to PNG without writing a full video. Saved hours of debugging the rendering pipeline.
- **Printing diagnostics at every step** — IK success rates, smoothing displacement, workspace coverage printed to stdout. Always add these.

### What caused problems
- **Reading before editing** — the `tool_use_error: File has not been read yet` error occurred when trying to edit `solver.py` without reading it first. Always Read before Edit.
- **Running the edit tool on a file that wasn't explicitly read in the current conversation** — even if you know what the file contains from context, the tool requires a prior Read call.
- **Large context accumulation** — this project ran for two full context windows. The session summary carried over all the critical technical decisions.

---

## 13. Expert Panel Consensus — Agreed Next Steps

From a synthesized discussion between five domain experts (AI Research, Startup Founder, AI Professor, Simulation Expert, Robot Learning Practitioner), ranked by impact/effort:

**Tier 1 — Do in existing codebase, no new hardware:**

1. **Output EE poses alongside joint angles** — add `(N×7: xyz + quaternion)` to joints.npz. ~20 lines in trajectory_solver.py. Unlocks Cartesian-space policy training.

2. **Filter trajectory segments by IK success** — split output into contiguous high-quality segments (≥95% success). Discard carry-forward-heavy regions. Prevents corrupting supervision signal in ACT/Diffusion Policy training.

3. **Grasp-close event detection** — detect when gripper width drops below ~0.03m (grasp) and rises (release). Add `grasp_start`/`grasp_end` to output. Enables task segmentation and alignment with simulated objects.

4. **Recover wrist roll from MediaPipe** — the current orientation is missing the rotation around the approach vector axis (wrist roll). Compute from the `(wrist, index_MCP, pinky_MCP)` triangle. ~60 lines in solver.py.

5. **Add configurable primitive object to panda.xml** — `<body name="object"><geom type="box"/>` with pose settable at render time. Makes simulation output semantically meaningful for policy training observations.

**Tier 2 — One hardware change:**

6. **Replace monocular depth with RGB-D camera** — Intel RealSense D435 (~$200). Drops temporal noise from 30–130mm to 2–5mm. Removes ±10% scale error. Likely raises IK success above 98%. Everything else in the pipeline stays identical.

**Tier 3 — New capabilities:**

7. **SAM2 object tracking** — zero-shot video segmentation on the manipulated object, combine 2D centroid with metric depth for 3D position. Converts output from "arm in free space" to "arm relative to object."

8. **IK atlas initialization** — precompute known-good IK solutions at grid of workspace positions. Initialize from nearest atlas entry at trajectory start and after failures.

9. **Benchmark against teleoperated baseline** — collect 20 H2R demos + 20 SpaceMouse demos on the same task, train ACT with identical hyperparameters, report real-robot success rate. **This is the critical validation experiment.**

10. **Bimanual extension** — `max_num_hands=2`, dual-arm panda.xml, dual-arm IK. High research impact, 2–4 week effort.

---

## 14. User Context

- **Name:** Janeshvar
- **Background:** New to robotics, completed this as first project, Windows 11, CUDA GPU
- **Goal:** Pursue robotics research, needs theoretical depth to complement practical experience
- **Learning path:** Currently working through ROBOTICS_PRIMER.md and LEARNING_ROADMAP.md
- **Most important near-term goal:** Policy training validation — train ACT/Diffusion Policy on H2R-generated data and test on a real robot
- **Prefers planning-first:** Explain approach before executing, especially for multi-step tasks
- **Communication style:** Direct, informal, occasionally steps away and returns — always document what was done

---

## 15. What Files to Read Before Starting Any Task

| If working on... | Read first |
|---|---|
| IK solver changes | `src/ik/solver.py`, `src/ik/trajectory_solver.py`, this file §6 |
| Rendering changes | `src/render/mujoco_renderer.py`, `robot/panda.xml`, this file §9-10 |
| Pipeline changes | `pipeline.py`, `scripts/` directory |
| Calibration changes | `src/calibration/surface.py`, `src/calibration/ui.py` |
| Depth/tracking changes | `src/tracking/depth_model.py`, `src/tracking/trajectory.py` |
| Any config changes | `src/config.py`, this file §4 |
| Workspace/placement | `src/ik/workspace.py` |
| New feature additions | This file §13 (next steps list) first — don't duplicate planned work |

---

*Last updated: 2026-05-25. Covers sessions from project inception through Phase 9 completion, README/THEORY documentation, expert panel discussion, and educational materials creation.*
