# H2R — Theory & Design Rationale

> *A simple experiment to generate robot imitation learning data by bypassing teleoperation.*

---

## The Core Problem

Training robot manipulation policies requires large datasets of robot demonstrations. The dominant collection method today is **teleoperation**: a human operator controls the physical robot in real time, and the system records the robot's joint angles while the human performs the task. This is effective but expensive:

- You need the physical robot available.
- You need a trained operator for each data collection session.
- Scaling to thousands of demonstrations requires proportional operator-hours.
- Recording in new environments requires bringing the robot there.

This project asks: **what if you could record a human doing the task naturally, then convert that into a valid robot trajectory automatically?**

The output — joint angles + gripper state for a Franka Panda — is a drop-in replacement for teleoperated demonstrations in standard imitation learning frameworks (ACT, Diffusion Policy, etc.).

---

## The Signal Chain

```
Monocular RGB video
  │
  ▼
2D hand landmarks (MediaPipe Hands)
  │  pixel coordinates, 21 landmarks
  ▼
Metric depth per landmark (Depth Anything V2 — indoor Hypersim)
  │  metres, not relative disparity
  ▼
3D points in camera frame (pinhole back-projection)
  │  (X, Y, Z) = ((px-cx)/fx, (py-cy)/fy, 1) * depth
  ▼
3D trajectory in table frame (rigid body transform T_cam→table)
  │  calibrated from 4 corner touches before recording
  ▼
Savitzky-Golay smoothing (offline, zero-lag)
  │  window=15, polynomial order=3
  ▼
Robot base placement (grid search, workspace coverage maximisation)
  │  40×40 grid over table XY, annular reachability check
  ▼
IK solution per frame (damped least-squares, warmstarted)
  │  Δq = Jᵀ(JJᵀ + λI)⁻¹ · e, warmstart from prev frame
  ▼
Joint trajectory (N×7 rad) + gripper (N metres)
  │  ← same format as a teleoperated demonstration
  ▼
MuJoCo simulation + composite video
```

Every step has a deliberate design choice. The sections below explain the reasoning.

---

## Why Monocular RGB, Not RGBD or Multi-Camera?

The explicit goal is **low barrier to entry**. A single consumer webcam should be sufficient. Depth cameras require additional hardware, precise mounting, and IR interference becomes a problem in uncontrolled lighting. The tradeoff: we must estimate metric depth from appearance alone.

The key insight is that Depth Anything V2 with the Hypersim indoor metric checkpoint predicts **absolute depth in metres**, not disparity. This removes the scale ambiguity problem that plagued earlier monocular approaches. The accuracy is ~10% at 0.5–2m, which is sufficient for robot trajectory data if followed by smoothing.

Calibration (four corner touches) provides the ground-truth coordinate frame without any external sensors.

---

## Why Table-Surface Calibration?

The hand trajectory needs to be expressed in a frame that is stable relative to the robot. Options:

1. **Camera frame** — unstable if the camera moves; also not robot-aligned.
2. **World frame** — requires a map or fiducial markers.
3. **Table frame** — natural for tabletop manipulation; defined by the surface the task occurs on.

The table frame is established with a one-time calibration: the user touches each of the four corners of the recording surface with their index fingertip. This gives four 3D points in camera frame (via back-projection through the depth map). A plane is fit to these points via SVD, and the resulting normal + corner positions define a rigid transform `T_cam_to_table`.

The key property: once calibrated, any video recorded from the same camera-table geometry produces trajectories in the same robot-relative coordinate system.

---

## Why Savitzky-Golay, Not EMA or Kalman?

Depth estimation is **temporally independent** — the model processes each frame in isolation with no memory of past frames. This produces depth noise with a standard deviation of 30–130mm at a fixed point. The noise is not correlated across frames (it is closer to white noise than a drift process).

Three filtering options:

| Filter | Lag | Handles jump noise | Works offline |
|---|---|---|---|
| EMA / one-pole IIR | Yes — proportional to window | No | Yes |
| Kalman (constant-velocity) | Minimal (prediction step) | Somewhat | Yes |
| Savitzky-Golay | **Zero (uses future frames)** | Yes | Yes (batch only) |

Since the data collection is entirely offline (we have the full video before processing), zero-lag filtering is possible. Savitzky-Golay fits a local polynomial to a window of frames — the current filtered value uses both past and future observations, so there is no causal lag. This is optimal for offline trajectory generation and produces visibly smoother output than EMA at the same effective bandwidth.

---

## The Depth-to-3D Back-Projection

The pinhole camera model gives:

```
X = (px - cx) * depth / fx
Y = (py - cy) * depth / fy
Z = depth
```

Where `(fx, fy, cx, cy)` are the camera intrinsics (focal lengths and principal point). For the Logitech Brio 100, the intrinsics were computed from the known diagonal FOV (58°) and resolution:

```
f = sqrt(W² + H²) / (2 * tan(29°))
cx = W/2,  cy = H/2
```

A critical detail: we do **not** sample depth at the exact pixel `depth[py, px]`. Instead we use a 5×5 median patch centred on the landmark. This is because MediaPipe places landmark positions at pixel boundaries of fingers and joints. At silhouette boundaries, the depth map often contains background depth mixed in (the finger is thin — some pixels see the finger, some see the table behind it). The median of a 5×5 neighbourhood is robust to this contamination.

---

## The IK Formulation

### Damped Least-Squares

The Jacobian J maps joint velocities to end-effector velocities:
```
ẋ = J q̇
```

Inverting this to get joint updates from a desired end-effector move:
```
Δq = Jᵀ(JJᵀ + λI)⁻¹ · e
```

The term `+ λI` is the damping factor. Without it, near kinematic singularities `JJᵀ` becomes ill-conditioned and the joint updates blow up. λ=1e-4 provides stability without significantly slowing convergence away from singularities.

This is applied iteratively. Each iteration:
1. Compute current EE position and rotation via forward kinematics (`mj_forward`)
2. Compute 6D error: `[pos_target − pos_current, rot_error]`
3. Compute Jacobian at current configuration
4. Apply DLS update
5. Clip to joint limits

### Warmstarting Across Frames

The most important design decision for trajectory IK: each frame initialises from the **previous frame's solution**, not from the home pose.

Without warmstarting, the solver has no information about the trajectory direction and will find the nearest configuration regardless of whether it is kinematically continuous with adjacent frames. The result is discontinuous joint trajectories — joints teleport between solutions at each frame, which is physically invalid and useless as training data.

With warmstarting, the solver is essentially performing numerical integration of the joint trajectory: each frame starts from a point that is already near-optimal for the previous frame, so small changes in EE position require only small changes in joint angles. This produces kinematically smooth trajectories.

### Joint Velocity Clamping

After solving each frame, the joint change `Δq = q_new − q_prev` is inspected per joint. If any joint would exceed the Panda's speed limit (2.175 rad/s, from the hardware spec), the entire `Δq` vector is scaled down so the fastest joint moves at exactly the limit. This preserves the relative joint motion profile while respecting hardware constraints.

The clamped position becomes the warmstart for the next frame. This is deliberate: using the unclamped position as the warmstart would misrepresent where the robot actually is.

### The Iterations-Hurt Counterintuition

Running 150 IK iterations instead of 50 **reduces** the success rate (from 91% to ~47% in experiments). The mechanism:

1. With 150 iterations, the solver converges further from the previous frame's configuration.
2. This produces larger `Δq` vectors, which get clamped more aggressively.
3. The clamped position is further from the true IK solution.
4. This is a worse warmstart for the next frame, which then also requires large corrections.
5. The cascade propagates forward through the trajectory.

The insight: for trajectory IK, **convergence per frame is less important than closeness to the previous frame's solution**. 50 iterations finds a good-enough solution that is near the warmstart; 150 iterations finds a better absolute solution but one that is further away in joint space.

---

## Robot Placement Optimisation

The Panda's reachable workspace is modelled as an annulus (spherical shell) with:
- Outer radius: 0.855m (maximum arm extension)
- Inner dead zone: 0.170m (too close, singular configurations)

We grid-search 1600 candidate base positions (40×40 grid centred on the trajectory centroid) and score each as the fraction of trajectory waypoints inside the annulus. The optimal base position typically achieves >95% coverage on tabletop trajectories.

**Limitation:** this is a sphere approximation. The actual Panda workspace is not spherical — it has orientation-dependent holes due to joint limits. A waypoint inside the sphere can still fail IK if the required end-effector orientation (palm down, fingers forward) conflicts with joint limits. This is why the achieved success rate (91.6%) is slightly below the geometric coverage score.

---

## Orientation Representation

The desired end-effector orientation is derived from the hand geometry:
- **X-axis of EE** ← `thumb_tip → index_tip` direction (the "grasp axis")
- **Z-axis of EE** ← palm normal = `cross(index_MCP − wrist, pinky_MCP − wrist)` (the "approach vector")
- **Y-axis** = `Z × X`, then X re-orthogonalised

This maps naturally to a typical overhead grasp: Z points toward the object (down), X spans the gripper opening direction. The resulting orientation matrix is passed directly to the IK solver as the target rotation.

Orientation error is computed as:
```
R_err = R_target · R_current^T
ω_err = 0.5 * [R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]]
```

This is the skew-symmetric part of `R_err`, which extracts the 3D axis-angle residual. It is zero when `R_target = R_current` and grows proportionally for small angular errors.

---

## What This Is (and Is Not)

**What it is:**
- A method to generate kinematically valid robot trajectory data from a human demonstration video.
- A complete offline pipeline: calibrate once, record many tasks, process in batch.
- Output is compatible with standard imitation learning frameworks as a teleoperation substitute.

**What it is not:**
- A replacement for high-precision teleoperation for safety-critical tasks.
- An online system — all processing is batch (video must be fully recorded before processing).
- Object-aware — there is no detection of objects being manipulated. Only the hand trajectory is captured.
- Bimanual — single hand only.
- Orientation-accurate beyond ~15°. The palm normal is an approximation; real grasps have wrist roll that is not modelled.

---

## Relationship to Existing Work

| Approach | Hardware needed | Real-time | Scales without robot |
|---|---|---|---|
| Teleoperation (standard) | Robot + controller | Yes | No |
| Human video → retargeting (this project) | Webcam | No | **Yes** |
| Motion capture | MoCap suit + markers | Yes | No |
| RGB-D hand tracking (e.g., DexPilot) | Depth camera | Yes | No |
| Synthetic data generation | None | N/A | Yes |

This project occupies a niche: **low-cost, camera-only, offline** retargeting. The tradeoff versus synthetic data is naturalness (demonstrations come from real human intent, not scripted simulations); the tradeoff versus teleoperation is accuracy (8.4% frame failures) and the absence of physical contact sensing.

---

## Future Directions

The current pipeline produces joint trajectories for imitation learning but has several open problems:

1. **Object detection integration** — Adding YOLO or SAM to track the object being manipulated would enable full scene-state encoding (object pose + gripper state), which is required for policy generalisation beyond end-effector mirroring.

2. **Orientation accuracy** — The palm normal is a noisy proxy for wrist orientation. A hand pose estimator (e.g., FrankMocap, HaMeR) would give full 6DOF wrist pose and improve grasp quality.

3. **Multi-embodiment retargeting** — The same pipeline could target different robot morphologies (bimanual arms, mobile manipulators) by swapping the URDF and IK solver. The table-frame representation is robot-agnostic.

4. **Online processing** — Replacing DAV2 with a real-time depth estimator (e.g., Depth Pro, MiDaS v3) and implementing EKF-based tracking would enable live retargeting with sub-second latency.

5. **Policy training closure** — The pipeline currently ends at data generation. Integrating a standard IL framework (ACT, Diffusion Policy) and training on collected demonstrations would close the loop from "human does task on video" to "robot can now do the task."

6. **Failure frame interpolation** — The 8.4% IK-failure frames currently carry forward the previous joint configuration. A better approach is trajectory interpolation through the failure region, treating it as a constrained smoothing problem.
