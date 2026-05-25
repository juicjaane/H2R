# Robot Learning Research — Learning Roadmap

A structured checklist for going from "completed one project" to "research-ready" in robot manipulation learning. Topics are grouped by subject area and ordered within each section from foundational to advanced. Work through them in order — each section builds on the previous.

For the full conceptual explanation of each topic, read **ROBOTICS_PRIMER.md** alongside this checklist.

---

## How to use this list

- Check off items as you reach working understanding (not just "read about it")
- "Working understanding" = you could explain it to someone else and answer follow-up questions
- Starred items ⭐ are the highest-leverage topics — if you only have limited time, do these first
- Each section ends with a **"project to cement understanding"** suggestion

---

## Module 1 — Mathematics Foundations

### Linear Algebra
- [ ] Vectors: addition, dot product, cross product, norm
- [ ] Matrices: multiplication, transpose, inverse
- [ ] ⭐ Rotation matrices: what they represent, properties (orthogonal, det=1)
- [ ] Eigenvalues and eigenvectors (conceptual)
- [ ] ⭐ Singular Value Decomposition (SVD): what it computes, why it's used for least-squares
- [ ] Null space and rank of a matrix

### Calculus & Optimization
- [ ] Partial derivatives and gradients
- [ ] Chain rule
- [ ] ⭐ Least-squares problems: what they are, why the solution is `(AᵀA)⁻¹Aᵀb`
- [ ] Gradient descent (conceptual)
- [ ] Lagrange multipliers (conceptual — for constrained optimization)

### Probability & Statistics
- [ ] Probability distributions: Gaussian, uniform
- [ ] Mean, variance, standard deviation
- [ ] Conditional probability and Bayes' theorem
- [ ] Maximum likelihood estimation (conceptual)
- [ ] ⭐ Kalman filter (conceptual — state estimation with noisy observations)

**Project:** Implement SVD-based plane fitting from scratch in numpy. Given 10 random 3D points near a plane with noise, recover the plane normal. This is exactly what `src/calibration/surface.py` does.

---

## Module 2 — Rigid Body Geometry

### Coordinate Frames and Transforms
- [ ] ⭐ What a coordinate frame is (origin + three orthogonal axes)
- [ ] ⭐ Rotation matrices (3×3): how to build one from axis vectors
- [ ] ⭐ Homogeneous transforms (4×4): combining rotation + translation into one matrix
- [ ] Composing transforms: `T_A_to_C = T_B_to_C @ T_A_to_B`
- [ ] Inverting a transform: `T_inv = [Rᵀ, -Rᵀt; 0 0 0 1]`
- [ ] Active vs passive rotation (transforming a point vs rotating a frame)

### Rotation Representations
- [ ] ⭐ Rotation matrices: advantages (composable), disadvantages (9 numbers for 3 DOF)
- [ ] ⭐ Euler angles (roll, pitch, yaw): intuitive but have gimbal lock
- [ ] ⭐ Axis-angle representation: `(axis, θ)` — compact, singularity-free for small angles
- [ ] Quaternions: 4-number representation, no gimbal lock, efficient composition
- [ ] Converting between representations
- [ ] ⭐ The rotation error used in IK: `ω_err = skew_symmetric_part(R_target @ R_current.T)`

### Camera Geometry
- [ ] ⭐ Pinhole camera model: how a 3D point maps to a 2D pixel
- [ ] ⭐ Camera intrinsics: focal length (fx, fy), principal point (cx, cy)
- [ ] ⭐ Back-projection: from pixel + depth to 3D point
- [ ] Camera extrinsics: pose of the camera in the world
- [ ] Lens distortion (radial/tangential) — what it is, when it matters
- [ ] Stereo camera geometry: epipolar lines, triangulation

**Project:** Write a function that takes a 3D point in camera frame and returns its pixel coordinate (projection), then write the inverse (back-projection). Verify they are inverses of each other.

---

## Module 3 — Robot Kinematics

### Forward Kinematics
- [ ] ⭐ What forward kinematics (FK) is: given joint angles → where is the end-effector?
- [ ] Denavit-Hartenberg (DH) parameters: the standard way to describe a serial robot
- [ ] Kinematic chain: how transforms compose along the arm
- [ ] ⭐ The Jacobian matrix: how joint velocities map to end-effector velocity
- [ ] ⭐ Geometric vs analytic Jacobian
- [ ] How MuJoCo computes FK and the Jacobian (`mj_forward`, `mj_jacBody`)

### Inverse Kinematics
- [ ] ⭐ What inverse kinematics (IK) is: given desired EE pose → find joint angles
- [ ] Why IK is hard: redundancy (7-DOF arm, 6-DOF task), joint limits, singularities
- [ ] ⭐ Jacobian pseudoinverse: `Δq = J⁺ · Δx = Jᵀ(JJᵀ)⁻¹ · Δx`
- [ ] ⭐ Damped Least-Squares (DLS): `Δq = Jᵀ(JJᵀ + λI)⁻¹ · Δx`
- [ ] What a singularity is: when `JJᵀ` becomes rank-deficient, why DLS helps
- [ ] ⭐ Warmstarting: why initializing from the previous frame's solution is critical
- [ ] Redundancy resolution: null-space motion (moving without changing EE pose)
- [ ] Analytic IK vs numeric IK: when each is appropriate
- [ ] Joint limit handling in iterative IK

### Workspace Analysis
- [ ] ⭐ Reachable workspace vs dexterous workspace
- [ ] Joint limits and how they shrink the reachable workspace
- [ ] Kinematic singularities and why they create workspace boundaries
- [ ] ⭐ Franka Panda specs: 7 DOF, reach 855mm, max velocity 2.175 rad/s per joint
- [ ] Base placement optimization for a given task trajectory

**Project:** Build a simple 2-DOF planar robot (two links, two joints). Implement FK and the Jacobian analytically. Then implement iterative DLS IK and verify it converges. Visualize the workspace.

---

## Module 4 — Robot Hardware and Control

### Actuators and Sensing
- [ ] Joint torque control vs position control vs velocity control
- [ ] Proprioception: joint encoders, resolvers
- [ ] Force/torque sensing at the wrist: what it measures, why it matters for contact tasks
- [ ] The Franka Panda: 7-DOF research arm, torque-controlled, 3kg payload
- [ ] Gripper mechanics: parallel jaw, suction cup, dexterous hands

### Low-Level Control
- [ ] ⭐ PID control: proportional, integral, derivative — what each term does
- [ ] Joint space control vs Cartesian space (task space) control
- [ ] Impedance control: compliance — how "stiff" or "soft" the robot behaves
- [ ] ⭐ Cartesian impedance control: control in EE space with virtual spring-damper
- [ ] Joint velocity limits and why they exist (hardware safety)
- [ ] Real-time control requirements: why robot control loops run at 1kHz

### Safety and Hardware Limits
- [ ] Joint limits (position): range of motion for each joint
- [ ] Joint velocity limits: max rad/s per joint
- [ ] Joint torque limits: max Nm per joint
- [ ] ⭐ Why violating velocity limits matters: hardware protection, and trajectory continuity
- [ ] Emergency stop mechanisms and collision detection

**Project:** Read the Franka Panda technical documentation. List all 7 joint position limits, velocity limits, and torque limits. Understand why joint 1 and joint 7 have very different limits.

---

## Module 5 — Perception for Manipulation

### Depth Sensing
- [ ] ⭐ Monocular depth estimation: what it can and can't do
- [ ] ⭐ Structured light depth sensors (Intel RealSense): how they work, accuracy, range
- [ ] Time-of-flight sensors: how they differ from structured light
- [ ] Stereo cameras: how triangulation gives depth
- [ ] ⭐ Metric vs relative depth: why the distinction matters for robotics
- [ ] Depth Anything V2: architecture overview (ViT encoder + DPT decoder)
- [ ] Depth noise characteristics: random vs systematic error

### Object Detection and Tracking
- [ ] ⭐ YOLO: real-time object detection, bounding boxes
- [ ] SAM (Segment Anything Model): zero-shot segmentation
- [ ] SAM2: video segmentation, tracking across frames
- [ ] 6DOF object pose estimation: what it means, why it's hard
- [ ] FoundationPose: model-based 6DOF pose tracking
- [ ] ⭐ Feature matching for tracking: SIFT, ORB, DINO features

### Hand Tracking
- [ ] ⭐ MediaPipe Hands: how it works, 21 landmark model, limitations
- [ ] Hand pose estimation vs hand tracking vs gesture recognition
- [ ] FrankMocap, HaMeR: full 6DOF wrist pose estimation from RGB
- [ ] Hand-object interaction detection: when is the hand grasping something?
- [ ] 2D-to-3D lifting: from pixel landmarks to metric 3D positions

**Project:** Run SAM2 on a short video of your hand picking up an object. Segment the object in frame 0, let SAM2 track it, and plot its 2D centroid trajectory over time.

---

## Module 6 — Imitation Learning

### Foundations
- [ ] ⭐ What imitation learning is: learning a policy from demonstrations
- [ ] ⭐ Behavioral Cloning (BC): supervised learning on (observation, action) pairs
- [ ] The covariate shift problem in BC: why BC fails on long-horizon tasks
- [ ] DAgger (Dataset Aggregation): interactive data collection to fix covariate shift
- [ ] ⭐ Offline vs online imitation learning

### Modern Policy Architectures
- [ ] ⭐ ACT (Action Chunking with Transformers, Zhao et al. 2023):
  - What "action chunking" is and why it helps
  - The CVAE architecture for multimodal action prediction
  - Temporal ensemble at inference
- [ ] ⭐ Diffusion Policy (Chi et al. 2023):
  - How diffusion models work (denoising from noise to action)
  - Why diffusion handles multimodal action distributions better than MSE
  - CNN-based vs Transformer-based observation encoding
- [ ] GATO (Reed et al. 2022): a single generalist policy for many tasks
- [ ] RT-1, RT-2 (Google DeepMind): scaling robot transformers
- [ ] ⭐ What observation and action spaces look like in practice:
  - Observations: RGB images, depth, proprioception (joint angles + velocities)
  - Actions: joint angles, EE delta pose, absolute EE pose

### Data and Evaluation
- [ ] ⭐ What a demonstration dataset looks like: episodes, frames, actions
- [ ] Dataset size requirements: BC typically needs 20–200 demos per task
- [ ] ⭐ Train/val split considerations for imitation learning
- [ ] Evaluation metrics: task success rate, trajectory smoothness
- [ ] BridgeData V2: large-scale real robot dataset across 24 environments
- [ ] ALOHA / ACT dataset: bimanual manipulation with 50 demos per task

**Project:** Read the ACT paper (https://arxiv.org/abs/2304.13705) end-to-end. Understand the CVAE architecture diagram. Be able to answer: why does action chunking reduce compounding errors?

---

## Module 7 — Robot Simulation

### MuJoCo
- [ ] ⭐ What MuJoCo is: physics engine for contact-rich simulation
- [ ] ⭐ MJCF (XML) format: how to define bodies, joints, geoms, actuators
- [ ] Forward kinematics in MuJoCo: `mj_forward` vs `mj_step`
- [ ] ⭐ Headless rendering: `mujoco.Renderer` for offscreen image generation
- [ ] Contact dynamics in MuJoCo: how contact forces are computed
- [ ] Actuator models: position servo, velocity servo, torque actuator
- [ ] Sensor simulation: touch sensors, force/torque, cameras

### Other Simulators
- [ ] Isaac Gym / Isaac Lab (NVIDIA): GPU-parallel simulation for RL
- [ ] PyBullet: open-source, good for learning
- [ ] Genesis: new 2024 simulator, very fast, differentiable
- [ ] Gazebo: ROS-integrated, used in field robotics
- [ ] ⭐ Sim-to-real transfer: why policies trained in sim often fail on real robots
- [ ] Domain randomization: randomly varying sim parameters to improve transfer
- [ ] Visual domain gap: why MuJoCo renders look different from real cameras

### Synthetic Data
- [ ] What synthetic data is and why robotics needs it
- [ ] BlenderProc, NVISII: photorealistic rendering for training data
- [ ] ⭐ The tradeoff: synthetic data is infinite but has domain gap; real data is scarce but accurate
- [ ] sim-to-sim transfer as a stepping stone before sim-to-real

**Project:** Open `robot/panda.xml` in this project. Add a second primitive object (a box) to the worldbody. Set its initial position and verify it renders correctly with `scripts/render_sim.py --smoke-test`.

---

## Module 8 — Data Collection Methods

- [ ] ⭐ Kinesthetic teaching: physically guiding the robot by hand, recording joint angles
- [ ] ⭐ Teleoperation with leader-follower: one robot (leader) controlled by human, follower mimics
- [ ] SpaceMouse / joystick teleoperation: 6DOF input device controls EE
- [ ] ⭐ ALOHA (Zhao et al. 2023): low-cost bimanual teleoperation with 3D-printed hardware
- [ ] GELLO: another low-cost teleoperation device using servo motors
- [ ] UMI (Chi et al. 2024): wrist-mounted GoPro, no robot needed, EE trajectory extraction
- [ ] ⭐ Human video retargeting (this project): monocular webcam, IK retargeting
- [ ] Motion capture for data collection: Vicon, OptiTrack
- [ ] ⭐ Tradeoffs: cost vs data quality vs scalability vs naturalness

**Project:** Watch 3 YouTube videos of ALOHA in action. Then watch a UMI demo. What's the key difference in the user experience of collecting one demonstration?

---

## Module 9 — The Research Landscape

### Key Papers to Read (in order)
- [ ] ⭐ **BC baseline**: "Learning from Demonstrations for Real World RL" (Nair et al. 2018) — why BC fails
- [ ] ⭐ **ACT**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al. 2023)
- [ ] ⭐ **Diffusion Policy**: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (Chi et al. 2023)
- [ ] **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale" (Brohan et al. 2022)
- [ ] **RT-2**: "RT-2: Vision-Language-Action Models" (Brohan et al. 2023) — using LLMs for robot policies
- [ ] **UMI**: "Universal Manipulation Interface" (Chi et al. 2024)
- [ ] **RoboAgent**: "RoboAgent: Generalisation and Efficiency in Robot Manipulation" (Bharadhwaj et al. 2023)
- [ ] ⭐ **VoxPoser**: "VoxPoser: Composable 3D Value Maps for Robot Manipulation with LLMs" (Huang et al. 2023)
- [ ] **GROOT**: "GROOT: Learning Generalizable Manipulation Policies with Object-Centric 3D Representations" (Zhu et al. 2023)
- [ ] **π₀ (pi-zero)**: "π₀: A Vision-Language-Action Flow Model for General Robot Control" (Black et al. 2024)

### Key Labs and Groups
- [ ] Stanford IPRL (Chelsea Finn, Dorsa Sadigh) — meta-learning, imitation learning
- [ ] Berkeley RAIL (Sergey Levine) — offline RL, robot learning at scale
- [ ] CMU Robotics (Deepak Pathak, David Held) — representation learning, manipulation
- [ ] MIT CSAIL (Pulkit Agrawal, Russ Tedrake) — trajectory optimization, contact-rich
- [ ] Google DeepMind Robotics — RT-1/2, mobile manipulation, foundation models
- [ ] Physical Intelligence (pi) — pi-zero, general-purpose robot policies
- [ ] Stanford IRIS (Jeannette Bohg) — perception for manipulation, grasping

### Key Conferences and Venues
- [ ] ⭐ RSS (Robotics: Science and Systems) — top robotics theory venue
- [ ] ⭐ CoRL (Conference on Robot Learning) — top learning + robotics venue
- [ ] ICRA (International Conference on Robotics and Automation) — largest robotics conference
- [ ] ICLR, NeurIPS, ICML — where the ML theory underlying robot learning publishes
- [ ] RA-L (Robotics and Automation Letters) — journal with rapid publication

**Project:** Go to paperswithcode.com/task/robot-manipulation and find the top-5 methods on any manipulation benchmark. For each: what observation space? what action space? what architecture?

---

## Module 10 — Research Skills

- [ ] How to read a research paper efficiently (abstract → intro → figures → conclusion → method)
- [ ] How to implement a paper from scratch (start with the simplest version)
- [ ] Writing a research proposal: problem statement, related work, approach, evaluation
- [ ] ⭐ Experimental design: what makes a robotics experiment convincing?
  - Ablation studies: remove one component at a time
  - Baselines: what is the "naive" approach you're comparing against?
  - Sample size: how many trials to establish statistical significance?
- [ ] Git for research: reproducibility, tagging experiment versions
- [ ] Weights & Biases or MLflow: tracking experiments
- [ ] How to set up a real robot lab: safety, workspace design, camera mounting

---

## Progression Summary

| Stage | Focus | When you're ready for the next stage |
|---|---|---|
| **Beginner** | Modules 1–3 | Can implement FK, IK, and camera projection from scratch |
| **Intermediate** | Modules 4–6 | Can train a BC policy on a provided dataset; understand ACT and Diffusion Policy |
| **Advanced** | Modules 7–9 | Can design and run a robot learning experiment end-to-end |
| **Research-ready** | Module 10 | Can identify a gap in the literature and propose a contribution |

**Your current position:** You've built a working end-to-end pipeline (Modules 2, 3, 5, 7 applied). You have practical intuition that most beginners lack. Fill in the theory gaps in Modules 1–6, then move directly to Module 9 (reading key papers).
