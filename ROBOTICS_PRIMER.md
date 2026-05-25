# Robot Learning Research — A Complete Primer

*Written for someone who has just completed their first robotics project and wants to understand the field deeply enough to do original research.*

---

## Part 1: The Big Picture

### What Is Robot Learning?

The field of robot learning sits at the intersection of robotics (physical systems that act in the world) and machine learning (systems that improve from data). The central question is deceptively simple: **how do you get a robot to learn to do useful things?**

For most of computing history, robots were programmed explicitly. An engineer would write code specifying: "if the sensor reads X, move joint 2 by Y degrees." This works well in controlled environments — car assembly lines, circuit board soldering, warehouse sorting — where the world is predictable and every scenario can be anticipated. It fails catastrophically the moment something unexpected happens: a slightly different object shape, a change in lighting, an obstacle the programmer didn't think of.

Robot learning takes a different approach: instead of programming the behavior, you provide examples of the desired behavior and let the robot learn the underlying policy itself. The robot observes sensor data (cameras, joint angles, force sensors) and learns to map observations to actions.

### Three Paradigms

There are three main ways to learn robot behavior, and understanding the differences is fundamental:

**1. Reinforcement Learning (RL)**
The robot interacts with the environment, receives a reward signal when it does something good (or a penalty when it does something bad), and adjusts its behavior to maximize cumulative reward. Think of training a dog — no explicit instructions, just rewards for good behavior.

*Pros:* Can discover strategies better than any human demonstrator; can improve beyond human performance.
*Cons:* Requires defining a reward function (hard for dexterous manipulation); needs enormous amounts of environment interaction (millions of attempts); unsafe for real hardware during training; difficult to train on real robots.

**2. Imitation Learning (IL)**
The robot learns from human demonstrations — someone shows it what to do, and the robot learns to replicate the behavior. This is the paradigm this project contributes to.

*Pros:* Can learn from a small number of good demonstrations; no reward function needed; natural for humans to provide.
*Cons:* Limited to human-level performance; prone to compounding errors (the robot drifts off the demonstrated trajectory and has never seen recovery); requires data collection infrastructure.

**3. Self-Supervised / Foundation Model Approaches**
Train on massive amounts of internet data (video, text, images) to learn general representations, then fine-tune on robot data. RT-2 (Google DeepMind) takes a vision-language model trained on internet data and fine-tunes it to output robot actions. π₀ (Physical Intelligence, 2024) uses flow matching on a vision-language backbone to produce general manipulation policies.

*Pros:* Leverages the scale of internet data; generalizes to new tasks with few or zero robot demonstrations.
*Cons:* Still requires some robot data; complex infrastructure; inference can be slow for real-time control.

### Why Manipulation Is Hard

Not all robot learning is equally difficult. A robot that walks on flat ground has a well-defined, physically regular task. Robotic manipulation — picking up objects, assembling parts, pouring liquids — is a different category of difficulty:

**High-dimensional action space.** A 7-DOF arm has 7 joint angles that can each move continuously. A bimanual system has 14. The space of possible arm configurations is enormous.

**Contact discontinuities.** When a robot finger touches an object, the physics changes discontinuously. Forces that were zero become non-zero; constraints that didn't exist appear. Contact dynamics are hard to simulate accurately and hard to reason about mathematically.

**Partial observability.** A camera gives you a 2D projection of a 3D world. You can't directly see the forces at contact points. Objects occlude each other. Lighting changes how everything looks.

**Long horizons with compounding errors.** A pick-and-place task might take 200 steps. An error at step 50 (slightly wrong grasp angle) propagates forward and causes total failure at step 150 (object drops during placement). Every error compounds.

**Object diversity.** Real-world objects have diverse shapes, weights, surface materials, and fragility. A policy trained on one mug must generalize to a different mug, then a bowl, then a bottle.

---

## Part 2: Mathematics You Actually Need

### Rotation Matrices

A rotation matrix is a 3×3 matrix R that rotates vectors. It has two key properties:
- **Orthogonal:** `R @ Rᵀ = I` (columns are mutually perpendicular unit vectors)
- **Determinant = +1:** (distinguishes rotation from reflection)

The simplest way to build a rotation matrix is from three orthogonal unit vectors: if you know the X, Y, and Z directions of a new frame expressed in the current frame, stack them as columns:

```
R = [x̂ | ŷ | ẑ]   (each column is a 3D unit vector)
```

To rotate a point `p` into the new frame: `p_new = R @ p`. To rotate back: `p_old = Rᵀ @ p_new`.

**Why this matters for robotics:** Every time you define a coordinate frame (the camera frame, the table frame, the robot base frame), you are implicitly defining a rotation matrix that transforms between it and some other frame. The calibration step in this project builds `T_cam_to_table` — a matrix that rotates and translates points from camera coordinates to table coordinates.

### Homogeneous Transforms

A rotation alone can't represent a coordinate frame change that also involves a translation (moving the origin). The solution is to use 4×4 homogeneous transform matrices:

```
T = [ R  | t ]   where R is 3×3 rotation, t is 3×1 translation
    [ 0  | 1 ]
```

To apply a transform to a 3D point `p`, append a 1 to make it 4D, multiply:
```
[p_new; 1] = T @ [p; 1]
```

To compose two transforms (apply A first, then B): `T_result = T_B @ T_A`.

To invert a rigid body transform (if T goes from A to B, the inverse goes from B to A):
```
T_inv = [ Rᵀ | -Rᵀt ]
        [ 0  |  1   ]
```

Note that the inverse is NOT simply `numpy.linalg.inv(T)` — that works numerically but the closed-form above is faster and numerically more stable.

**Why this matters:** Every coordinate frame conversion in robotics — camera to world, world to robot base, robot base to end-effector — is a homogeneous transform. The transform chain `camera → table → robot base` in this project is three such matrices composed together.

### The Jacobian

The Jacobian is the most important mathematical object in robot kinematics. It answers the question: if I move the joints by a small amount, how much does the end-effector move?

Formally, if `q` is the vector of joint angles (7 numbers for the Panda) and `x` is the end-effector pose (6 numbers: 3 position + 3 orientation), then:

```
dx = J(q) · dq
```

J is a 6×7 matrix (for the Panda), and it depends on the current joint configuration `q`. The first 3 rows relate joint velocities to EE linear velocity; the last 3 rows relate to EE angular velocity.

Each column of J is the "instantaneous effect of joint i on the end-effector" — a 6D vector describing how the EE moves when only joint i rotates.

**Why it depends on q:** The Jacobian changes as the robot moves. Think about it physically: when your arm is fully extended, moving your shoulder has a large effect on your hand position; when your arm is folded close to your body, the same shoulder motion has a smaller effect. The Jacobian captures this configuration-dependent relationship.

### Damped Least-Squares IK

Inverse kinematics asks: given a desired end-effector pose `x_target`, find joint angles `q` such that `FK(q) = x_target`.

One approach: iteratively update `q` using the Jacobian. If the current EE pose has error `e = x_target - FK(q)`, we want `dq` such that `J · dq = e`. This is an overdetermined/underdetermined linear system (7 unknowns, 6 equations for the Panda), so there's no unique solution.

The pseudoinverse solution (`dq = J⁺ · e`) works, but near singularities `J⁺` blows up — a tiny task error requires enormous joint motions.

The fix is **Damped Least-Squares (DLS)**:

```
dq = Jᵀ (JJᵀ + λI)⁻¹ · e
```

The term `+ λI` (adding λ times the identity matrix to `JJᵀ`) is called **Tikhonov regularization** or damping. It ensures the matrix being inverted is always well-conditioned (never singular), at the cost of introducing a small error in the task-space convergence. Near singularities, the damping "softens" the joint update — the robot moves more slowly and smoothly through singular configurations instead of making large jerky jumps.

**Choosing λ:** Too small → unstable near singularities. Too large → slow convergence everywhere. `λ = 1e-4` (used in this project) is a commonly used default for robot arms with joint angles in radians.

---

## Part 3: The Pinhole Camera Model

Understanding cameras mathematically is essential for any robotics work involving vision.

### Projection

A pinhole camera maps a 3D point `(X, Y, Z)` in camera frame to a 2D pixel `(u, v)`:

```
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

Where:
- `fx`, `fy` are the focal lengths in pixels (for most cameras, `fx ≈ fy`)
- `cx`, `cy` is the principal point — where the optical axis hits the image plane, usually near the image center
- `Z` is the depth (distance along the camera's Z axis)

Notice the `/Z`: this is the perspective projection. Objects farther away appear smaller. The `/Z` is why 3D reconstruction from a single image is inherently ambiguous — you can't tell if something is small and close or large and far away.

### Back-Projection (Lifting 2D to 3D)

If you know the depth `Z` of a pixel `(u, v)`, you can recover the 3D point:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

This is exactly what `src/calibration/surface.py:pixel_to_3d()` does. The critical prerequisite is knowing `Z` — the metric depth at that pixel. This is what Depth Anything V2 provides.

### Intrinsics and Extrinsics

**Intrinsic parameters** describe the camera's internal geometry: `(fx, fy, cx, cy)`. These are fixed properties of the camera hardware. They can be measured by photographing a calibration checkerboard at many orientations (using OpenCV's `calibrateCamera` function), or estimated from the known FOV as done in this project.

**Extrinsic parameters** describe where the camera is in the world: the rotation and translation from the world (or robot) coordinate frame to the camera frame. These must be calibrated for every new camera placement — which is what the 4-corner table calibration accomplishes.

### Why Monocular Depth Is Hard

The projection equation shows the problem: when you project from 3D to 2D, the depth `Z` disappears. A photograph records `(u, v)` but discards `Z`. Recovering `Z` from a single image requires additional constraints — geometric priors about the scene, learned priors from training data (what Depth Anything V2 uses), or stereo/structured light.

Depth Anything V2 with the Hypersim metric checkpoint was trained on large datasets of indoor scenes with ground-truth depth. It learns the statistical relationship between appearance (textures, shadows, relative sizes) and metric depth. But this is a learned prior — it can fail on scenes that look different from training data, and it has no mechanism to be exactly correct.

---

## Part 4: How Robots Are Controlled

### The Control Stack

Modern robot controllers are organized in a hierarchy from high-level reasoning down to low-level hardware:

```
Task Planner (seconds timescale): "pick up the cup, put it in the box"
    ↓
Motion Planner (100ms timescale): "move from configuration A to configuration B"
    ↓
Trajectory Tracker (10ms timescale): "track this joint trajectory in real time"
    ↓
Joint Controller (1ms timescale): "send these torque commands to the motors"
    ↓
Hardware (electrical): servo drives, encoders, motors
```

Machine learning (specifically imitation learning) currently operates at the **task planner and motion planner** levels. The lower levels remain classical control.

### PID Control

PID (Proportional-Integral-Derivative) is the fundamental building block of low-level robot control. For joint position control:

```
torque(t) = Kp * (q_desired - q_actual) 
          + Ki * integral(q_desired - q_actual)
          + Kd * derivative(q_desired - q_actual)
```

- **Proportional (Kp):** Apply force proportional to the error. The bigger the error, the bigger the correction. Problem: a pure P controller has steady-state error (the robot never quite reaches the target — it only stops when the correction is balanced by friction).
- **Integral (Ki):** Accumulate error over time and apply a correction proportional to the accumulated error. Eliminates steady-state error. Problem: can cause oscillation if too large.
- **Derivative (Kd):** Apply a damping force proportional to how fast the error is changing. Prevents overshoot and oscillation. Problem: amplifies sensor noise.

Real robots don't expose PID gains to the machine learning researcher — the hardware controller handles this. But understanding PID matters because it explains why robot joints can track desired position trajectories smoothly, which is the assumption when we output joint angle sequences from IK.

### Impedance Control

Pure position control is stiff — the robot blindly tracks the commanded trajectory regardless of external forces. If an obstacle is in the way, it pushes harder and harder until something breaks.

Impedance control makes the robot behave like a mechanical spring-damper system. Instead of commanding a position, you define a virtual spring with some stiffness K connecting the desired position to the actual position:

```
F = K * (x_desired - x_actual) + D * (ẋ_desired - ẋ_actual)
```

Low K = compliant (soft) robot that yields to obstacles. High K = stiff robot that closely tracks the commanded trajectory. Impedance control is why you can safely physically guide a Panda by hand — the robot yields rather than fighting you.

For imitation learning, the Franka Panda is often controlled in **Cartesian impedance mode**: you send desired EE positions, and the robot tracks them with a calibrated compliance. This is simpler for the learning system than joint-level control.

---

## Part 5: Imitation Learning in Depth

### The Demonstration Distribution

Imitation learning starts with a dataset D = {(o₁, a₁), (o₂, a₂), ..., (oₙ, aₙ)} where:
- `oᵢ` is an observation at time step i (camera image, joint angles, etc.)
- `aᵢ` is the action the human took at that step (joint angles, EE velocity, gripper command)

The simplest approach, **Behavioral Cloning (BC)**, treats this as a supervised learning problem: learn a function `π(o) → a` that maps observations to actions, minimizing the prediction error on the training data.

**This seems straightforward. Why doesn't it always work?**

### The Covariate Shift Problem

Here's the fundamental issue with BC: the training data only covers the distribution of observations that appear when a *human* is controlling the robot. But at test time, you're deploying a *learned policy*, which will make slightly different choices — and those choices lead to slightly different states — which lead to observations that never appeared in the training data.

Concretely: if the human always grasped the cup from slightly to the left, the policy only knows what to do from slightly-left positions. If noise pushes the robot slightly to the right, the policy encounters an observation it has never seen and makes an arbitrary decision. That arbitrary decision leads to a worse state. The errors compound.

This is called **covariate shift**: the distribution of observations at training time (human policy) differs from the distribution at test time (learned policy). BC does not account for this gap.

### DAgger and Interactive Collection

**DAgger** (Dataset Aggregation, Ross et al. 2011) fixes covariate shift interactively. The algorithm:

1. Train an initial policy π₁ on the original demonstration dataset D₁.
2. Run π₁ in the real environment, reaching states outside the training distribution.
3. Ask a human expert to label the actions at those new states.
4. Add the new (state, action) pairs to D, retrain to get π₂.
5. Repeat.

DAgger provably converges to the expert policy (under some assumptions), because it trains on the states the policy actually visits. The downside is it requires ongoing human involvement — the human must be available to label new states during training.

### Action Chunking with Transformers (ACT)

ACT (Zhao et al., 2023) was introduced alongside ALOHA, a low-cost bimanual teleoperation system. It addresses two problems: the multimodality of demonstrations and the temporal correlation of actions.

**Multimodality:** When different demonstrations perform the same task in different ways (one human always moves left around the obstacle, another always moves right), a policy trained with mean squared error will predict the average of both behaviors — neither of which is actually a valid trajectory. A Conditional Variational Autoencoder (CVAE) architecture handles this by learning a distribution over possible actions conditioned on the observation, rather than a single point estimate.

**Action chunking:** Instead of predicting one action at a time, ACT predicts a chunk of `k` future actions simultaneously. At inference, the chunk is executed open-loop (without re-querying the policy until the chunk is complete), then a new chunk is predicted. This reduces the frequency of querying the potentially noisy policy and allows the robot to commit to a smooth short-horizon trajectory rather than reacting jerkily at every time step.

**Temporal ensembling:** At each time step, ACT has predictions from multiple overlapping action chunks (the current chunk plus previous chunks that extend into the present). These are averaged with exponential weighting toward the most recent predictions, smoothing out inconsistencies between overlapping chunks.

### Diffusion Policy

Diffusion Policy (Chi et al., 2023) takes a completely different approach to the multimodality problem. It models the action distribution using a **diffusion model** — the same type of generative model used in Stable Diffusion for images.

The key insight: instead of directly predicting an action `a` from observation `o`, train a model to *denoise* actions. Start with random Gaussian noise in action space, and iteratively remove the noise using a learned denoising network conditioned on the observation. After `T` denoising steps, you have a sample from the action distribution.

**Why this handles multimodality:** A denoising process can converge to different modes of the distribution from different noise initializations. If the data has two clusters of demonstrations (go left vs go right), the denoiser learns to push random noise toward whichever cluster is closest. This is something MSE regression fundamentally cannot do — MSE always averages the modes.

**Observation encoding:** Diffusion Policy uses a CNN or Transformer encoder to process the observation image into a feature vector, which conditions the denoising network at each step.

**Practical tradeoff vs ACT:** Diffusion Policy tends to produce smoother, higher-quality trajectories and handles multimodal data better. ACT is simpler to implement and faster at inference (no iterative denoising). Both are used in practice; the choice depends on the specific task.

---

## Part 6: Simulation for Robot Learning

### Why We Need Simulation

Real robot experiments are expensive in three dimensions:

**Time:** A robot performing a 10-second manipulation task takes 10 seconds. Running 1 million training iterations at 30fps requires 9+ hours of real time, just for rollouts. In simulation, you can run hundreds of environments in parallel on a GPU and run at 10,000+ FPS.

**Cost:** Robot time is valuable. If you're training a policy that initially has random behavior, those random actions will collide with objects, drop things, and potentially damage the robot or the environment. With simulation, crashes are free.

**Data coverage:** Real-world data collection is constrained to whatever scenes you can physically set up. Simulation allows generating arbitrarily diverse training data: different lighting, different object positions, different robot base positions, different viewpoints.

### MuJoCo in Depth

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine created by Emo Todorov at the University of Washington and now maintained by Google DeepMind. It is the dominant simulation platform for robot manipulation research.

**What makes MuJoCo good for manipulation:**
- Fast and accurate contact dynamics using a **convex optimization formulation** for contact forces (unlike older simulators that used penalty-based contact, which was numerically unstable)
- Efficient computation of analytical Jacobians (`mj_jacBody`), critical for fast IK
- Rich MJCF (XML) format for defining scenes
- Python bindings (`import mujoco`) for easy integration

**The XML format (MJCF):**
```xml
<mujoco>
  <worldbody>
    <light name="main" pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.01"/>
    <body name="box" pos="0.3 0 0.05">
      <joint name="box_x" type="slide" axis="1 0 0"/>
      <joint name="box_y" type="slide" axis="0 1 0"/>
      <geom name="box_geom" type="box" size="0.03 0.03 0.03" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
```

Bodies (rigid objects), joints (degrees of freedom), geoms (collision geometry + visual mesh), actuators (what drives the joints), and sensors are the main building blocks.

**`mj_forward` vs `mj_step`:**
- `mj_forward`: computes the forward kinematics — given current `qpos` (joint positions), compute all body positions/orientations, Jacobians, sensor values. Does NOT advance time.
- `mj_step`: advances the simulation by one timestep — computes contact forces, integrates dynamics, updates `qpos` and `qvel`. This is the full physics simulation.

For robot rendering (as in this project), you only need `mj_forward`. For training with physical interactions (contact dynamics, grasping), you need `mj_step`.

### The Sim-to-Real Gap

The biggest challenge with simulation-based training is the **sim-to-real gap**: policies trained in simulation often fail when deployed on real hardware. The gap has several sources:

**Visual gap:** Simulated renders look different from real cameras. Lighting, shadows, material textures, reflections, and sensor noise all differ. A CNN trained on simulation images may not recognize the same objects in real camera images.

**Dynamics gap:** Simulation makes simplifying assumptions about friction, compliance, inertia, and contact. A real robot joint has gear backlash, cable elasticity, and thermal-dependent friction. A real object has a different mass distribution than assumed.

**Delay gap:** Real control systems have communication latency (sending commands) and sensor latency (reading state). Simulation is often zero-delay.

**Domain Randomization:** The dominant technique for bridging the gap is to randomize simulation parameters during training: vary object positions, textures, lighting, camera viewpoint, physics parameters (friction coefficients, masses). Train the policy to be robust to all these variations. At test time, the real world is just one more instance in this randomized distribution.

---

## Part 7: The Data Collection Landscape

Understanding how training data is collected is central to robot learning research — it determines what's possible, what's expensive, and what limitations the resulting policies will have.

### Kinesthetic Teaching

The human physically grabs the robot arm and guides it through the desired motion. The robot records its own joint angles (and forces/torques) during the demonstration. Since the robot is the "recorder," the output is already in joint space — no retargeting needed.

*Best for:* Tasks that require feeling the robot (compliance important), low-cost single demos, situations where the robot is already present.
*Problems:* Slow and awkward for the operator; can't demonstrate at full speed; hard to coordinate two arms simultaneously.

### Leader-Follower Teleoperation

Two identical robots are mechanically or electronically coupled. The "leader" is physically controlled by the human; the "follower" (the actual robot being recorded) mirrors the leader's movements. ALOHA uses this approach: two modified ViperX robot arms as the leader, a low-cost bimanual setup.

*Best for:* Bimanual tasks, natural and fast data collection, high DOF tasks.
*Problems:* Requires two identical robots or a custom leader device; still requires physical access to the setup.

### Device-Based Teleoperation

A 6DOF input device (SpaceMouse, Oculus controller, joystick) controls the robot's end-effector. The human moves the device; the robot EE follows via Cartesian impedance control, and the resulting joint angles are recorded.

*Best for:* Remote data collection; when the operator doesn't need to be at the robot; relatively natural for operators who learn the interface.
*Problems:* Cognitive load of mapping device movements to robot movements; no haptic feedback (can't feel if the grasp succeeded); SpaceMouse has limited DOF for full 6DOF control.

### UMI — Universal Manipulation Interface

Chi et al. (2024) proposed a novel approach: strap a GoPro camera to a hand-held gripper (a physical handle with two finger-like protrusions). The operator physically performs the manipulation task while holding the gripper. The video from the wrist camera, combined with accelerometer data and known gripper geometry, provides EE trajectory and observation simultaneously.

The key insight: you don't need the robot during data collection. You collect the data anywhere, then replay it on the actual robot. Policies trained on UMI data achieved strong generalization because:
1. The wrist camera observation matches exactly what the robot would see (since the robot would also have a wrist camera)
2. The collection setup is cheap and portable

*This is the same core idea as H2R,* but UMI uses RGBD from the wrist and requires a physical prop. H2R uses a stationary webcam and requires no props at all.

### Human Video Retargeting (This Project)

No props, no teleoperation device, no robot present. Just a webcam and a human performing the task naturally. The pipeline converts the natural motion into robot trajectories via IK.

*Advantages:* Zero hardware beyond a webcam; natural demonstration style; scales to any internet video (in principle).
*Limitations:* IK retargeting introduces errors; no physical contact sensing; depth from monocular is noisy; orientation not fully recoverable.

### Synthetic Data

Generated entirely in simulation: scripted demonstrations, motion planning, or RL policies in simulation. Can generate unlimited amounts of data without any human in the loop.

*Best for:* Geometric tasks where sim fidelity is sufficient; pretraining visual encoders; data augmentation.
*Problems:* Sim-to-real gap; hard to generate naturalistic grasp sequences; scripted demos may not cover the right distribution.

---

## Part 8: Foundation Models in Robotics

### The Scaling Hypothesis Applied to Robots

In NLP and vision, large-scale pretraining on internet data has produced models (GPT-4, DALL-E, Stable Diffusion) that generalize to new tasks in ways that smaller, task-specific models cannot. The natural question for robotics: can the same scaling approach work?

The answer is "possibly, but it's complicated." Language and images are abundant on the internet. Robot trajectory data is not. A language model can train on a trillion tokens; the largest public robot datasets have tens of thousands of demonstrations.

### RT-1 and RT-2 (Google DeepMind)

**RT-1** (Robotics Transformer 1, Brohan et al. 2022) trained a Transformer that takes a natural language instruction and a sequence of camera images as input, and outputs tokenized robot actions. The key contribution was the data scale: 130,000 demonstrations collected over 17 months across 13 robot arms, covering 700+ tasks.

**RT-2** (Brohan et al. 2023) went further: instead of training a robot-specific model, they fine-tuned a pretrained vision-language model (PaLM-E or PaLI-X) on robot data. The model could generalize to instructions and objects that never appeared in the robot dataset, by leveraging knowledge from internet pretraining. A robot asked to "pick up something that can be used to open a bottle" could identify a bottle opener it had never seen in robot demonstrations.

### VoxPoser

VoxPoser (Huang et al. 2023) uses a large language model (LLM) and a vision-language model (VLM) together to generate **3D value maps** — spatial fields over the robot's workspace specifying where it should go and what it should avoid. The LLM translates a language instruction into code that queries the VLM to identify relevant regions in the camera image, then combines them into a 3D affordance map. A motion planner then optimizes a trajectory through the affordance map.

Crucially, VoxPoser requires **zero robot training data**. It leverages only the pretrained LLM and VLM to perform manipulation tasks. The tradeoff: it's slow (several seconds per plan), and the quality degrades for tasks requiring precise contact.

### π₀ (pi-zero, Physical Intelligence 2024)

π₀ is currently the most capable general-purpose manipulation policy. It uses a flow matching model (a variant of diffusion) built on top of a pretrained vision-language backbone (based on PaliGemma). It was trained on a large proprietary dataset across multiple robot embodiments.

The key architectural insight: flow matching produces smooth, continuous action trajectories rather than the stochastic steps of standard diffusion. This matters for robot control because jerky high-frequency action noise is physically problematic.

---

## Part 9: What Research in This Field Actually Looks Like

### An Anatomy of a Robot Learning Paper

Most papers in this field have the same structure:

1. **Problem statement:** "Current methods for X fail because Y."
2. **Proposed approach:** "We propose Z, which addresses Y by..."
3. **Method:** The technical contribution — a new architecture, training procedure, data collection method, or formulation.
4. **Experiments:** On a real robot (essential), plus often simulation. Usually 10–50 trials per condition, reported as success rate.
5. **Ablations:** Remove one component of the proposed method, show performance drops. This demonstrates that the component is actually doing what you claim.
6. **Comparison to baselines:** Your method vs the current best approach vs a simple baseline (e.g., plain BC).

### What Counts as a Contribution

Not every paper introduces a new algorithm. Valid contributions in robot learning research include:
- **New task benchmark:** A new task that highlights a capability gap
- **New dataset:** Demonstrating that more/better data helps
- **New evaluation methodology:** A better way to measure success
- **New data collection method:** Making demonstration collection cheaper or more scalable
- **Negative results:** Showing that a widely believed assumption is wrong
- **System integration:** Making a pipeline that works end-to-end where no end-to-end system existed

**Where H2R fits:** It's a new data collection method paper — specifically, a demonstration that monocular video retargeting can produce usable robot training data without teleoperation hardware. The natural next step (which would make it publishable) is the validation experiment: show a policy trained on this data achieves a non-trivial success rate on a real robot.

### How to Evaluate Robot Learning

Robot learning papers report task success rates on a real robot. But running experiments on real hardware is expensive and slow — a single experiment might take a day to run. Good experimental practice:

**Number of trials:** 20–50 trials per condition is typical. With 30 trials, a 70% success rate means 21 successes ± a confidence interval of roughly ±16%. More trials = narrower confidence intervals.

**Conditions to test:** Your method, at least one strong baseline, and an ablation (your method minus the key component). So typically 3–4 conditions × 30 trials = 90–120 real robot trials minimum.

**Initial conditions:** Always specify and randomize where the object starts. A policy that only works when the object is in exactly one position is not useful.

**Fair comparisons:** Train all methods on the same number of demonstrations. Hyperparameter tune each method independently. Report all failures honestly.

### Key Open Problems (2025 State of the Field)

**1. Generalization to novel objects and scenes.** Current policies are brittle — trained on a red cup, they fail on a blue cup. Solving this requires better visual representations, more diverse training data, or some form of abstraction.

**2. Contact-rich manipulation.** Screwing in a bolt, peeling a sticker, folding cloth — tasks where the control of contact forces matters. These require haptic sensing and dynamics modeling that current visual-only policies lack.

**3. Long-horizon planning.** Preparing a meal involves hundreds of steps over minutes. Current policies work for 5–30 second tasks. Long-horizon requires memory, sub-task composition, and error recovery.

**4. Bimanual dexterous manipulation.** Using two hands in a coordinated way — opening a jar, tying a knot — is enormously harder than single-arm manipulation. Data is scarce; the coordination problem is hard.

**5. Scalable data collection.** Training a general-purpose policy requires millions of diverse demonstrations. How do you collect them without spending millions of dollars and thousands of operator-hours? This is the exact problem H2R is attempting to address.

**6. Sim-to-real for contact tasks.** Simulation transfer for free-space motion is mostly solved. Transfer for contact-rich manipulation (grasping, pushing, screwing) remains open because contact dynamics are hard to simulate accurately.

---

## Part 10: Your Path Forward

You have completed a working end-to-end robot learning pipeline. That puts you ahead of most people who start in this field — you have practical intuition that usually only comes after months of coursework.

Here is the honest assessment of where you stand and what to do next:

**What you now understand from doing:**
- Camera geometry and depth sensing (and their limits)
- IK solvers and why warmstarting matters
- Trajectory representation and smoothing
- Robot simulation and rendering
- The full data collection → IK → simulation pipeline

**What you need to build from first principles:**
- The mathematics: rotation matrices, Jacobians, optimization theory — work through these carefully, not just "used in a library"
- Imitation learning algorithms: read the ACT and Diffusion Policy papers, then implement BC from scratch on a simple task
- Control theory: PID, impedance control — understanding how the robot actually moves

**The most important thing you can do next:**
Deploy a policy. Take the data this pipeline generates, train ACT or Diffusion Policy on it, and run it on a real robot (or simulate a complete pick-place task with a known object). The moment you see a policy actually work — or fail in an interesting way — your intuition for this field will jump by an order of magnitude.

The field is young, moving fast, and desperately needs people who can build real systems. You already proved you can do that. Now build the theoretical foundation to know *why* things work, and you'll be positioned to contribute original work.

---

*This primer covers the concepts that appeared in building H2R and the adjacent territory you'll need for research. The companion checklist (LEARNING_ROADMAP.md) turns this into a structured study plan with specific papers, projects, and milestones.*
