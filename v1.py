import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np
import torch
import time

from depth_anything_v2.dpt import DepthAnythingV2

# =========================================================
# Helper Functions
# =========================================================

def normalize(v):

    norm = np.linalg.norm(v)

    if norm < 1e-8:
        return v

    return v / norm


def pixel_to_3d(
    u,
    v,
    z,
    fx,
    fy,
    cx,
    cy
):

    X = ((u - cx) * z) / fx
    Y = ((v - cy) * z) / fy
    Z = z

    return np.array([X, Y, Z])


def clamp_velocity(
    current,
    previous,
    max_step
):

    delta = current - previous

    distance = np.linalg.norm(delta)

    if distance > max_step:

        delta = (
            delta / distance
        ) * max_step

    return previous + delta


def map_hand_to_robot(hand_pos):

    x, y, z = hand_pos

    # =====================================================
    # Human Space -> Robot Space
    # =====================================================

    robot_x = 0.45 + x * 0.35
    robot_y = y * 0.35
    robot_z = 0.45 + z * 0.25

    # =====================================================
    # Workspace Limits
    # =====================================================

    robot_x = np.clip(
        robot_x,
        0.2,
        0.75
    )

    robot_y = np.clip(
        robot_y,
        -0.45,
        0.45
    )

    robot_z = np.clip(
        robot_z,
        0.1,
        0.8
    )

    return np.array([
        robot_x,
        robot_y,
        robot_z
    ])


def build_grasp_rotation(
    grasp_axis,
    approach_direction
):

    x_axis = normalize(
        grasp_axis
    )

    z_axis = normalize(
        approach_direction
    )

    y_axis = np.cross(
        z_axis,
        x_axis
    )

    y_axis = normalize(
        y_axis
    )

    # Re-orthogonalize
    x_axis = np.cross(
        y_axis,
        z_axis
    )

    x_axis = normalize(
        x_axis
    )

    R = np.column_stack([
        x_axis,
        y_axis,
        z_axis
    ])

    return R


# =========================================================
# Device
# =========================================================

DEVICE = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print(f"Using device: {DEVICE}")

# =========================================================
# Depth Anything V2
# =========================================================

model_configs = {
    'vits': {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [
            48,
            96,
            192,
            384
        ]
    }
}

depth_model = DepthAnythingV2(
    **model_configs['vits']
)

depth_model.load_state_dict(
    torch.load(
        'checkpoints/depth_anything_v2_vits.pth',
        map_location=DEVICE
    )
)

depth_model = (
    depth_model
    .to(DEVICE)
    .eval()
)

print("Depth model loaded.")

# =========================================================
# MediaPipe
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# =========================================================
# Webcam
# =========================================================

cap = cv2.VideoCapture(0)

# =========================================================
# Camera Intrinsics
# =========================================================

fx = 600
fy = 600

cx = 320
cy = 240

# =========================================================
# Temporal Stabilization
# =========================================================

ALPHA = 0.2

MAX_STEP = 0.08

smoothed_grasp = np.array([
    0.0,
    0.0,
    0.0
])

previous_grasp = np.array([
    0.0,
    0.0,
    0.0
])

# =========================================================
# MuJoCo Setup
# =========================================================

model = mujoco.MjModel.from_xml_path(
    "robot/panda.xml"
)

data = mujoco.MjData(model)

# =========================================================
# End Effector
# =========================================================

ee_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

print(
    "End effector body id:",
    ee_body_id
)

# =========================================================
# IK Parameters
# =========================================================

IK_GAIN = 2.0

DAMPING = 1e-4

# =========================================================
# Initial Robot Pose
# =========================================================

initial_qpos = np.array([
    0.0,
    -0.5,
    0.0,
    -2.0,
    0.0,
    1.5,
    0.8
])

data.qpos[:7] = initial_qpos

mujoco.mj_forward(
    model,
    data
)

# =========================================================
# Viewer
# =========================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        # =================================================
        # Camera
        # =================================================

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1
        )

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # =================================================
        # Hand Tracking
        # =================================================

        hand_results = hands.process(
            rgb
        )

        # =================================================
        # Depth Estimation
        # =================================================

        depth = depth_model.infer_image(
            frame
        )

        # =================================================
        # Depth Visualization
        # =================================================

        depth_vis = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        depth_vis = cv2.applyColorMap(
            depth_vis,
            cv2.COLORMAP_INFERNO
        )

        # =================================================
        # Hand Processing
        # =================================================

        if hand_results.multi_hand_landmarks:

            for hand_landmarks in (
                hand_results.multi_hand_landmarks
            ):

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # =========================================
                # Landmark Storage
                # =========================================

                landmarks_3d = {}
                landmarks_2d = {}

                for idx, lm in enumerate(
                    hand_landmarks.landmark
                ):

                    px = int(lm.x * w)
                    py = int(lm.y * h)

                    px = np.clip(
                        px,
                        0,
                        w - 1
                    )

                    py = np.clip(
                        py,
                        0,
                        h - 1
                    )

                    d = depth[py, px]

                    point_3d = pixel_to_3d(
                        px,
                        py,
                        d,
                        fx,
                        fy,
                        cx,
                        cy
                    )

                    landmarks_3d[idx] = point_3d

                    landmarks_2d[idx] = (
                        px,
                        py
                    )

                # =========================================
                # Important 3D Landmarks
                # =========================================

                wrist_3d = landmarks_3d[0]

                thumb_tip_3d = (
                    landmarks_3d[4]
                )

                index_tip_3d = (
                    landmarks_3d[8]
                )

                index_mcp_3d = (
                    landmarks_3d[5]
                )

                pinky_mcp_3d = (
                    landmarks_3d[17]
                )

                # =========================================
                # 2D Points
                # =========================================

                thumb_tip_2d = (
                    landmarks_2d[4]
                )

                index_tip_2d = (
                    landmarks_2d[8]
                )

                # =========================================
                # Grasp Position
                # =========================================

                grasp_position = (
                    thumb_tip_3d
                    + index_tip_3d
                ) / 2.0

                # =========================================
                # Stabilization
                # =========================================

                clamped_grasp = (
                    clamp_velocity(
                        grasp_position,
                        previous_grasp,
                        MAX_STEP
                    )
                )

                smoothed_grasp = (
                    ALPHA
                    * clamped_grasp
                    + (1 - ALPHA)
                    * smoothed_grasp
                )

                previous_grasp = (
                    smoothed_grasp.copy()
                )

                # =========================================
                # Robot Position Target
                # =========================================

                robot_target = (
                    map_hand_to_robot(
                        smoothed_grasp
                    )
                )

                # =========================================
                # Grasp Orientation
                # =========================================

                grasp_axis = normalize(
                    index_tip_3d
                    - thumb_tip_3d
                )

                palm_a = normalize(
                    index_mcp_3d
                    - wrist_3d
                )

                palm_b = normalize(
                    pinky_mcp_3d
                    - wrist_3d
                )

                approach_direction = normalize(
                    np.cross(
                        palm_a,
                        palm_b
                    )
                )

                target_rot = (
                    build_grasp_rotation(
                        grasp_axis,
                        approach_direction
                    )
                )

                # =========================================
                # Current EE Pose
                # =========================================

                current_pos = (
                    data.xpos[
                        ee_body_id
                    ].copy()
                )

                current_rot = (
                    data.xmat[
                        ee_body_id
                    ]
                    .reshape(3, 3)
                    .copy()
                )

                # =========================================
                # Position Error
                # =========================================

                pos_error = (
                    robot_target
                    - current_pos
                )

                # =========================================
                # Orientation Error
                # =========================================

                rot_error_matrix = (
                    target_rot
                    @ current_rot.T
                )

                rot_error = np.array([

                    rot_error_matrix[2,1]
                    - rot_error_matrix[1,2],

                    rot_error_matrix[0,2]
                    - rot_error_matrix[2,0],

                    rot_error_matrix[1,0]
                    - rot_error_matrix[0,1]

                ]) * 0.5

                # =========================================
                # Full Task Error
                # =========================================

                task_error = np.concatenate([
                    pos_error,
                    rot_error
                ])

                # =========================================
                # Jacobian
                # =========================================

                jacp = np.zeros(
                    (3, model.nv)
                )

                jacr = np.zeros(
                    (3, model.nv)
                )

                mujoco.mj_jacBody(
                    model,
                    data,
                    jacp,
                    jacr,
                    ee_body_id
                )

                # =========================================
                # Full 6D Jacobian
                # =========================================

                J_full = np.vstack([
                    jacp[:, :7],
                    jacr[:, :7]
                ])

                # =========================================
                # Damped Least Squares IK
                # =========================================

                JT = J_full.T

                identity = np.eye(6)

                inv = np.linalg.inv(
                    J_full @ JT
                    + DAMPING * identity
                )

                delta_q = (
                    JT
                    @ inv
                    @ task_error
                )

                # =========================================
                # Joint Targets
                # =========================================

                target_q = (
                    data.qpos[:7]
                    + IK_GAIN
                    * delta_q
                )

                # =========================================
                # Joint Limits
                # =========================================

                target_q = np.clip(
                    target_q,
                    model.jnt_range[:7, 0],
                    model.jnt_range[:7, 1]
                )

                # =========================================
                # Apply Control
                # =========================================

                data.ctrl[:7] = target_q

                # =========================================
                # Gripper Control
                # =========================================

                grasp_width = np.linalg.norm(
                    index_tip_3d
                    - thumb_tip_3d
                )

                gripper_cmd = np.clip(
                    grasp_width * 5.0,
                    0.0,
                    0.04
                )

                data.ctrl[7] = (
                    gripper_cmd
                )

                # =========================================
                # Draw Interaction
                # =========================================

                cv2.circle(
                    frame,
                    thumb_tip_2d,
                    10,
                    (255, 0, 0),
                    cv2.FILLED
                )

                cv2.circle(
                    frame,
                    index_tip_2d,
                    10,
                    (0, 255, 0),
                    cv2.FILLED
                )

                cv2.line(
                    frame,
                    thumb_tip_2d,
                    index_tip_2d,
                    (0, 255, 255),
                    3
                )

                # =========================================
                # UI
                # =========================================

                cv2.putText(
                    frame,
                    "FULL 6DOF IK CONTROL",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Pos Error: "
                    f"{np.linalg.norm(pos_error):.3f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Rot Error: "
                    f"{np.linalg.norm(rot_error):.3f}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Grip: "
                    f"{gripper_cmd:.3f}",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

        # =================================================
        # Physics Step
        # =================================================

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        # =================================================
        # Windows
        # =================================================

        cv2.imshow(
            "6DOF Hand Robot Control",
            frame
        )

        cv2.imshow(
            "Depth Map",
            depth_vis
        )

        # =================================================
        # Quit
        # =================================================

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

# =========================================================
# Cleanup
# =========================================================

cap.release()

cv2.destroyAllWindows()