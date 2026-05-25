import cv2
import mediapipe as mp
import numpy as np

# ==========================================
# Helper Functions
# ==========================================

def normalize(v):

    norm = np.linalg.norm(v)

    if norm == 0:
        return v

    return v / norm


def angle_between(v1, v2):

    v1 = normalize(v1)
    v2 = normalize(v2)

    dot = np.dot(v1, v2)

    dot = np.clip(dot, -1.0, 1.0)

    return np.degrees(np.arccos(dot))


# ==========================================
# MediaPipe Setup
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)

# ==========================================
# Gesture Parameters
# ==========================================

PINCH_START = 40
PINCH_END = 55

POINT_THRESHOLD = 80

ALPHA = 0.2

smoothed_distance = 0

pinch_state = False

gesture = "INVALID"

# ==========================================
# Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    h, w, _ = frame.shape

    # ======================================
    # Hand Detection
    # ======================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==================================
            # Landmark Extraction
            # ==================================

            landmarks = {}

            for idx, lm in enumerate(hand_landmarks.landmark):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                landmarks[idx] = np.array([cx, cy])

            # ==================================
            # Important Landmarks
            # ==================================

            wrist = landmarks[0]

            thumb_mcp = landmarks[2]
            thumb_tip = landmarks[4]

            index_mcp = landmarks[5]
            index_tip = landmarks[8]

            pinky_mcp = landmarks[17]

            # ==================================
            # Draw Fingertips
            # ==================================

            cv2.circle(
                frame,
                tuple(thumb_tip),
                10,
                (255, 0, 0),
                cv2.FILLED
            )

            cv2.circle(
                frame,
                tuple(index_tip),
                10,
                (0, 255, 0),
                cv2.FILLED
            )

            # ==================================
            # Pinch Center
            # ==================================

            pinch_center = (
                (thumb_tip + index_tip) // 2
            )

            cv2.circle(
                frame,
                tuple(pinch_center),
                8,
                (0, 0, 255),
                cv2.FILLED
            )

            # ==================================
            # Pinch Distance
            # ==================================

            raw_distance = np.linalg.norm(
                index_tip - thumb_tip
            )

            smoothed_distance = (
                ALPHA * raw_distance
                + (1 - ALPHA) * smoothed_distance
            )

            # ==================================
            # Stable Pinch State
            # ==================================

            if (
                not pinch_state
                and smoothed_distance < PINCH_START
            ):

                pinch_state = True

            elif (
                pinch_state
                and smoothed_distance > PINCH_END
            ):

                pinch_state = False

            # ==================================
            # Palm Geometry
            # ==================================

            a = index_mcp - wrist
            b = pinky_mcp - wrist

            a_3d = np.array([a[0], a[1], 0])
            b_3d = np.array([b[0], b[1], 0])

            palm_normal = np.cross(a_3d, b_3d)

            palm_normal = normalize(palm_normal)

            # ==================================
            # Index Finger Direction
            # ==================================

            index_direction = (
                index_tip - index_mcp
            )

            index_direction_3d = np.array([
                index_direction[0],
                index_direction[1],
                0
            ])

            index_direction_3d = normalize(
                index_direction_3d
            )

            # ==================================
            # Pointing Angle
            # ==================================

            theta = angle_between(
                index_direction_3d,
                palm_normal
            )

            # ==================================
            # Gesture Classification
            # ==================================

            if pinch_state:

                gesture = "PINCH"

            elif (
                smoothed_distance > POINT_THRESHOLD
            ):

                gesture = "POINT"

            else:

                gesture = "INVALID"

            # ==================================
            # Visualizations
            # ==================================

            cv2.line(
                frame,
                tuple(thumb_tip),
                tuple(index_tip),
                (0, 255, 255),
                3
            )

            cv2.line(
                frame,
                tuple(wrist),
                tuple(index_mcp),
                (255, 255, 0),
                2
            )

            cv2.line(
                frame,
                tuple(wrist),
                tuple(pinky_mcp),
                (255, 0, 255),
                2
            )

            # ==================================
            # Display
            # ==================================

            cv2.putText(
                frame,
                f"Distance: {int(smoothed_distance)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Theta: {int(theta)}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

    else:

        gesture = "NO HAND"

        cv2.putText(
            frame,
            gesture,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ======================================
    # Show Window
    # ======================================

    cv2.imshow(
        "Gesture Interaction System",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()