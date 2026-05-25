"""
MediaPipe hand tracker — wraps setup and per-frame inference.

Key landmarks used by this pipeline:
  0  = wrist
  4  = thumb tip
  5  = index MCP (knuckle)
  8  = index tip
  17 = pinky MCP (knuckle)
"""

from __future__ import annotations
from typing import Optional

import numpy as np
try:
    import mediapipe as mp
except ImportError:
    mp = None

# Landmark indices (named for readability throughout the codebase)
LM_WRIST       = 0
LM_THUMB_TIP   = 4
LM_INDEX_MCP   = 5
LM_INDEX_TIP   = 8
LM_PINKY_MCP   = 17


class HandTracker:
    """
    Wraps MediaPipe Hands for single-hand detection.

    Usage:
        tracker = HandTracker()
        result = tracker.process(rgb_frame)
        if result:
            px, py = result[LM_INDEX_TIP]
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._draw = mp.solutions.drawing_utils

    def process(self, rgb_frame: np.ndarray) -> Optional[dict[int, tuple[int, int]]]:
        """
        Run hand detection on an RGB frame.

        Returns:
            dict mapping landmark_id -> (px, py) in pixels, or None if no hand.
        """
        h, w = rgb_frame.shape[:2]
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        landmarks: dict[int, tuple[int, int]] = {}
        for idx, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            landmarks[idx] = (px, py)
        return landmarks

    def draw(self, bgr_frame: np.ndarray, landmarks_2d: dict[int, tuple[int, int]]) -> None:
        """Draw hand skeleton overlay on a BGR frame in-place (for visualisation only)."""
        # Rebuild a MediaPipe NormalizedLandmarkList for drawing
        h, w = bgr_frame.shape[:2]
        hand_lm = self._mp_hands.HandLandmark
        lm_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for i in range(21):
            lm = lm_list.landmark.add()
            if i in landmarks_2d:
                px, py = landmarks_2d[i]
                lm.x = px / w
                lm.y = py / h
            else:
                lm.x = lm.y = 0.0
        self._draw.draw_landmarks(
            bgr_frame, lm_list, self._mp_hands.HAND_CONNECTIONS
        )

    def close(self):
        self._hands.close()
