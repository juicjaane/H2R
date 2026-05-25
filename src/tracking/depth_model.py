"""
Metric depth estimation using Depth Anything V2 (Hypersim indoor checkpoint).
Returns depth in meters. Use infer_frame() for single frames in the pipeline.

IMPORTANT: This uses the metric variant from metric_depth/depth_anything_v2/dpt.py
           NOT the standard relative-depth model. Do not confuse the two.
"""

import sys
import numpy as np
import torch

# Ensure metric_depth path is on sys.path (done by src/config.py import)
import src.config  # noqa: F401 — side effect: adds DAV2 metric path to sys.path
from depth_anything_v2.dpt import DepthAnythingV2
from src.config import (
    DEPTH_METRIC_CKPT,
    DEPTH_MODEL_CONFIG,
    DEPTH_METRIC_MAX_DEPTH,
)


class MetricDepthModel:
    """
    Thin wrapper around the metric DAV2 model.

    Usage:
        model = MetricDepthModel()
        depth_m = model.infer_frame(bgr_frame)  # numpy HxW float32, meters
    """

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cfg = {**DEPTH_MODEL_CONFIG, "max_depth": DEPTH_METRIC_MAX_DEPTH}
        self._model = DepthAnythingV2(**cfg)

        ckpt_path = str(DEPTH_METRIC_CKPT)
        state = torch.load(ckpt_path, map_location="cpu")
        self._model.load_state_dict(state)
        self._model = self._model.to(device).eval()

        print(f"[MetricDepthModel] Loaded {ckpt_path} on {device}")

    def infer_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Args:
            bgr_frame: HxWx3 uint8 BGR image (as returned by cv2.VideoCapture)
        Returns:
            depth: HxW float32 array, values in meters (0 to ~20m indoors)
        """
        with torch.no_grad():
            depth = self._model.infer_image(bgr_frame)
        return depth.astype(np.float32)

    def sample_patch(
        self,
        depth: np.ndarray,
        px: int,
        py: int,
        radius: int = 2,
    ) -> float:
        """
        Return median depth over a (2r+1)×(2r+1) patch centred on (px, py).
        Avoids the silhouette-noise problem of single-pixel sampling.
        """
        h, w = depth.shape
        y0, y1 = max(0, py - radius), min(h, py + radius + 1)
        x0, x1 = max(0, px - radius), min(w, px + radius + 1)
        patch = depth[y0:y1, x0:x1]
        return float(np.median(patch))
