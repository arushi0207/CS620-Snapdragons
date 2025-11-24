from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from .base import BaseFeatureExtractor
from ..registry import register_extractor

# 🔁 New: use FastCV bridge instead of MediaPipe
from ..fastcv_bridge import fastcv_detect_faces


@register_extractor("mediapipe_face")
class MediaPipeFaceExtractor(BaseFeatureExtractor):
    """
    FastCV-backed face detector mimicking the MediaPipeFaceExtractor interface.

    Produces a dictionary with bounding box, keypoints, ROI corners, and landmarks
    for the best-scoring face in the frame. Landmarks are not provided by FastCV
    and are currently left empty.
    """

    def setup(self):
        # No heavy model load needed for FastCV; DLL is loaded by fastcv_bridge.
        pass

    def extract(
        self, frame_bgr: "np.ndarray", *, timestamp: float | None = None
    ) -> Dict[str, Any]:
        # Convert BGR -> RGB (to stay consistent with the rest of your pipeline)
        frame_rgb = np.ascontiguousarray(self.bgr2rgb(frame_bgr))

        # Grayscale for FastCV MSER-based detection
        # Y' = 0.299 R + 0.587 G + 0.114 B
        gray = np.dot(frame_rgb[..., :3], [0.299, 0.587, 0.114]).astype(
            np.uint8, copy=False
        )

        # Call into your C FastCV bridge
        rects = fastcv_detect_faces(gray, max_faces=1)

        payload: Dict[str, Any] = {
            "bbox": None,
            "keypoints": [],
            "roi_corners": [],
            "landmarks": [],
            "num_landmarks": 0,
            # You can add confidence_* later if you derive a score
        }

        if not rects:
            return {"mediapipe_face": payload}

        # Take the first detected region as the "face"
        r = rects[0]
        x = float(r.x)
        y = float(r.y)
        w = float(r.width)
        h = float(r.height)

        # bbox in [x_min, y_min, x_max, y_max] format
        payload["bbox"] = [x, y, x + w, y + h]

        # Use rectangle corners as simple keypoints + ROI corners
        corners = [
            (x, y),         # top-left
            (x + w, y),     # top-right
            (x + w, y + h), # bottom-right
            (x, y + h),     # bottom-left
        ]
        payload["keypoints"] = [[cx, cy] for cx, cy in corners]
        payload["roi_corners"] = [[cx, cy] for cx, cy in corners]

        # No true landmarks from FastCV; leave empty for now
        payload["landmarks"] = []
        payload["num_landmarks"] = 0

        return {"mediapipe_face": payload}

    # --- Keep these helpers for compatibility (even though we don't use them now) ---

    @staticmethod
    def _first_non_empty_tensor(
        tensors: Iterable[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.nelement() > 0:
                return tensor
        return None

    @staticmethod
    def _squeeze_detection_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        MediaPipe returns tensors with an explicit detection dimension.
        We only retain the first detection (best face) in the batch.
        """
        if tensor.dim() >= 3:
            return tensor[0]
        return tensor

    def _process_landmarks(
        self, landmark_batches: list[list[torch.Tensor]]
    ) -> Dict[str, Any]:
        # FastCV backend does not supply landmarks yet.
        return {"landmarks": [], "num_landmarks": 0}
