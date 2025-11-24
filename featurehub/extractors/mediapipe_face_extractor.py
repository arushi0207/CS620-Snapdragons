from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

import torch

from .base import BaseFeatureExtractor
from ..registry import register_extractor

# from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from qai_hub_models.models.mediapipe_face.model import MediaPipeFace
from featurehub.native.fastcv_bridge import fastcv_detect_faces


@register_extractor("mediapipe_face")
class MediaPipeFaceExtractor(BaseFeatureExtractor):
    """
    Wrapper for the MediaPipe Face detector + landmark models.

    Produces a dictionary with bounding box, keypoints, ROI corners, and 2D landmarks
    (with per-point confidence) for the best-scoring face in the frame.
    """

    def setup(self):
        model = MediaPipeFace.from_pretrained()
        self.app = MediaPipeFaceApp.from_pretrained(model)

    def extract(
        self, frame_bgr: "np.ndarray", *, timestamp: float | None = None
    ) -> Dict[str, Any]:
        frame_rgb = np.ascontiguousarray(self.bgr2rgb(frame_bgr))

        (
            batched_boxes,
            batched_keypoints,
            batched_roi_corners,
            *landmark_batches,
        ) = self.app.predict_landmarks_from_image(frame_rgb, raw_output=True)

        payload: Dict[str, Any] = {
            "bbox": None,
            "keypoints": [],
            "roi_corners": [],
            "landmarks": [],
            "num_landmarks": 0,
        }

        box_tensor = self._first_non_empty_tensor(batched_boxes)
        if box_tensor is not None:
            bbox_vals = box_tensor.reshape(-1).detach().cpu().tolist()
            payload["bbox"] = [float(v) for v in bbox_vals]

        keypoint_tensor = self._first_non_empty_tensor(batched_keypoints)
        if keypoint_tensor is not None:
            pts = self._squeeze_detection_tensor(keypoint_tensor)
            payload["keypoints"] = [
                [float(x), float(y)] for x, y in pts.detach().cpu().tolist()
            ]

        roi_tensor = self._first_non_empty_tensor(batched_roi_corners)
        if roi_tensor is not None:
            roi_pts = self._squeeze_detection_tensor(roi_tensor)
            payload["roi_corners"] = [
                [float(x), float(y)] for x, y in roi_pts.detach().cpu().tolist()
            ]

        landmarks_payload = self._process_landmarks(landmark_batches)
        payload.update(landmarks_payload)

        return {"mediapipe_face": payload}

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
        if not landmark_batches:
            return {"landmarks": [], "num_landmarks": 0}

        batched_landmarks = landmark_batches[0] if landmark_batches else []
        landmark_tensor = self._first_non_empty_tensor(batched_landmarks)
        if landmark_tensor is None or landmark_tensor.nelement() == 0:
            return {"landmarks": [], "num_landmarks": 0}

        landmarks = self._squeeze_detection_tensor(landmark_tensor).detach().cpu()
        landmark_list = landmarks.tolist()

        payload: Dict[str, Any] = {
            "landmarks": [
                [float(x), float(y), float(conf)] for x, y, conf in landmark_list
            ],
            "num_landmarks": int(landmarks.shape[0]),
        }

        confidences = landmarks[:, 2]
        if confidences.numel() > 0:
            payload["confidence_mean"] = float(confidences.mean().item())
            payload["confidence_min"] = float(confidences.min().item())
        return payload
