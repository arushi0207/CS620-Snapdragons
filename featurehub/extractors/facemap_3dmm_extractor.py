from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np
import torch

from .base import BaseFeatureExtractor
from ..registry import register_extractor

from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.models.facemap_3dmm.model import MODEL_ID, FaceMap_3DMM
from qai_hub_models.models.facemap_3dmm.utils import (
    project_landmark,
    transform_landmark_coordinates,
)


@register_extractor("facemap_3dmm")
class FaceMap3DMMExtractor(BaseFeatureExtractor):
    """Wrapper for the FaceMap 3DMM model.

    Returns {"facemap_3dmm": {...}} containing landmarks, pose and coefficients.
    """

    def setup(self, is_test: bool = False):
        parser = get_on_device_demo_parser(
            get_model_cli_parser(FaceMap_3DMM), add_output_dir=True
        )
        args = parser.parse_args([] if is_test else [])
        model = demo_model_from_cli_args(FaceMap_3DMM, MODEL_ID, args)
        validate_on_device_demo_args(args, MODEL_ID)
        print("Model Loaded")

        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(target_device).eval()
        self.device = target_device
        self.input_height, self.input_width = FaceMap_3DMM.get_input_spec()["image"][0][2:]

    def extract(self, frame_bgr: np.ndarray, *, timestamp: float | None = None) -> Dict[str, Any]:
        img_rgb = frame_bgr[..., ::-1]
        x0, x1, y0, y1 = self._resolve_face_box(img_rgb.shape[1], img_rgb.shape[0])
        crop = img_rgb[y0 : y1 + 1, x0 : x1 + 1]
        resized = cv2.resize(
            crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        )
        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            .float()
            / 255.0
        )

        with torch.no_grad():
            output = self.model(tensor).squeeze(0).detach().cpu()

        payload = self._decode_output(output, (x0, y0, x1, y1))
        return {"facemap_3dmm": payload}

    def _resolve_face_box(self, width: int, height: int) -> Tuple[int, int, int, int]:
        norm_box: Iterable[float] | None = self.kwargs.get("face_box")
        if norm_box is not None:
            values = list(norm_box)
            if len(values) == 4:
                if all(0.0 <= v <= 1.0 for v in values):
                    left, right, top, bottom = values
                    return (
                        int(round(width * left)),
                        max(int(round(width * right)) - 1, 0),
                        int(round(height * top)),
                        max(int(round(height * bottom)) - 1, 0),
                    )
                return (
                    int(round(values[0])),
                    int(round(values[1])),
                    int(round(values[2])),
                    int(round(values[3])),
                )
        # Fallback: whole frame.
        return 0, width - 1, 0, height - 1

    def _decode_output(self, output: torch.Tensor, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        x0, y0, x1, y1 = bbox

        landmark = project_landmark(output)
        transform_landmark_coordinates(
            landmark,
            (x0, y0, x1, y1),
            self.input_height,
            self.input_width,
        )

        alpha_id = output[0:219]
        alpha_exp = output[219:258]
        pitch = output[258] * (np.pi / 2)
        yaw = output[259] * (np.pi / 2)
        roll = output[260] * (np.pi / 2)
        tx = output[261] * 60
        ty = output[262] * 60
        f = output[263] * 150 + 450

        payload: Dict[str, Any] = {
            "landmarks": landmark.detach().cpu().numpy().tolist(),
            "pose_rpy": [float(roll), float(pitch), float(yaw)],
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "translation": [float(tx), float(ty)],
            "focal_length": float(f),
            "coeffs": {
                "alpha_id": alpha_id.detach().cpu().numpy().tolist(),
                "alpha_exp": alpha_exp.detach().cpu().numpy().tolist(),
            },
        }
        return payload
