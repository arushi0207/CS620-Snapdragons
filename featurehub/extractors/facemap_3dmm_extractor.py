from __future__ import annotations
import importlib
from typing import Any, Dict
import numpy as np

from .base import BaseFeatureExtractor
from ..registry import register_extractor
from qai_hub_models.models.facemap_3dmm.app import FaceMap_3DMMApp

from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)

from qai_hub_models.models.facemap_3dmm.model import (
    INPUT_IMAGE_PATH,
    MODEL_ID,
    FaceMap_3DMM,
)


@register_extractor("facemap_3dmm")
class FaceMap3DMMExtractor(BaseFeatureExtractor):
    """封装 FaceMap 3DMM（qai_hub_models）。

    返回：{"facemap_3dmm": {"landmarks3d": Nx3, "pose_rpy": [r,p,y], "coeffs": {...}}}
    若模型不包含某字段则省略。
    """ 

    def setup(self, is_test: bool = False):
        model_cls = FaceMap_3DMM
        model_id = MODEL_ID
        parser = get_model_cli_parser(model_cls)
        parser = get_on_device_demo_parser(parser, add_output_dir=True)
        # parser.add_argument(
        #     "--face-box",
        #     type=_parse_face_box,
        #     default="0.0,1.0,0.0,1.0",
        #     help=(
        #         "Part of image where to apply face landmark algorithm. "
        #         "This should be centered around the face for best landmark performance. "
        #         "We recommend using a face detector to retrieve the face box (not included in this demo). "
        #         "The values are expressed as 'left,right,top,bottom' with floating point values "
        #         "normalized to [0, 1]."
        #     ),
        # )
        args = parser.parse_args([] if is_test else None)
        model = demo_model_from_cli_args(model_cls, model_id, args)
        validate_on_device_demo_args(args, model_id)
        print("Model Loaded")

        self.app = FaceMap_3DMMApp(model)

    def extract(self, frame_bgr: np.ndarray, *, timestamp: float | None = None) -> Dict[str, Any]:
        img_rgb = frame_bgr[..., ::-1]

        # 兼容不同 App API
        for fn_name in ("predict", "run_image", "__call__"):
            if hasattr(self.app, fn_name):
                out = getattr(self.app, fn_name)(img_rgb)
                break
        else:
            raise RuntimeError("FaceMap App has no callable predict/run_image.")

        payload: Dict[str, Any] = {}
        if isinstance(out, dict):
            # 常见键位：landmarks3d / landmarks / coeffs / pose
            if "landmarks3d" in out:
                payload["landmarks3d"] = np.asarray(out["landmarks3d"]).tolist()
            elif "landmarks" in out:
                # 2D 也收下
                payload["landmarks"] = np.asarray(out["landmarks"]).tolist()
            if "coeffs" in out:
                payload["coeffs"] = out["coeffs"]
            if "pose_rpy" in out:
                payload["pose_rpy"] = out["pose_rpy"]
            elif all(k in out for k in ("roll", "pitch", "yaw")):
                payload["pose_rpy"] = [out["roll"], out["pitch"], out["yaw"]]
        else:
            # 若直接返回 ndarray，假设为 Nx3 landmarks
            arr = np.asarray(out)
            if arr.ndim == 2 and arr.shape[1] in (2, 3):
                key = "landmarks3d" if arr.shape[1] == 3 else "landmarks"
                payload[key] = arr.tolist()

        if not payload:
            raise RuntimeError("Unrecognized FaceMap output format; please check qai_hub_models version.")

        return {"facemap_3dmm": payload}