from __future__ import annotations
import importlib
from typing import Any, Dict
import numpy as np
import cv2


from .base import BaseFeatureExtractor
from ..registry import register_extractor




@register_extractor("hrnet_pose")
class HRNetPoseExtractor(BaseFeatureExtractor):
    """Wrapper around HRNetPose (qai_hub_models).

    Returns: {"hrnet_pose": {"keypoints": [[x, y, score], ...]}}
    """

    def setup(self):
        # Delay the import to produce a clearer error when mmpose is unavailable.
        try:
            mod = importlib.import_module("qai_hub_models.models.hrnet_pose")
        except Exception as e:
            raise RuntimeError(
                "Failed to import 'qai_hub_models.models.hrnet_pose'. "
                "Ensure qai_hub_models is installed and MMPose/MMCV wheels match your torch version."
            ) from e


        # App class usually exists; keep this compatible if the structure changes.
        App = getattr(mod, "App", None)
        if App is None:
            # Some versions may place it in a submodule.
            mod_app = importlib.import_module("qai_hub_models.models.hrnet_pose.demo")
            App = getattr(mod_app, "App")
        self.app = App()


    @staticmethod
    def _heatmaps_to_keypoints(heat: np.ndarray, img_wh: tuple[int, int]):
        """Convert heatmaps [C, H, W] into keypoints and rescale to the original image size."""
        if heat.ndim == 4:
            heat = heat[0]
        C, H, W = heat.shape
        Wimg, Himg = img_wh
        kpts = []
        for c in range(C):
            idx = np.argmax(heat[c])
            y, x = np.unravel_index(idx, (H, W))
            score = float(heat[c, y, x])
            kpts.append([float(x) * Wimg / W, float(y) * Himg / H, score])
        return kpts


    def extract(self, frame_bgr: np.ndarray, *, timestamp: float | None = None) -> Dict[str, Any]:
        img_rgb = self.bgr2rgb(frame_bgr)
        Wimg, Himg = self.img_size(frame_bgr)


        # Support different App APIs: predict / run_image / __call__.
        for fn_name in ("predict", "run_image", "__call__"):
            if hasattr(self.app, fn_name):
                out = getattr(self.app, fn_name)(img_rgb)
                break
            else:
                raise RuntimeError("HRNetPose App has no callable predict/run_image.")


        # Common return patterns: either keypoints or heatmaps.
        kpts = None
        if isinstance(out, dict):
            if "keypoints" in out:
                kpts = out["keypoints"]
            elif "poses" in out and isinstance(out["poses"], dict) and "keypoints" in out["poses"]:
                kpts = out["poses"]["keypoints"]
            elif "heatmaps" in out:
                kpts = self._heatmaps_to_keypoints(np.asarray(out["heatmaps"]), (Wimg, Himg))
            elif isinstance(out, np.ndarray):
                # Raw heatmaps.
                if out.ndim in (3, 4):
                    kpts = self._heatmaps_to_keypoints(out, (Wimg, Himg))


        if kpts is None:
            # Last attempt: handle list[dict] outputs.
            if isinstance(out, list) and len(out) and isinstance(out[0], dict) and "keypoints" in out[0]:
                kpts = out[0]["keypoints"]


        if kpts is None:
            raise RuntimeError("Unrecognized HRNetPose output format; please check qai_hub_models version.")


        return {"hrnet_pose": {"keypoints": kpts}}
