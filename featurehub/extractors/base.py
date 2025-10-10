from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseFeatureExtractor(ABC):
    """
    Base class for feature extractors.
    input: BGR image as numpy array (H, W, 3)
    output: dictionary of features
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.setup()


    def setup(self):
        pass


    @abstractmethod
    def extract(self, frame_bgr: np.ndarray, *, timestamp: float | None = None) -> Dict[str, Any]:
        raise NotImplementedError


    @staticmethod
    def bgr2rgb(img: np.ndarray) -> np.ndarray:
        return img[..., ::-1]


    @staticmethod
    def img_size(img: np.ndarray) -> Tuple[int, int]:
        h, w = img.shape[:2]
        return w, h