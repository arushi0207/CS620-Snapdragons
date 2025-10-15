from __future__ import annotations
import cv2
import numpy as np


def draw_landmarks(img: np.ndarray, pts: list[list[float]]):
    for p in pts:
        if len(p) >= 2:
            x, y = int(p[0]), int(p[1])
            cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
