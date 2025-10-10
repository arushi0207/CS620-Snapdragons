from __future__ import annotations
import cv2
import numpy as np

# COCO 17 点连接（示意，可按需调整）
COCO_PAIRS = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (1, 2), (2, 3), (3, 4)
]


def draw_pose(img: np.ndarray, keypoints: list[list[float]]):
    for i, (x, y, score) in enumerate(keypoints):
        if score is not None and score < 0.1:
            continue
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    for a, b in COCO_PAIRS:
        if a < len(keypoints) and b < len(keypoints):
            xa, ya, sa = keypoints[a]
            xb, yb, sb = keypoints[b]
            if sa is not None and sa < 0.1:
                continue
            if sb is not None and sb < 0.1:
                continue
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)


def draw_landmarks(img: np.ndarray, pts: list[list[float]]):
    for p in pts:
        if len(p) >= 2:
            x, y = int(p[0]), int(p[1])
            cv2.circle(img, (x, y), 2, (0, 255, 255), -1)