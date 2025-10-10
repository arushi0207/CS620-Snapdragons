from __future__ import annotations
import cv2
import math
from typing import Iterator, Tuple


def frames_from_video(src: str | int, target_fps: float | None = None) -> Iterator[Tuple[int, float, "np.ndarray"]]:
    """逐帧读取视频或摄像头（src=0）。

    返回 (frame_idx, timestamp_sec, frame_bgr)。如果 target_fps 指定，则按近似倍数抽帧。
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video/camera: {src}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not src_fps or math.isnan(src_fps):
        src_fps = 30.0

    stride = 1
    if target_fps and target_fps > 0:
        stride = max(1, int(round(src_fps / target_fps)))

    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if stride > 1 and (idx % stride) != 0:
            continue
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        yield idx, ts, frame

    cap.release()