from __future__ import annotations
from typing import List, Dict, Any

from .registry import get_extractor
from .visualize import draw_landmarks
from .io.writers import FeatureWriters
from . import extractors


class Pipeline:
    def __init__(self, extractor_names: List[str], out_dir: str, *, visualize: bool = False):
        self.extractors = [get_extractor(n)() for n in extractor_names]
        self.out_dir = out_dir
        self.visualize = visualize
        self.writers = FeatureWriters(out_dir, write_jsonl=True, write_npz=True)
        self._records: List[Dict[str, Any]] = []

    def process_frame(self, frame_idx: int, ts: float, frame_bgr: np.ndarray) -> np.ndarray:
        H, W = frame_bgr.shape[:2]
        rec: Dict[str, Any] = {"frame_idx": frame_idx, "timestamp": ts, "image_size": [W, H]}
        vis = frame_bgr.copy() if self.visualize else None

        for ext in self.extractors:
            out = ext.extract(frame_bgr, timestamp=ts)
            rec.update(out)

        # Overlay visualizations.
        if self.visualize and vis is not None:
            fm = rec.get("facemap_3dmm", {})
            if fm.get("landmarks3d"):
                # Draw only the projected (x, y) coordinates.
                draw_landmarks(vis, [[x, y] for x, y, _ in fm["landmarks3d"]])
            elif fm.get("landmarks"):
                draw_landmarks(vis, fm["landmarks"])

        self.writers.write_frame(rec)
        self._records.append(rec)
        return vis if vis is not None else frame_bgr

    def close(self):
        self.writers.close()

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self._records
