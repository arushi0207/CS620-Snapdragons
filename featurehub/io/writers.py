from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import numpy as np


class FeatureWriters:
    def __init__(self, out_dir: str, write_jsonl: bool = True, write_npz: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.write_jsonl = write_jsonl
        self.write_npz = write_npz
        self.jsonl_fp = open(os.path.join(out_dir, "features.jsonl"), "w", encoding="utf-8") if write_jsonl else None
        # 聚合到内存，最后一次性存 .npz
        self._frames: List[int] = []
        self._timestamps: List[float] = []
        self._hrnet_all: List[Any] = []
        self._facemap_all: List[Any] = []

    def write_frame(self, rec: Dict[str, Any]):
        if self.jsonl_fp:
            self.jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._frames.append(rec.get("frame_idx", -1))
        self._timestamps.append(rec.get("timestamp", 0.0))
        self._hrnet_all.append(rec.get("hrnet_pose", {}).get("keypoints"))
        fm = rec.get("facemap_3dmm", {})
        self._facemap_all.append({k: v for k, v in fm.items()})

    def close(self):
        if self.jsonl_fp:
            self.jsonl_fp.close()
        if self.write_npz:
            np.savez_compressed(
                os.path.join(self.out_dir, "features.npz"),
                frames=np.array(self._frames),
                timestamps=np.array(self._timestamps, dtype=np.float32),
                hrnet_keypoints=np.array(self._hrnet_all, dtype=object),
                facemap=np.array(self._facemap_all, dtype=object),
            )