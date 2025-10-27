from __future__ import annotations
import json
import os
from typing import Any, Dict, Iterable, List, Optional
import numpy as np


class FeatureWriters:
    def __init__(
        self,
        out_dir: str,
        *,
        feature_keys: Optional[Iterable[str]] = None,
        write_jsonl: bool = True,
        write_npz: bool = True,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.write_jsonl = write_jsonl
        self.write_npz = write_npz
        self.jsonl_fp = open(os.path.join(out_dir, "features.jsonl"), "w", encoding="utf-8") if write_jsonl else None
        # Accumulate in memory and flush to .npz at the end.
        self._frames: List[int] = []
        self._timestamps: List[float] = []
        self._feature_data: Dict[str, List[Any]] = {
            key: [] for key in (feature_keys or [])
        }

    def write_frame(self, rec: Dict[str, Any]):
        if self.jsonl_fp:
            self.jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._frames.append(rec.get("frame_idx", -1))
        self._timestamps.append(rec.get("timestamp", 0.0))

        meta_keys = {"frame_idx", "timestamp", "image_size"}
        feature_keys_in_rec = [
            key for key in rec.keys() if key not in meta_keys
        ]

        current_index = len(self._frames) - 1

        for key in feature_keys_in_rec:
            if key not in self._feature_data:
                self._feature_data[key] = [None] * current_index

        for key in list(self._feature_data.keys()):
            self._feature_data[key].append(rec.get(key))

    def close(self):
        if self.jsonl_fp:
            self.jsonl_fp.close()
        if self.write_npz:
            npz_payload: Dict[str, Any] = {
                "frames": np.array(self._frames),
                "timestamps": np.array(self._timestamps, dtype=np.float32),
            }

            for key, values in self._feature_data.items():
                npz_payload[key] = np.array(values, dtype=object)

            if "facemap_3dmm" in self._feature_data and "facemap" not in npz_payload:
                npz_payload["facemap"] = np.array(
                    self._feature_data["facemap_3dmm"], dtype=object
                )

            np.savez_compressed(
                os.path.join(self.out_dir, "features.npz"),
                **npz_payload,
            )
