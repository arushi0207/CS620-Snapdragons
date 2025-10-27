from __future__ import annotations
import argparse
import os
from typing import Optional

import cv2

from featurehub.pipeline import Pipeline
from featurehub.sources.video_reader import frames_from_video
from featurehub.scoring.dummy import DummyScoreModel


def _infer_writer_fps(video_path: str, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return fallback
    fps = cap.get(cv2.CAP_PROP_FPS) or fallback
    cap.release()
    return fps if fps > 0 else fallback


def main():
    ap = argparse.ArgumentParser(description="Run configured feature extractors on a video.")
    ap.add_argument("--video", required=True, help="Path to the input video.")
    ap.add_argument("--out", required=True, help="Directory to store extracted features.")
    ap.add_argument("--output-video", default=None, help="Optional path to write an annotated MP4.")
    ap.add_argument("--fps", type=float, default=None, help="Target FPS for sampling frames.")
    ap.add_argument(
        "--extractors",
        nargs="+",
        default=["facemap_3dmm", "mediapipe_face"],
        help="Feature extractors to run (e.g., facemap_3dmm mediapipe_face).",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    should_visualize = args.output_video is not None
    pipeline = Pipeline(args.extractors, out_dir=args.out, visualize=should_visualize)

    video_writer: Optional[cv2.VideoWriter] = None
    writer_fps = args.fps or _infer_writer_fps(args.video)

    try:
        for idx, ts, frame in frames_from_video(args.video, target_fps=args.fps):
            vis = pipeline.process_frame(idx, ts, frame)

            if args.output_video:
                if video_writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(args.output_video, fourcc, writer_fps, (w, h))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Failed to open VideoWriter for {args.output_video}")
                video_writer.write(vis)
    finally:
        pipeline.close()
        if video_writer:
            video_writer.release()

    scorer = DummyScoreModel()
    score = scorer.predict(pipeline.records)
    print(f"Predicted speech score: {score}")


if __name__ == "__main__":
    main()
