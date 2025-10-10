from __future__ import annotations
import argparse
import os
import yaml
import cv2

from featurehub.pipeline import Pipeline
from featurehub.sources.video_reader import frames_from_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=float, default=None, help="target FPS for sampling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extractors", default="hrnet_pose,facemap_3dmm")
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    names = [x.strip() for x in args.extractors.split(",") if x.strip()]
    pipe = Pipeline(names, out_dir=args.out, visualize=args.visualize)

    for idx, ts, frame in frames_from_video(args.video, target_fps=args.fps):
        vis = pipe.process_frame(idx, ts, frame)
        if args.visualize:
            cv2.imshow("features", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    pipe.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()