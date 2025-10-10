from __future__ import annotations
import argparse
import os
import cv2
from featurehub.pipeline import Pipeline
from featurehub.sources.video_reader import frames_from_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default="outputs/live")
    ap.add_argument("--extractors", default="facemap_3dmm") # "hrnet_pose,facemap_3dmm"
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    names = [x.strip() for x in args.extractors.split(",") if x.strip()]
    pipe = Pipeline(names, out_dir=args.out, visualize=True)

    saving = True
    print("Press 's' to toggle saving, 'q' to quit.")

    for idx, ts, frame in frames_from_video(args.cam, target_fps=args.fps):
        vis = pipe.process_frame(idx, ts, frame)
        cv2.putText(vis, f"idx={idx} t={ts:.2f}s saving={saving}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imshow("SnapX-FeatureFusion", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            saving = not saving
            if not saving:
                # Pause writing by toggling the writers into a drop mode.
                pipe.writers.write_jsonl = False
                pipe.writers.write_npz = False
            else:
                pipe.writers.write_jsonl = True
                pipe.writers.write_npz = True

    pipe.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
