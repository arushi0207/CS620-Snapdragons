import argparse
from pathlib import Path

import numpy as np
import cv2  # pip install opencv-python if you don't have it

from featurehub.fastcv_bridge import fastcv_detect_faces


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run FastCV-based face detection on a video and extract the best face."
    )
    ap.add_argument(
        "--video",
        required=True,
        help="Path to input video file.",
    )
    ap.add_argument(
        "--out-frame",
        default="best_face_frame.jpg",
        help="Path to save the frame with the best detected face.",
    )
    ap.add_argument(
        "--out-video",
        default=None,
        help="Optional path to write an annotated video with face boxes.",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Process every Nth frame (default: 2) to go faster.",
    )
    return ap.parse_args()


def bgr_to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) frame to grayscale uint8."""
    # frame_bgr is HxWx3 (B,G,R)
    frame_rgb = frame_bgr[..., ::-1]  # BGR -> RGB
    gray = np.dot(frame_rgb[..., :3], [0.299, 0.587, 0.114]).astype(
        np.uint8, copy=False
    )
    return gray


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Optional: annotated output video
    writer = None
    if args.out_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))

    best_face = None  # (area, frame_index, bbox, frame_bgr)
    frame_idx = 0
    processed = 0
    detected_frames = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Only process every Nth frame to save time
        if frame_idx % args.stride != 0:
            if writer is not None:
                writer.write(frame_bgr)
            frame_idx += 1
            continue

        processed += 1

        gray = bgr_to_gray(frame_bgr)
        rects = fastcv_detect_faces(gray, max_faces=1)

        if rects:
            detected_frames += 1
            r = rects[0]
            x, y, w, h = int(r.x), int(r.y), int(r.width), int(r.height)
            area = w * h

            # Draw box for visualization
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Track the "best" (largest) face across the video
            if best_face is None or area > best_face[0]:
                best_face = (area, frame_idx, (x, y, w, h), frame_bgr.copy())
        else:
            # No detection on this frame
            pass

        if writer is not None:
            writer.write(frame_bgr)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"Processed frames (stride={args.stride}): {processed}")
    print(f"Frames with detection: {detected_frames}")

    if best_face is None:
        print("No face detected in any frame.")
        return

    _, best_frame_idx, (bx, by, bw, bh), best_frame = best_face
    print(
        f"Best face at frame {best_frame_idx}: "
        f"x={bx}, y={by}, w={bw}, h={bh}"
    )

    # Save the best frame as an image
    out_frame_path = Path(args.out_frame)
    cv2.imwrite(str(out_frame_path), best_frame)
    print(f"Saved best face frame to: {out_frame_path}")


if __name__ == "__main__":
    main()
