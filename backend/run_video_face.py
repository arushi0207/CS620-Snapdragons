# run_video_face.py
# Minimal video-to-video runner for Qualcomm AI Hub MediaPipe Face
# Works entirely on host: reads MP4, annotates each frame, writes/plays output.

import argparse
import cv2
import numpy as np

from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from qai_hub_models.models.mediapipe_face.model import MediaPipeFace


def build_app() -> MediaPipeFaceApp:
    torch_model = MediaPipeFace.from_pretrained()
    detector = torch_model.face_detector
    landmark_detector = torch_model.face_landmark_detector
    anchors = detector.anchors
    app = MediaPipeFaceApp(
        detector,
        landmark_detector,
        anchors,
        detector.get_input_spec(),
        landmark_detector.get_input_spec(),
    )
    return app


def main():
    parser = argparse.ArgumentParser(description="Run MediaPipe Face on a video file.")
    parser.add_argument("--video", required=True, help="Path to input video (e.g., input.mp4)")
    parser.add_argument("--out", default="", help="Optional path to save annotated MP4 (e.g., out.mp4)")
    parser.add_argument("--display", action="store_true", help="Show a preview window while processing")
    parser.add_argument("--max-frames", type=int, default=0, help="Process at most N frames (0 = all)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer for: {args.out}")

    app = build_app()
    window_name = "QAIHM MediaPipe Face (Video)"

    frame_count = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            annotated_rgb = app.predict_landmarks_from_image(frame_rgb)[0]
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            if writer:
                writer.write(annotated_bgr)

            if args.display:
                cv2.imshow(window_name, annotated_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if args.out:
        print(f"Saved annotated video to: {args.out}")


if __name__ == "__main__":
    main()
