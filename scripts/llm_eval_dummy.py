from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dummy evaluation API for frontend integration (no real LLM)."
    )

    # Keep CLI compatible with scripts/llm_eval.py so frontend/backend can switch easily.
    ap.add_argument("--video", required=True, help="Path to the input video.")
    ap.add_argument("--out", required=True, help="Directory to write outputs.")
    ap.add_argument(
        "--model",
        default="dummy-model",
        help="(unused) model id, kept for compatibility.",
    )
    ap.add_argument(
        "--prompt",
        default=None,
        help="(unused) custom English prompt for evaluation.",
    )
    ap.add_argument(
        "--prompt-file",
        default=None,
        help="(unused) path to a file containing the prompt.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="(unused) max new tokens to generate.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="(unused) sampling temperature.",
    )
    ap.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="(unused) frame sampling FPS.",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=64,
        help="(unused) cap on frames after sampling.",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=64,
        help="(unused) number of frames to sample from the video.",
    )
    ap.add_argument(
        "--use-video-mode",
        action="store_true",
        help="(unused) use video input pathway.",
    )
    ap.add_argument(
        "--output-json",
        default="evaluation_llava.json",
        help="Output JSON filename inside --out.",
    )

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Construct a deterministic dummy evaluation result for frontend integration.
    result = {
        "video_path": args.video,
        "model": "dummy-evaluator",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "overall_score": 0.85,
        "summary": "This is a dummy evaluation for frontend integration (no real LLM is used).",
        "details": {
            "clarity": 0.9,
            "structure": 0.8,
            "engagement": 0.75,
            "timing": 0.88,
        },
        "notes": [
            "This result is a fixed dummy output for frontend-backend integration testing only.",
            "When a real model is available, you can switch back to scripts/llm_eval.py.",
        ],
    }

    out_path = os.path.join(args.out, args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[DUMMY] Wrote evaluation to {out_path}")


if __name__ == "__main__":
    main()
