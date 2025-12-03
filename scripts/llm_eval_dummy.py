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
        "summary": "1) Overall Summary: The video features a person standing in front of a plain white wall, speaking directly to the camera. The individual maintains a neutral posture, with consistent eye contact and minimal movement, suggesting a focused and deliberate delivery. The lighting is even, and the background is unobtrusive, allowing the viewer to concentrate on the speaker’s message.\n\n2) Strengths:\n- The speaker maintains steady eye contact, which helps establish connection with the audience.\n- The framing is centered and stable, ensuring the viewer’s attention remains on the speaker.\n- The background is minimalistic, removing distractions and enhancing clarity.\n- The speaker’s facial expressions are natural and consistent, indicating confidence and composure.\n\n3) Opportunities for Improvement:\n- The speaker could use more varied gestures to engage the audience and emphasize key points.\n- There is limited body movement, which may reduce dynamic engagement.\n- The camera angle could be slightly adjusted to include more of the speaker’s upper body for better visibility.\n\n4) Actionable Suggestions:\n1. Add subtle hand gestures to reinforce key points and keep the audience engaged.\n2. Increase body movement to enhance energy and expressiveness.\n3. Consider using a slightly wider camera angle to include more of the upper body.\n4. Use natural facial expressions to convey emotion and maintain audience interest.\n5. Experiment with different camera angles to improve visual engagement.\n6. Keep the background clean and distraction-free.\n7. Maintain consistent eye contact and avoid looking away.\n8. Practice speaking with confidence to ensure natural delivery.",
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
