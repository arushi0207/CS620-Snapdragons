from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from featurehub.llm import llava_onevision


def default_qwen2_prompt() -> str:
    """
    Default visual-only public speaking coach prompt for Qwen2-VL.
    """
    return (
        "You are a public speaking coach. You will see frames from a presentation video.\n\n"
        "Give a concise, visual-only evaluation. Do NOT infer anything about voice, audio, "
        "or verbal content. Focus almost entirely on the speaker’s nonverbal behavior, "
        "NOT the scenery.\n\n"
        "Specifically assess:\n"
        "- posture and stance\n"
        "- facial expression (warmth, authenticity, variation)\n"
        "- eye contact with the camera\n"
        "- gesture variety and control\n"
        "- body movement and use of space\n"
        "- how centered they are in the frame\n"
        "- whether lighting and background help or distract\n\n"
        "Only mention the background or scenery if it clearly helps or hurts the presentation.\n\n"
        "Structure your response as:\n"
        "1) Overall Summary (2–3 sentences).\n"
        "2) Strengths (3–5 bullet points).\n"
        "3) Opportunities for Improvement (3–5 bullet points).\n"
        "4) Actionable Suggestions (5–8 numbered, specific, behavior-focused tips).\n\n"
        "Keep it supportive, direct, and actionable. Keep total length under 300 words."
    )


def main():
    ap = argparse.ArgumentParser(
        description="Run model-based text evaluation on a presentation video."
    )
    ap.add_argument("--video", required=True, help="Path to the input video.")
    ap.add_argument("--out", required=True, help="Directory to write outputs.")
    ap.add_argument(
        "--model",
        default="qwen2-pt",
        help=(
            "Model backend identifier. Examples:\n"
            "  qwen2-pt                    -> Qwen2-VL-2B-Instruct (PyTorch, CPU)\n"
            "  Qwen/Qwen3-VL-2B-Instruct   -> Qwen3-VL (PyTorch)\n"
            "  llava-onevision             -> LLaVA-OneVision\n"
            "  gemini-1.5-pro              -> Gemini backend"
        ),
    )
    ap.add_argument(
        "--prompt", default=None, help="Custom English prompt for evaluation."
    )
    ap.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a file containing the prompt.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (for Qwen / LLaVA / Gemini backends).",
    )
    ap.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Optional frame sampling FPS. If omitted, use all frames (LLaVA).",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=64,
        help="Cap on frames after sampling (default: 64 for safety, LLaVA).",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=64,
        help="Number of frames to sample from the video (Qwen backends).",
    )
    ap.add_argument(
        "--use-video-mode",
        action="store_true",
        help="Use LLaVA video input pathway (may overflow context).",
    )
    ap.add_argument(
        "--output-json",
        default="evaluation_llava.json",
        help="Output JSON filename inside --out.",
    )
    ap.add_argument(
        "--onnx-path",
        default=None,
        help=(
            "Path to the ONNX model directory/file for Qwen3-VL ONNX exports. "
            "Ignored for Qwen2 PyTorch backend."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Figure out model id early so we can pick a backend-specific default prompt
    model_id = args.model
    model_id_lower = model_id.lower()

    # ----- load prompt -----
    prompt: Optional[str] = None
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        # Backend-specific defaults
        if "qwen2" in model_id_lower:
            prompt = default_qwen2_prompt()
        else:
            # reasonable default shared with LLaVA path
            prompt = llava_onevision.default_prompt()

    # ----- dispatch by backend -----

    # 1) Gemini backend: use google-genai client with video + prompt input
    if "gemini" in model_id_lower:
        from featurehub.llm import gemini

        result = gemini.generate_evaluation(
            video_path=args.video,
            prompt=prompt,
            model_id=model_id,
        )

    # 2) Qwen2-VL PyTorch backend (CPU-only as per mentor)
    elif "qwen2" in model_id_lower:
        from featurehub.llm import qwen2_vl_inference

        # You can point model_id here to a local folder if you downloaded it,
        # e.g. model_id = r"C:\Users\Arushi Taneja\Documents\Qualcomm\models\qwen2-pt"
        result = qwen2_vl_inference.generate_evaluation(
            video_path=args.video,
            prompt=prompt,
            model_id=model_id
            if "/" in model_id
            else "Qwen/Qwen2-VL-2B-Instruct",
            max_new_tokens=args.max_new_tokens,
            num_frames=args.num_frames,
            temperature=args.temperature,
        )

    # 3) Qwen3 backends (PyTorch or your custom ONNX export)
    elif "qwen3" in model_id_lower or (
        "qwen" in model_id_lower and "qwen2" not in model_id_lower
    ):
        from featurehub.llm import qwen3_vl_inference

        if not args.onnx_path:
            # PyTorch path
            result = qwen3_vl_inference.generate_evaluation(
                video_path=args.video,
                prompt=prompt,
                model_id=model_id,
                max_new_tokens=args.max_new_tokens,
                num_frames=args.num_frames,
            )
        else:
            # Your custom Qwen3 ONNX export path
            from featurehub.llm import qwen3_vl_inference_onnx

            result = qwen3_vl_inference_onnx.generate_evaluation(
                video_path=args.video,
                onnx_path=args.onnx_path,
                num_frames=args.num_frames,
            )
            out_path_onnx = os.path.join(args.out, "evaluation_qwen3_onnx.json")
            with open(out_path_onnx, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Wrote Qwen3 ONNX evaluation to {out_path_onnx}")

    # 4) LLaVA-OneVision backend
    elif "llava-onevision" in model_id_lower or "llava" in model_id_lower:
        result = llava_onevision.generate_evaluation(
            video_path=args.video,
            prompt=prompt,
            model_id=model_id,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_video_mode=args.use_video_mode,
        )

    else:
        raise RuntimeError(
            f"Unknown model backend for --model={model_id}. "
            "Expected something containing 'gemini', 'qwen2', 'qwen3', or 'llava-onevision'."
        )

    # ----- common output -----
    out_path = os.path.join(args.out, args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote evaluation to {out_path}")


if __name__ == "__main__":
    main()
