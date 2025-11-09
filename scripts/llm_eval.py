from __future__ import annotations

import argparse
import json
import os
from typing import Optional
import torch

from featurehub.llm.llava_onevision import (
    generate_evaluation,
    DEFAULT_MODEL_ID,
    default_prompt,
)


def main():
    ap = argparse.ArgumentParser(description="Run LLaVA-OneVision text evaluation on a presentation video.")
    ap.add_argument("--video", required=True, help="Path to the input video.")
    ap.add_argument("--out", required=True, help="Directory to write outputs.")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    ap.add_argument("--prompt", default=None, help="Custom English prompt for evaluation.")
    ap.add_argument("--prompt-file", default=None, help="Path to a file containing the prompt.")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate.")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    ap.add_argument("--sample-fps", type=float, default=None, help="Optional frame sampling FPS. If omitted, use all frames.")
    ap.add_argument("--max-frames", type=int, default=64, help="Cap on frames after sampling (default: 64 for safety).")
    ap.add_argument("--use-video-mode", action="store_true", help="Use video input pathway (may overflow context).")
    ap.add_argument("--output-json", default="evaluation_llava.json", help="Output JSON filename inside --out.")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                    help="Quantization mode: 4bit or 8bit (requires bitsandbytes).")
    ap.add_argument("--compute-dtype", choices=["float16", "bfloat16"], default="float16",
                    help="Compute dtype used by quantized layers.")
    ap.add_argument("--device-map", default="auto",
                    help='Device map for loading (e.g., "auto", "cuda:0").')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    prompt: Optional[str] = None
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = default_prompt()

    compute_dtype = torch.float16 if args.compute_dtype == "float16" else torch.bfloat16

    result = generate_evaluation(
        video_path=args.video,
        prompt=prompt,
        model_id=args.model,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_video_mode=args.use_video_mode,
        quant=args.quant,
        compute_dtype=compute_dtype,
        device_map=args.device_map,
    )

    out_path = os.path.join(args.out, args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote evaluation to {out_path}")


if __name__ == "__main__":
    main()
