from __future__ import annotations

import numpy as np
from typing import Any, Dict, List

import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ---------------------------------------------------------------------
# Global singletons so we only load the model ONCE per process
# ---------------------------------------------------------------------
_DEVICE = torch.device("cpu")  # ✅ force CPU for Snapdragon requirement
_PROCESSOR: AutoProcessor | None = None
_MODEL: Qwen2VLForConditionalGeneration | None = None


def _get_qwen2_model(model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """
    Lazy-load Qwen2-VL processor + model on CPU.
    """
    global _PROCESSOR, _MODEL

    if _PROCESSOR is None or _MODEL is None:
        print(f"[Qwen2-VL] Loading model {model_id} on CPU...")
        _PROCESSOR = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        _MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # CPU-friendly
        )
        _MODEL.to(_DEVICE)
        _MODEL.eval()
        print("[Qwen2-VL] Model loaded.")

    return _PROCESSOR, _MODEL


# ---------------------------------------------------------------------
# Video → frames
# ---------------------------------------------------------------------
def _sample_frames(
    video_path: str,
    num_frames: int,
) -> List[Image.Image]:
    """
    Uniformly sample up to num_frames from the video and return PIL Images.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"No frames found in video: {video_path}")

    num = min(num_frames, total)
    indices = np.linspace(0, total - 1, num=num, dtype=int)

    frames: List[Image.Image] = []
    for idx in indices:
        frame = vr[idx]  # decord NDArray (H,W,3), uint8
        img = Image.fromarray(frame.asnumpy())
        frames.append(img)

    return frames


# ---------------------------------------------------------------------
# Run Qwen2-VL on a single frame
# ---------------------------------------------------------------------
def _run_on_single_frame(
    image: Image.Image,
    prompt: str,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    """
    Build a multimodal conversation (image + text) and run Qwen2-VL.generate().
    """
    processor, model = _get_qwen2_model(model_id)

    # Qwen2-VL uses a chat template + content blocks
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )

    batch = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )
    batch = {k: v.to(_DEVICE) for k, v in batch.items()}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**batch, **gen_kwargs)

    # Drop the prompt tokens, keep only newly generated text
    input_len = batch["input_ids"].shape[-1]
    generated = outputs[:, input_len:]

    decoded = processor.batch_decode(
        generated, skip_special_tokens=True
    )

    return decoded[0].strip() if decoded else ""


# ---------------------------------------------------------------------
# Public API used by scripts/llm_eval.py
# ---------------------------------------------------------------------
def generate_evaluation(
    video_path: str,
    prompt: str,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    max_new_tokens: int = 256,
    num_frames: int = 16,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    High-level entrypoint for Qwen2-VL (PyTorch, CPU).

    - Samples frames from the video
    - Uses the middle frame as representative
    - Returns a JSON-serializable dict
    """
    frames = _sample_frames(video_path, num_frames=num_frames)
    if not frames:
        raise RuntimeError(f"No frames sampled from video: {video_path}")

    frame_index = len(frames) // 2
    frame = frames[frame_index]

    response_text = _run_on_single_frame(
        image=frame,
        prompt=prompt,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return {
        "backend": "qwen2-pt",
        "model_id": model_id,
        "device": str(_DEVICE),
        "video_path": video_path,
        "prompt": prompt,
        "num_frames_sampled": len(frames),
        "frame_used": frame_index,
        "response": response_text,
    }
