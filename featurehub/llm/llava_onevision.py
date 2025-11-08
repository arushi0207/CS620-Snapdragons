from __future__ import annotations

import os
from typing import List, Optional, Dict, Any
import inspect

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


DEFAULT_MODEL_ID = "lmms-lab/LLaVA-OneVision-1.5-4B-stage0"


def default_prompt() -> str:
    return (
        "You are a public speaking coach. Watch the following presentation video and provide a concise, visual-only evaluation. "
        "Do not infer anything about voice quality, audio, or verbal content. Assess: posture and stance, facial expression, "
        "eye contact with the camera, gesture variety and control, body movement, use of space, camera framing, lighting, "
        "background distractions, and use of visual aids (if any).\n\n"
        "Structure your response as:\n"
        "1) Overall Summary (2–3 sentences).\n"
        "2) Strengths (3–5 bullet points).\n"
        "3) Opportunities for Improvement (3–5 bullet points).\n"
        "4) Actionable Suggestions (5–8 numbered, specific, behavior-focused tips).\n\n"
        "Keep it supportive, direct, and actionable. Keep total length under 350 words."
    )


def load_llava(model_id: str = DEFAULT_MODEL_ID):
    """Load LLaVA-OneVision model and processor."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def read_all_frames_as_pil(video_path: str) -> List[Image.Image]:
    """Read the full video and return a list of PIL images (RGB)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: List[Image.Image] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def sample_video_frames_as_pil(
    video_path: str,
    sample_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> List[Image.Image]:
    """Optionally sample frames; if both are None, return all frames."""
    if sample_fps is None and max_frames is None:
        return read_all_frames_as_pil(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not native_fps or torch.isnan(torch.tensor(native_fps)):
        native_fps = 30.0

    stride = 1
    if sample_fps and sample_fps > 0:
        stride = max(1, int(round(native_fps / sample_fps)))

    frames: List[Image.Image] = []
    idx = -1
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            idx += 1
            if stride > 1 and (idx % stride) != 0:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("Sampling produced zero frames; adjust sample_fps/max_frames.")
    return frames


def build_messages_for_video(video_ref: str, prompt: str) -> list[dict]:
    """Build chat-format messages including a single video and a user prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_ref},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_messages_for_images(num_images: int, prompt: str) -> list[dict]:
    """Build chat messages for multiple images followed by a text prompt."""
    contents = ([{"type": "image", "image": f"frame_{i:04d}"} for i in range(num_images)]
                + [{"type": "text", "text": prompt}])
    return [{"role": "user", "content": contents}]


def generate_evaluation(
    video_path: str,
    prompt: Optional[str] = None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    sample_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    use_video_mode: bool = False,
    safety_frames_cap: int = 64,
) -> Dict[str, Any]:
    """Run LLaVA-OV on a video with a prompt and return a JSON-able dict."""
    if prompt is None:
        prompt = default_prompt()

    # Load model & processor
    model, processor = load_llava(model_id)

    # Frames (full video by default) — with safety caps to avoid token overflow
    frames: List[Image.Image]
    if sample_fps is None and max_frames is None:
        frames = read_all_frames_as_pil(video_path)
    else:
        frames = sample_video_frames_as_pil(video_path, sample_fps=sample_fps, max_frames=max_frames)

    # Decide mode: prioritize robust image pathway unless explicitly using video mode
    input_mode = "image"
    img_subset: List[Image.Image]
    # Apply a safety cap if user didn't specify max_frames
    if max_frames is None and len(frames) > safety_frames_cap:
        img_subset = frames[:safety_frames_cap]
        clipped = True
    else:
        img_subset = frames if max_frames is None else frames[:max_frames]
        clipped = max_frames is not None and len(frames) > max_frames

    if use_video_mode:
        # Try the official video pathway; may produce long sequences — caller beware
        try:
            messages = build_messages_for_video(video_path, prompt)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs = processor(
                text=[text],
                images=None,
                videos=[img_subset],  # still cap frames to avoid overflow
                padding=True,
                return_tensors="pt",
            )
            input_mode = "video"
        except Exception:
            # Fall through to image pathway
            input_mode = "image"

    if input_mode == "image":
        messages = build_messages_for_images(len(img_subset), prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_inputs = processor(
            text=[text],
            images=img_subset,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

    # Place on device: LLaVA-OV examples push to CUDA explicitly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        batch_inputs = batch_inputs.to(device)
    except Exception:
        # Best-effort placement; if sharded, model.generate will handle routing
        pass

    # Filter unsupported kwargs to avoid generation validation errors (e.g., 'second_per_grid_ts')
    # Build model kwargs respecting the model's forward signature
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_kw:
            filtered_inputs = dict(batch_inputs)
        else:
            allowed = set(params.keys()) | {"input_ids", "attention_mask"}
            filtered_inputs = {k: v for k, v in batch_inputs.items() if k in allowed}
            # Drop known problematic video timestamp keys if present
            for bad in ["second_per_grid_ts", "seconds_per_grid_ts", "video_timestamps"]:
                filtered_inputs.pop(bad, None)
    except Exception:
        filtered_inputs = dict(batch_inputs)

    # Generate
    generated_ids = model.generate(
        **filtered_inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0] if output_text else ""

    result = {
        "model": model_id,
        "language": "en",
        "video": os.path.basename(video_path),
        "video_path": os.path.abspath(video_path),
        "num_frames": len(frames),
        "sampling": {
            "sample_fps": sample_fps,
            "max_frames": max_frames,
        },
        "prompt": prompt,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "response": response,
    }
    result["sampling"]["mode"] = input_mode
    result["sampling"]["frames_used"] = len(img_subset)
    result["sampling"]["clipped"] = bool(clipped)
    return result
