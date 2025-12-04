from __future__ import annotations

import os
import time
import inspect
from typing import Any, Dict, List, Optional

import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

DEFAULT_MODEL_ID = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"

def default_prompt() -> str:
    """Return the default English evaluation prompt.

    Keep it self-contained so callers can use it directly or override.
    """
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


# ---------------------------
# Loading utilities
# ---------------------------

def load_llava(model_id: str = DEFAULT_MODEL_ID):
    """Load LLaVA-OneVision model and processor.

    Environment overrides (optional):
      - LLAVA_DTYPE: {auto,float16,fp16,bfloat16,bf16} (default: auto)
      - LLAVA_DEVICE_MAP: e.g. {auto,cuda:0,cpu,none} (default: auto)
      - LLAVA_FORCE_CUDA: '1' to force model.to('cuda') if available
      - LLAVA_PROCESSOR_FAST: '1' use fast processor, '0' slow
    """
    # Select dtype
    dtype_env = os.environ.get("LLAVA_DTYPE", "auto").lower()
    if dtype_env in ("float16", "fp16", "half"):
        dtype_kw: Any = torch.float16
    elif dtype_env in ("bfloat16", "bf16"):
        dtype_kw = torch.bfloat16
    else:
        dtype_kw = "auto"

    device_map_env = os.environ.get("LLAVA_DEVICE_MAP", "cuda:0")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype_kw, device_map=device_map_env, trust_remote_code=True
    )
    try:
        model.eval()
    except Exception:
        pass

    # Optional best-effort hard move to CUDA
    if os.environ.get("LLAVA_FORCE_CUDA", "0").lower() in ("1", "true") and torch.cuda.is_available():
        try:
            model.to("cuda")
        except Exception:
            pass

    # Processor (optionally force slow/fast)
    proc_fast_env = os.environ.get("LLAVA_PROCESSOR_FAST")
    if proc_fast_env is not None:
        use_fast = proc_fast_env.strip().lower() in ("1", "true")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=use_fast)
    else:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    return model, processor


# ---------------------------
# Video utilities
# ---------------------------

def read_video_frames(
    video_path: str,
    sample_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    fallback_fps: float = 30.0,
) -> List[Image.Image]:
    """Read frames from a video and return a list of RGB PIL images.

    - If both `sample_fps` and `max_frames` are None, read the entire video.
    - If `sample_fps` is set, stride frames approximately to match the rate.
    - Always stop once `max_frames` is reached (if provided).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    try:
        native_fps = float(native_fps)
    except Exception:
        native_fps = 0.0
    if native_fps <= 0:
        native_fps = fallback_fps

    stride = 1
    if sample_fps and sample_fps > 0:
        stride = max(1, int(round(native_fps / float(sample_fps))))

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
        raise RuntimeError("No frames decoded; adjust sample_fps/max_frames.")
    return frames


# ---------------------------
# Chat/message + processor inputs
# ---------------------------

def _build_messages(mode: str, prompt: str, *, num_images: int | None = None, video_ref: str | None = None) -> list[dict]:
    """Build chat-format messages for image or video mode.

    mode: 'image' or 'video'. When 'image', provide `num_images`; when 'video', provide `video_ref`.
    """
    if mode == "video":
        assert video_ref is not None
        return [{"role": "user", "content": [{"type": "video", "video": video_ref}, {"type": "text", "text": prompt}]}]
    # image mode
    assert num_images is not None
    contents = ([{"type": "image", "image": f"frame_{i:04d}"} for i in range(num_images)] + [{"type": "text", "text": prompt}])
    return [{"role": "user", "content": contents}]


def _prepare_inputs(
    processor,
    mode: str,
    frames: List[Image.Image],
    video_path: str,
    prompt: str,
):
    """Create the model-ready batch inputs using the HF processor.

    For image mode, pass `images=frames`.
    For video mode, pass `videos=[frames]` (LLaVA-OV expects a list of frame lists).
    """
    if mode == "video":
        messages = _build_messages("video", prompt, video_ref=video_path)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=None, videos=[frames], padding=True, return_tensors="pt")
    # image mode
    messages = _build_messages("image", prompt, num_images=len(frames))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=frames, videos=None, padding=True, return_tensors="pt")


def _to_device(batch_inputs, device: str):
    """Move batch inputs to device when possible (best-effort)."""
    try:
        return batch_inputs.to(device)
    except Exception:
        return batch_inputs


def _filter_model_inputs(model, src: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unexpected keys so generate() doesn't complain on some models.

    We respect the model.forward signature unless it has **kwargs, and we drop a
    few known video timestamp keys that can cause validation errors.
    """
    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_kw:
            out = dict(src)
        else:
            allowed = set(params.keys()) | {"input_ids", "attention_mask"}
            out = {k: v for k, v in src.items() if k in allowed}
        for bad in ("second_per_grid_ts", "seconds_per_grid_ts", "video_timestamps"):
            out.pop(bad, None)
        return out
    except Exception:
        return dict(src)


def _maybe_warmup(model, filtered_inputs: Dict[str, Any]) -> Optional[float]:
    """Optional tiny warmup for kernel compilation/caches.

    Enabled only when LLAVA_WARMUP in {'1','true'}; returns warmup seconds or None.
    """
    if os.environ.get("LLAVA_WARMUP", "0").lower() not in ("1", "true"):
        return None
    try:
        t0 = time.time()
        # Use clones to avoid potential in-place mutation
        warm = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in filtered_inputs.items()}
        with torch.inference_mode():
            _ = model.generate(**warm, max_new_tokens=1, do_sample=False)
        return time.time() - t0
    except Exception:
        return None


def _decode_outputs(processor, batch_inputs, generated_ids) -> str:
    """Trim prompt tokens and decode to text."""
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_inputs.input_ids, generated_ids)]
    texts = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return texts[0] if texts else ""


# ---------------------------
# Public API
# ---------------------------

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
    """Run LLaVA-OneVision on a video and return a JSON-able result.

    This function keeps the existing external API used by scripts/llm_eval.py,
    but the internal flow is simplified and heavily commented for clarity.
    """
    prompt = default_prompt() if prompt is None else prompt

    # 1) Load model and processor
    model, processor = load_llava(model_id)

    # 2) Read frames (optionally sampled); apply a safety cap to avoid overflow
    frames = read_video_frames(video_path, sample_fps=sample_fps, max_frames=max_frames)
    if max_frames is None and len(frames) > safety_frames_cap:
        img_subset = frames[:safety_frames_cap]
        clipped = True
    else:
        img_subset = frames
        clipped = max_frames is not None and len(frames) > max_frames

    # 3) Choose input mode and prepare HF inputs
    mode = "video" if use_video_mode else "image"
    batch_inputs = _prepare_inputs(processor, mode, img_subset, video_path, prompt)

    # 4) Place inputs on device (best-effort)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_inputs = _to_device(batch_inputs, device)

    # 5) Filter unexpected kwargs for model.generate
    filtered_inputs = _filter_model_inputs(model, batch_inputs)

    # Optional warmup (disabled by default, enable with LLAVA_WARMUP=1)
    warmup_s = _maybe_warmup(model, filtered_inputs)

    # 6) Build generation kwargs; allow env to disable sampling entirely
    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    use_sampling = temperature is not None and float(temperature) > 0.0
    if os.environ.get("LLAVA_DISABLE_SAMPLING", "0").lower() in ("1", "true", "yes"):
        use_sampling = False
    if use_sampling:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})

    # Set eos/pad ids if available to silence warnings on some models
    try:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = int(eos_id)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = int(pad_id)
    except Exception:
        pass

    # 7) Generate and decode
    t0 = time.time()
    with torch.inference_mode():
        try:
            generated_ids = model.generate(**filtered_inputs, **gen_kwargs)
        except Exception:
            # Fallback: if sampling errors out, retry without sampling to salvage run
            if use_sampling:
                gen_kwargs_fallback = {k: v for k, v in gen_kwargs.items() if k not in {"do_sample", "temperature"}}
                generated_ids = model.generate(**filtered_inputs, **gen_kwargs_fallback)
            else:
                raise
    gen_s = time.time() - t0
    response = _decode_outputs(processor, batch_inputs, generated_ids)

    # 8) Assemble result
    result: Dict[str, Any] = {
        "model": model_id,
        "language": "en",
        "video": os.path.basename(video_path),
        "video_path": os.path.abspath(video_path),
        "num_frames": len(frames),
        "sampling": {
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "mode": mode,
            "frames_used": len(img_subset),
            "clipped": bool(clipped),
        },
        "prompt": prompt,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "response": response,
    }
    try:
        result["timings"] = {"warmup_s": warmup_s, "generate_s": gen_s}
        result["device_map"] = str(getattr(model, "hf_device_map", None))
    except Exception:
        pass
    return result
