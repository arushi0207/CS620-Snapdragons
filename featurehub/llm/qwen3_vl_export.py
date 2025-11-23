from __future__ import annotations

"""
Utilities to export Qwen3-VL to ONNX as two graphs:
- Vision encoder: video patches -> visual embeddings + deepstack features
- Text decoder: consumes embeddings + past key values for autoregressive decoding

The Snapdragon deployment stack can then handle pre/post-processing and the
generation loop outside of ONNX Runtime.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache
import transformers.masking_utils as masking_utils
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_modeling

# Keep defaults close to the repo layout
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_OUTPUT_DIR = os.path.join("assets", "models", "qwen3_vl")


def _install_simple_causal_mask():
    """
    Replace the transformers causal mask with a simpler, ONNX-friendly version to avoid
    torch.export / torchscript tracing issues with functorch vmap in masking_utils.
    """

    def simple_causal_mask(
        config,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        position_ids: torch.Tensor,
        **kwargs,
    ):
        batch_size, query_len = input_embeds.shape[:2]
        device = input_embeds.device
        dtype = input_embeds.dtype
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = past_len + query_len

        # Base causal mask (allow attending to past + current positions)
        q_positions = cache_position
        if q_positions.numel() != query_len:
            q_positions = torch.arange(past_len, past_len + query_len, device=device)
        k_positions = torch.arange(kv_len, device=device)
        causal = q_positions.view(1, query_len, 1) >= k_positions.view(1, 1, kv_len)
        causal = causal.expand(batch_size, query_len, kv_len)

        # Optional padding mask
        if attention_mask is not None:
            pad = attention_mask[:, None, None, -kv_len:].to(torch.bool)
            allowed = causal & pad
        else:
            allowed = causal

        mask = torch.where(
            allowed, torch.tensor(0.0, device=device, dtype=dtype), torch.finfo(dtype).min
        )
        return mask.unsqueeze(1)  # (batch, 1, q_len, kv_len)

    qwen3_vl_modeling.create_causal_mask = simple_causal_mask
    masking_utils.create_causal_mask = simple_causal_mask


@dataclass
class SampleBatch:
    """Lightweight container for sample tensors used during export."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor


def _load_model_and_processor(
    model_id: str,
    dtype: torch.dtype,
    local_files_only: bool,
    attn_impl: str = "eager",
):
    """
    Load the HF model + processor in eval mode. Using the official transformers
    implementation (no trust_remote_code needed for Qwen3-VL).
    """
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    model.eval()
    _install_simple_causal_mask()
    # Force eager attention to avoid unsupported custom ops in ONNX/TensorRT.
    try:
        model.config._attn_implementation = attn_impl
    except Exception:
        pass
    return model, processor


def _prepare_processor_inputs(
    processor,
    video_path: str,
    prompt: str,
    num_frames: int,
) -> Dict[str, torch.Tensor]:
    """
    Run the HF processor once to obtain realistic tensors (shapes match the
    runtime pipeline). This keeps `pixel_values_videos` aligned with
    `video_grid_thw`, which is critical for the vision encoder.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": video_path,
                    "max_frames": num_frames,
                    "total_pixels": 20480 * 32 * 32,
                    "min_pixels": 64 * 32 * 32,
                    "sample_fps": 2,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )
    return inputs


def _compute_position_ids(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, video_grid_thw: torch.Tensor):
    """
    Use the model helper to compute multimodal position_ids and rope_deltas once.
    """
    position_ids, rope_deltas = model.model.get_rope_index(
        input_ids,
        image_grid_thw=None,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    model.model.rope_deltas = rope_deltas
    return position_ids


def _build_sample_batch(
    model,
    processor,
    video_path: str,
    prompt: str,
    num_frames: int,
    device: torch.device,
) -> SampleBatch:
    """
    Prepare one batch of sample tensors (moved to the target device) for tracing.
    """
    inputs = _prepare_processor_inputs(processor, video_path, prompt, num_frames)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    video_grid_thw = inputs["video_grid_thw"].to(device)
    position_ids = _compute_position_ids(model, input_ids, attention_mask, video_grid_thw)
    cache_position = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)

    return SampleBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values_videos=inputs["pixel_values_videos"].to(device),
        video_grid_thw=video_grid_thw,
        position_ids=position_ids.to(device),
        cache_position=cache_position,
    )


def _empty_past_kv(num_layers: int, num_key_value_heads: int, head_dim: int, batch_size: int, device, dtype):
    """
    Build an empty legacy-style KV cache so ONNX tracing includes cache inputs.
    """
    zeros = []
    shape = (batch_size, num_key_value_heads, 0, head_dim)
    for _ in range(num_layers):
        zeros.append(torch.zeros(shape, device=device, dtype=dtype))
        zeros.append(torch.zeros(shape, device=device, dtype=dtype))
    return tuple(zeros)


class VisionEncoderWrapper(torch.nn.Module):
    """
    Thin wrapper around the vision tower so ONNX export only includes the
    necessary inputs/outputs for Snapdragon deployment.
    """

    def __init__(self, vision_model):
        super().__init__()
        self.visual = vision_model

    def forward(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor):
        vision_embeds, deepstack_features = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return (vision_embeds,) + tuple(deepstack_features)


class DecoderWrapper(torch.nn.Module):
    """
    Wrap the text decoder + LM head. Visual embeddings are injected directly
    into the token embeddings; past key values are provided/returned in
    legacy (tuple) format for ORT friendliness.
    """

    def __init__(self, model, num_layers: int):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.video_token_id = model.config.video_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        video_embeds: torch.Tensor,
        deepstack_0: torch.Tensor,
        deepstack_1: torch.Tensor,
        deepstack_2: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        # Embed tokens (text + placeholder video tokens)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Replace placeholder tokens with visual embeddings
        video_mask = (input_ids == self.video_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask.expand_as(inputs_embeds), video_embeds)
        visual_pos_masks = video_mask.squeeze(-1)
        deepstack_visual_embeds = [deepstack_0, deepstack_1, deepstack_2]

        # Past KV cache (legacy tuple -> DynamicCache)
        cache = None
        if past_key_values:
            legacy = tuple(
                (past_key_values[2 * i], past_key_values[2 * i + 1]) for i in range(self.num_layers)
            )
            cache = DynamicCache.from_legacy_cache(legacy)

        outputs = self.model.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=True,
        )

        logits = self.model.lm_head(outputs.last_hidden_state)
        present = outputs.past_key_values.to_legacy_cache()
        flat_present = tuple(elem for layer in present for elem in layer)

        return (logits,) + flat_present


def _vision_dynamic_shapes(deepstack_count: int):
    """
    Dynamic shape hints for vision inputs (outputs need no entries).
    """
    return {
        "pixel_values_videos": {0: "vision_patches"},
        "video_grid_thw": {0: "num_videos"},
    }


# def _decoder_dynamic_shapes(num_layers: int, deepstack_count: int):
#     """
#     Dynamic shape hints for decoder inputs. Position ids are [3, batch, seq_len].
#     """
#     shapes = {
#         "input_ids": {0: "batch", 1: "seq_len"},
#         "attention_mask": {0: "batch", 1: "total_seq"},
#         "position_ids": {1: "batch", 2: "seq_len"},
#         "cache_position": {0: "seq_len"},
#         "video_embeds": {0: "vision_tokens"},
#     }
#     for idx in range(deepstack_count):
#         shapes[f"deepstack_{idx}"] = {0: "vision_tokens"}
#     for layer in range(num_layers):
#         shapes[f"past_key_values.{layer}.key"] = {0: "batch", 2: "past_seq"}
#         shapes[f"past_key_values.{layer}.value"] = {0: "batch", 2: "past_seq"}
#     return shapes

def _decoder_dynamic_shapes(num_layers: int, deepstack_count: int):
    shapes = [
        {0: "batch", 1: "seq_len"},  # input_ids
        {0: "batch", 1: "total_seq"},  # attention_mask
        {1: "batch", 2: "seq_len"},  # position_ids
        {0: "seq_len"},  # cache_position
        {0: "vision_tokens"},  # video_embeds
    ]
    shapes += [{0: "vision_tokens"} for _ in range(deepstack_count)]  # deepstack_0..N

    # past_key_values arrives as a single tuple arg; mirror that structure.
    past_shapes = tuple({0: "batch", 2: "past_seq"} for _ in range(2 * num_layers))  # k/v per layer
    shapes.append(past_shapes)

    return tuple(shapes)

def _vision_output_names(deepstack_count: int) -> List[str]:
    names = ["vision_embeddings"]
    names.extend([f"deepstack_{i}" for i in range(deepstack_count)])
    return names


def _decoder_io_names(num_layers: int, deepstack_count: int) -> Tuple[List[str], List[str]]:
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "cache_position",
        "video_embeds",
    ]
    input_names += [f"deepstack_{i}" for i in range(deepstack_count)]
    for layer in range(num_layers):
        input_names.append(f"past_key_values.{layer}.key")
        input_names.append(f"past_key_values.{layer}.value")

    output_names = ["logits"]
    for layer in range(num_layers):
        output_names.append(f"present.{layer}.key")
        output_names.append(f"present.{layer}.value")

    return input_names, output_names


def export_qwen3_vl_to_onnx(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dtype: torch.dtype = torch.float16,
    opset: int = 18,
    video_path: str = "test.mp4",
    prompt: str = "Describe the video.",
    num_frames: int = 8,
    local_files_only: bool = True,
):
    """
    Orchestrate the full export:
    1) Build sample inputs via the processor (so shapes align).
    2) Export vision encoder (pixel_values_videos + video_grid_thw -> visual embeddings).
    3) Export text decoder (input_ids + embeddings + past kv -> logits + next kv).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = _load_model_and_processor(model_id, dtype=dtype, local_files_only=local_files_only)
    model.to(device)

    sample_batch = _build_sample_batch(
        model,
        processor,
        video_path=video_path,
        prompt=prompt,
        num_frames=num_frames,
        device=device,
    )

    # Vision encoder export
    vision_wrapper = VisionEncoderWrapper(model.model.visual).to(device)
    deepstack_count = len(model.config.vision_config.deepstack_visual_indexes)
    with torch.no_grad():
        vision_out = vision_wrapper(sample_batch.pixel_values_videos, sample_batch.video_grid_thw)

    vision_path = os.path.join(output_dir, "vision_encoder.onnx")
    torch.onnx.export(
        vision_wrapper,
        (sample_batch.pixel_values_videos, sample_batch.video_grid_thw),
        vision_path,
        input_names=["pixel_values_videos", "video_grid_thw"],
        output_names=_vision_output_names(deepstack_count),
        dynamic_shapes=_vision_dynamic_shapes(deepstack_count),
        opset_version=opset,
        do_constant_folding=False,
        dynamo=True,
        fallback=True,
    )

    # Decoder export (prefill-style: past cache is empty but present is returned)
    text_cfg = model.config.text_config
    past = _empty_past_kv(
        num_layers=text_cfg.num_hidden_layers,
        num_key_value_heads=text_cfg.num_key_value_heads,
        head_dim=text_cfg.hidden_size // text_cfg.num_attention_heads,
        batch_size=sample_batch.input_ids.shape[0],
        device=device,
        dtype=dtype,
    )

    decoder_wrapper = DecoderWrapper(model, num_layers=text_cfg.num_hidden_layers).to(device)
    decoder_inputs: Sequence[torch.Tensor] = (
        sample_batch.input_ids,
        sample_batch.attention_mask,
        sample_batch.position_ids,
        sample_batch.cache_position,
        vision_out[0],
        *vision_out[1:],
        *past,
    )

    decoder_path = os.path.join(output_dir, "decoder.onnx")
    decoder_input_names, decoder_output_names = _decoder_io_names(text_cfg.num_hidden_layers, deepstack_count)
    torch.onnx.export(
        decoder_wrapper,
        decoder_inputs,
        decoder_path,
        input_names=decoder_input_names,
        output_names=decoder_output_names,
        dynamic_shapes=_decoder_dynamic_shapes(text_cfg.num_hidden_layers, deepstack_count),
        opset_version=opset,
        do_constant_folding=False,
        dynamo=True,
        fallback=True
    )

    return {
        "vision_encoder": vision_path,
        "decoder": decoder_path,
        "device": str(device),
        "dtype": str(dtype),
        "opset": opset,
    }


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export Qwen3-VL vision encoder + text decoder to ONNX.")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id (must be cached locally).")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write ONNX files.")
    ap.add_argument("--video", default="test.mp4", help="Sample video used to build tracing inputs.")
    ap.add_argument("--prompt", default="Describe the video.", help="Sample prompt for tracing inputs.")
    ap.add_argument("--num-frames", type=int, default=8, help="Sample frame count for tracing (affects shapes).")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset to target.")
    ap.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "fp32"],
        help="Precision for export; Snapdragon prefers fp16.",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Force local cache only (prevents network fetches).",
    )
    return ap


def main():
    ap = _build_argparser()
    args = ap.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    # print("Number of frames:", args.num_frames)
    paths = export_qwen3_vl_to_onnx(
        model_id=args.model_id,
        output_dir=args.output_dir,
        dtype=dtype,
        opset=args.opset,
        video_path=args.video,
        prompt=args.prompt,
        num_frames=args.num_frames,
        local_files_only=args.offline,
    )
    print("Exported ONNX files:", paths)


if __name__ == "__main__":
    main()
