from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import torch
import transformers.masking_utils as masking_utils
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_modeling

default_prompt =    """You are a public speaking coach. Watch the following presentation video and provide a concise, visual-only evaluation. 
                    Do not infer anything about voice quality, audio, or verbal content. Assess: posture and stance, facial expression, 
                    eye contact with the camera, gesture variety and control, body movement, use of space, camera framing, lighting, 
                    background distractions, and use of visual aids (if any).\n\n
                    Structure your response as:\n
                    1) Overall Summary (2–3 sentences).\n
                    2) Strengths (3–5 bullet points).\n
                    3) Opportunities for Improvement (3–5 bullet points).\n
                    4) Actionable Suggestions (5–8 numbered, specific, behavior-focused tips).\n\n
                    Keep it supportive, direct, and actionable. Keep total length under 350 words."""

def build_qwen3vl_inputs(
    video,
    prompt: str,
    processor: AutoProcessor,
    total_pixels=20480 * 32 * 32,
    min_pixels=64 * 32 * 32,
    max_frames=2048,
    sample_fps=2,
    device: str = "cuda",
):
    """
    Convert (video + prompt) into Qwen3-VL tensor inputs using your existing logic.

    Returns:
        inputs: a dict whose values are tensors, such as:
            input_ids, attention_mask, position_ids,
            pixel_values_videos, video_grid_thw, ...
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": video,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    "max_frames": max_frames,
                    "sample_fps": sample_fps,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 3) Use qwen_vl_utils to process vision inputs (video path -> patch sequence)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )

    if video_inputs is not None:
        # video_inputs contains (video_tensor, meta) pairs and should be unpacked
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

    inputs = inputs.to(device)

    return inputs

import torch.nn as nn

def _install_simple_causal_mask():
    """
    The default masking path uses functorch vmap and higher-order autograd ops
    that break both torch.export and torchscript-based ONNX tracing. Replace it
    with a plain tensor implementation that is ONNX friendly.
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

        q_positions = cache_position
        if q_positions.numel() != query_len:
            q_positions = torch.arange(past_len, past_len + query_len, device=device)
        k_positions = torch.arange(kv_len, device=device)
        causal = q_positions.view(1, query_len, 1) >= k_positions.view(1, 1, kv_len)
        causal = causal.expand(batch_size, query_len, kv_len)

        if attention_mask is not None:
            pad = attention_mask[:, None, None, -kv_len:].to(torch.bool)
            allowed = causal & pad
        else:
            allowed = causal

        mask = torch.where(
            allowed, torch.tensor(0.0, device=device, dtype=dtype), torch.finfo(dtype).min
        )
        return mask.unsqueeze(1)

    qwen3_vl_modeling.create_causal_mask = simple_causal_mask
    masking_utils.create_causal_mask = simple_causal_mask

class Qwen3VLONNXWrapper(nn.Module):
    """
    Wrapper used for ONNX export:
    - Inputs: input_ids, attention_mask, position_ids, pixel_values_videos, video_grid_thw
    - Output: logits
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=False,  # Usually disable cache during ONNX export
        )
        print("outputs shapes", {k: v.shape for k, v in outputs.items() if hasattr(v, 'shape')})
        # Only export logits here; sampling/decoding should be implemented outside ONNX
        return outputs.logits
    
import torch.onnx
import onnx

def export_qwen3_vl_onnx(
    video_path: str,
    prompt: str = default_prompt,
    model_id: str = "Qwen3/Qwen3-VL-2B-Instruct",
    onnx_path: str = "qwen3_vl_2b_instruct.onnx",
    num_frames: int = 2,
    opset: int = 17,
):
    """
    Build a sample input from a real video + prompt, then export Qwen3-VL-2B-Instruct to ONNX.
    The exported model inputs are exactly the 5 tensors you see here.
    """

    # 1) Load processor and model (for ONNX export, keeping everything on CPU is recommended)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="float16",  # Or "auto"/float32, depending on your needs
        device_map=None,
    )
    _install_simple_causal_mask()
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    model.to("cpu")
    model.eval()

    # 2) Build one sample input on CPU (note device="cpu" here)
    inputs = build_qwen3vl_inputs(
        video=video_path,
        prompt="hi",
        processor=processor,
        max_frames=num_frames,
        device="cpu",
    )

    # Only take the four keys you actually need
    input_ids = inputs["input_ids"]              # [1, 5257]
    attention_mask = inputs["attention_mask"]    # [1, 5257]
    pixel_values_videos = inputs["pixel_values_videos"]  # [20160, 1536]
    video_grid_thw = inputs["video_grid_thw"]    # [1, 3]
    batch_size = input_ids.shape[0]
    sequence_length = input_ids.shape[1]
    position_ids = torch.ones(3, batch_size, sequence_length, dtype=torch.int64)

    # 3) Wrap with our ONNX wrapper
    wrapper = Qwen3VLONNXWrapper(model)

    # List input names in order (consistent with wrapper.forward parameter order)
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values_videos",
        "video_grid_thw",
    ]
    output_names = ["logits"]

    example_args = (
        input_ids,
        attention_mask,
        position_ids,
        pixel_values_videos,
        video_grid_thw,
    )

    # 4) Configure dynamic_axes — which dimensions are variable at inference time
    dynamic_axes = {
        # Text tensors: batch size and sequence length are variable
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "position_ids": {1: "batch", 2: "seq"},
        # Video tokens: first dimension is video_seq (variable); feature dim 1536 is fixed
        "pixel_values_videos": {0: "batch", 1: "num_frames"},

        "video_grid_thw": {0: "video_count"},

        "logits": {0: "batch", 1: "seq"},
    }
    wrapper.eval()
    example_args_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
    }

    # wrapper(**example_args_dict)  # test

    # make directory if not exists
    import os
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # 5) Actually export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_args,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    print(f"[OK] ONNX model saved to: {onnx_path}")

    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model passed onnx.checker.")
    except Exception as e:
        print("[WARN] ONNX check failed:", e)

def main():
    export_qwen3_vl_onnx(
        video_path="input.mp4",
        prompt=default_prompt,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        onnx_path="exportedonnx/qwen3_vl_2b_instruct.onnx",
        num_frames=2,
    )

if __name__ == "__main__":
    main()
