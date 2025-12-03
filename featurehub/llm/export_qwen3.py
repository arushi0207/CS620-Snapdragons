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
    使用你现有的逻辑，把 (video + prompt) 转成 Qwen3-VL 的张量输入。

    返回:
        inputs: 一个 dict，里面都是张量，比如:
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

    # 3) 用 qwen_vl_utils 处理视觉输入（视频路径 -> patch 序列）
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )

    if video_inputs is not None:
        # video_inputs 里是 (video_tensor, meta) 对，应拆开
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
    给 ONNX 导出用的包装：
    - 输入：input_ids, attention_mask, position_ids, pixel_values_videos, video_grid_thw
    - 输出：logits
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
            use_cache=False,  # 导出 ONNX 时一般关掉 cache
        )
        # 这里只导出 logits，采样/解码逻辑放在 ONNX 外面自己写
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
    用一个真实的视频 + prompt 构造样例输入，然后把 Qwen3-VL-2B-Instruct 导出成 ONNX。
    导出的模型输入就是你现在看到的 5 个张量。
    """

    # 1）加载 processor & 模型（导出 ONNX 建议放在 CPU 上）
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="float16",  # 或者 "auto"/float32，看你需要
        device_map=None,
    )
    _install_simple_causal_mask()
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    model.to("cpu")
    model.eval()

    # 2）构造一次 CPU 上的输入（注意这里 device="cpu"）
    inputs = build_qwen3vl_inputs(
        video=video_path,
        prompt=prompt,
        processor=processor,
        max_frames=num_frames,
        device="cpu",
    )

    # 只取你实际有的四个 key
    input_ids = inputs["input_ids"]              # [1, 5257]
    attention_mask = inputs["attention_mask"]    # [1, 5257]
    pixel_values_videos = inputs["pixel_values_videos"]  # [20160, 1536]
    video_grid_thw = inputs["video_grid_thw"]    # [1, 3]
    position_ids, _ = model.model.get_rope_index(
        input_ids,
        image_grid_thw=None,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )

    # 3）包装成我们的 ONNX wrapper
    wrapper = Qwen3VLONNXWrapper(model)

    # 按顺序列出输入名（和 wrapper.forward 的参数顺序一致）
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

    # 4）设置 dynamic_axes —— 哪些维度在推理时是可变的
    dynamic_axes = {
        # 文本部分：batch 和 seq 长度可变
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "position_ids": {0: "pos_three", 1: "batch", 2: "seq"},

        # 视频 token：第一维是 video_seq，可变；特征维 1536 固定
        "pixel_values_videos": {0: "video_seq"},

        "video_grid_thw": {0: "video_count"},

        "logits": {0: "batch", 1: "seq"},
    }

    # 5）真正导出 ONNX
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
            dynamo=False,  # 使用老的 torchscript 导出路径，绕过 torch.export 的限制
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
        video_path="test.mp4",
        prompt=default_prompt,
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        onnx_path="qwen3_vl_2b_instruct.onnx",
        num_frames=2,
    )

if __name__ == "__main__":
    main()
