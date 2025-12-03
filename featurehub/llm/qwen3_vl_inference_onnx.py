import os
import math
import hashlib
from typing import Any, Dict, Iterable, List, Sequence
from pathlib import Path

import numpy as np
from IPython.display import Markdown, display
from PIL import Image
from decord import VideoReader, cpu
import onnxruntime as ort
import torch
from transformers import AutoProcessor

# Keep preprocessing consistent with the export pipeline.
try:
    from .export_qwen3 import build_qwen3vl_inputs, default_prompt
except ImportError:
    from export_qwen3 import build_qwen3vl_inputs, default_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_PATH = str(PROJECT_ROOT / "qwen3_vl_onnx" / "qwen3_vl_2b_instruct.onnx")


def get_video_frames(video_path: str, num_frames: int = 128, cache_dir: str = ".cache"):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode("utf-8")).hexdigest()
    video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f"{video_hash}_{num_frames}_frames.npy")
    timestamps_cache_file = os.path.join(cache_dir, f"{video_hash}_{num_frames}_timestamps.npy")

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)

    return video_file_path, frames, timestamps


def create_image_grid(images: Sequence[np.ndarray], num_columns: int = 8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new("RGB", (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def _normalize_token_ids(values: Iterable[int | List[int] | None]) -> List[int]:
    ids: List[int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            ids.extend(int(v) for v in value if v is not None)
        else:
            ids.append(int(value))
    return ids


def _load_ort_session(onnx_path: str, providers: List[str] | None = None) -> ort.InferenceSession:
    provider_order = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    resolved = [p for p in provider_order if p in available]
    if not resolved:
        raise RuntimeError(f"No valid ONNX Runtime providers available. Requested {provider_order}, available {available}.")
    return ort.InferenceSession(onnx_path, providers=resolved)


def _onnx_type_to_dtype(type_str: str | None):
    if type_str is None:
        return None
    if "float16" in type_str:
        return np.float16
    if "float" in type_str:
        return np.float32
    if "int64" in type_str:
        return np.int64
    if "int32" in type_str:
        return np.int32
    return None


def _to_numpy(tensor: torch.Tensor, target_dtype):
    array = tensor.cpu().numpy()
    if target_dtype is not None and array.dtype != target_dtype:
        array = array.astype(target_dtype)
    return array


def _compute_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_grid_thw: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
    vision_start_token_id: int,
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    """
    Minimal reimplementation of Qwen3-VL get_rope_index for video-only export.
    Builds a (3, batch, seq_len) position_ids tensor on CPU.
    """
    device = input_ids.device
    batch, seq_len = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    video_grid_thw = video_grid_thw.to(device)
    image_grid_thw = image_grid_thw.to(device) if image_grid_thw is not None else None

    position_ids = torch.zeros((3, batch, seq_len), device=device, dtype=input_ids.dtype)

    for b in range(batch):
        ids = input_ids[b][attention_mask[b] == 1]
        input_tokens = ids.tolist()

        vision_start_indices = (ids == vision_start_token_id).nonzero(as_tuple=False).squeeze(1)
        if vision_start_indices.numel() > 0:
            vision_tokens = ids[vision_start_indices + 1]
        else:
            vision_tokens = torch.tensor([], device=device, dtype=ids.dtype)

        image_nums = int((vision_tokens == image_token_id).sum().item())
        video_nums = int((vision_tokens == video_token_id).sum().item())

        st = 0
        remain_images, remain_videos = image_nums, video_nums
        image_index = 0
        video_index = 0
        llm_pos_ids_list: List[torch.Tensor] = []

        for _ in range(image_nums + video_nums):
            ed_image = len(input_tokens) + 1
            ed_video = len(input_tokens) + 1
            if remain_images > 0:
                try:
                    ed_image = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_image = len(input_tokens) + 1
            if remain_videos > 0:
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
            if text_len > 0:
                llm_pos_ids_list.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t, device=device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = (
                torch.arange(llm_grid_h, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx)

        if llm_pos_ids_list:
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[:, b, attention_mask[b] == 1] = llm_positions

    return position_ids


def inference(
    video,
    prompt: str,
    processor,
    session: ort.InferenceSession,
    max_new_tokens: int = 2048,
    total_pixels: int = 20480 * 32 * 32,
    min_pixels: int = 64 * 32 * 32,
    max_frames: int = 2048,
    sample_fps: int = 2,
):
    """
    Perform multimodal inference using the exported ONNX decoder.

    Args:
        video: Path/URL or a list of frames for the video input.
        prompt: Text prompt.
        processor: Hugging Face processor for Qwen3-VL.
        session: ONNX Runtime session loaded from the exported graph.
        max_new_tokens: Max tokens to generate.
        total_pixels/min_pixels/max_frames/sample_fps: Passed through to preprocessing.
    """
    with torch.no_grad():
        inputs = build_qwen3vl_inputs(
            video=video,
            prompt=prompt,
            processor=processor,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            max_frames=max_frames,
            sample_fps=sample_fps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    # (Pdb) p inputs.keys()
    # KeysView({'input_ids': tensor([[151644,    872,    198,  ..., 151644,  77091,    198]],
    #        device='cuda:0'), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]], device='cuda:0'), 'pixel_values_videos': tensor([[0.5216, 0.5216, 0.5216,  ..., 0.4745, 0.4667, 0.4588],
    #         [0.5373, 0.5373, 0.5373,  ..., 0.4824, 0.4824, 0.4824],
    #         [0.5294, 0.5294, 0.5294,  ..., 0.4275, 0.4275, 0.4196],
    #         ...,
    #         [0.4745, 0.4745, 0.4824,  ..., 0.5686, 0.5451, 0.5451],
    #         [0.5765, 0.5686, 0.5686,  ..., 0.7020, 0.7098, 0.6941],
    #         [0.5059, 0.4980, 0.4902,  ..., 0.5686, 0.5765, 0.5686]],
    #        device='cuda:0'), 'video_grid_thw': tensor([[ 7, 40, 72]], device='cuda:0')})
    # input_ids: torch.Size([1, 5273])
    # attention_mask: torch.Size([1, 5273])
    # pixel_values_videos: torch.Size([20160, 1536])
    # video_grid_thw: torch.Size([1, 3])
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_type_map = {i.name: i.type for i in session.get_inputs()}
    id_dtype = _onnx_type_to_dtype(input_type_map.get("input_ids"))
    mask_dtype = _onnx_type_to_dtype(input_type_map.get("attention_mask"))
    pos_dtype = _onnx_type_to_dtype(input_type_map.get("position_ids"))
    pixel_dtype = _onnx_type_to_dtype(input_type_map.get("pixel_values_videos"))
    grid_dtype = _onnx_type_to_dtype(input_type_map.get("video_grid_thw"))
    has_position_ids_input = "position_ids" in input_type_map

    pixel_values_videos = _to_numpy(inputs["pixel_values_videos"], pixel_dtype)
    video_grid_thw = _to_numpy(inputs["video_grid_thw"], grid_dtype)

    print(f"Input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, pixel_values_videos={inputs['pixel_values_videos'].shape}, video_grid_thw={inputs['video_grid_thw'].shape}")

    stop_token_ids = set(
        _normalize_token_ids(
            [
                getattr(processor.tokenizer, "eos_token_id", None),
                processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ]
        )
    )

    generated: List[int] = []
    video_grid_thw_t = inputs["video_grid_thw"].to("cpu")
    image_grid_thw_t = inputs.get("image_grid_thw")
    if image_grid_thw_t is not None:
        image_grid_thw_t = image_grid_thw_t.to("cpu")
    spatial_merge_size = 2  # from Qwen3-VL vision_config.spatial_merge_size
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    # ONNX graph does not return cache, so we decode autoregressively by re-running the full sequence.
    for _ in range(max_new_tokens):
        ort_inputs = {
            "input_ids": _to_numpy(input_ids, id_dtype),
            "attention_mask": _to_numpy(attention_mask, mask_dtype),
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        if has_position_ids_input:
            position_ids = _compute_position_ids(
                input_ids.to("cpu"),
                attention_mask.to("cpu"),
                video_grid_thw_t,
                image_grid_thw_t,
                spatial_merge_size=spatial_merge_size,
                vision_start_token_id=vision_start_token_id,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
            )
            ort_inputs["position_ids"] = _to_numpy(position_ids, pos_dtype)
        logits = session.run(None, ort_inputs)[0]
        next_token_id = int(np.argmax(logits[0, -1]))

        if stop_token_ids and next_token_id in stop_token_ids:
            break

        generated.append(next_token_id)
        next_token = torch.tensor([[next_token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        next_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)

    output_text = processor.batch_decode([generated], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


def generate_messages(video_path: str, prompt: str = default_prompt):
    display(Markdown(f"### Input Video Frames from {video_path}"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages


def generate_evaluation(
    video_path: str,
    prompt: str = default_prompt,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    onnx_path: str = DEFAULT_ONNX_PATH,
    max_new_tokens: int = 2048,
    num_frames: int = 64,
    providers: List[str] | None = None,
):
    processor = AutoProcessor.from_pretrained(model_id)
    session = _load_ort_session(onnx_path, providers=providers)

    video_path, frames, _ = get_video_frames(video_path, num_frames=num_frames)
    image_grid = create_image_grid(frames, num_columns=8)
    display(image_grid.resize((640, 640)))

    response = inference(
        video_path,
        prompt,
        processor=processor,
        session=session,
        max_new_tokens=max_new_tokens,
        max_frames=num_frames,
    )

    result: Dict[str, Any] = {
        "model": model_id,
        "language": "en",
        "video": os.path.basename(video_path),
        "video_path": os.path.abspath(video_path),
        "num_frames": num_frames,
        "prompt": prompt,
        "generation": {
            "max_new_tokens": max_new_tokens,
        },
        "response": response,
    }

    return result
