import os
import math
import hashlib
from typing import Any, Dict
import requests

from IPython.display import Markdown, display
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
import decord
from decord import VideoReader, cpu
# pip install qwen-vl-utils[decord]

from transformers import AutoProcessor, AutoModelForImageTextToText

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



def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

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


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image

def inference(video, prompt, processor, model, max_new_tokens=2048, total_pixels=20480 * 32 * 32, min_pixels=64 * 32 * 32, max_frames= 2048, sample_fps = 2):
    """
    Perform multimodal inference on input video and text prompt to generate model response.

    Args:
        video (str or list/tuple): Video input, supports two formats:
            - str: Path or URL to a video file. The function will automatically read and sample frames.
            - list/tuple: Pre-sampled list of video frames (PIL.Image or url). 
              In this case, `sample_fps` indicates the frame rate at which these frames were sampled from the original video.
        prompt (str): User text prompt to guide the model's generation.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Default is 2048.
        total_pixels (int, optional): Maximum total pixels for video frame resizing (upper bound). Default is 20480*32*32.
        min_pixels (int, optional): Minimum total pixels for video frame resizing (lower bound). Default is 16*32*32.
        sample_fps (int, optional): ONLY effective when `video` is a list/tuple of frames!
            Specifies the original sampling frame rate (FPS) from which the frame list was extracted.
            Used for temporal alignment or normalization in the model. Default is 2.

    Returns:
        str: Generated text response from the model.

    Notes:
        - When `video` is a string (path/URL), `sample_fps` is ignored and will be overridden by the video reader backend.
        - When `video` is a frame list, `sample_fps` informs the model of the original sampling rate to help understand temporal density.
    """

    messages = [
        {"role": "user", "content": [
                {"video": video,
                "total_pixels": total_pixels, 
                "min_pixels": min_pixels, 
                "max_frames": max_frames,
                'sample_fps':sample_fps},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                   image_patch_size= 16,
                                                                   return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


# generate messages based on input
def generate_messages(video_path, prompt = default_prompt):

    display(Markdown(f"### Input Video Frames from {video_path}"))
    messages = [
        {
            "role":"user",
            "content":[
                {
                    "type":"video",
                    "video": video_path
                },
                {
                    "type":"text",
                    "text": prompt
                }
            ]
        }
    ]
    return messages

def generate_evaluation(
    video_path: str,
    prompt: str = default_prompt,
    model_id: str = "Qwen3/Qwen3-VL-2B-Instruct" ,
    max_new_tokens: int = 2048,
    num_frames: int = 64
):
    processor = AutoProcessor.from_pretrained(model_id)
    model, output_loading_info = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype="auto", device_map="cuda", output_loading_info=True)
    print("output_loading_info", output_loading_info)


    video_path, frames, timestamps = get_video_frames(video_path, num_frames=num_frames)
    image_grid = create_image_grid(frames, num_columns=8)
    display(image_grid.resize((640, 640)))

    response = inference(video_path, prompt, processor=processor, model=model, max_new_tokens=max_new_tokens, max_frames=num_frames)

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
