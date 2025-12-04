from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from google import genai

from featurehub.llm import llava_onevision


# Keep the same public pattern as other backends
DEFAULT_MODEL_ID = "gemini-2.5-flash"


def default_prompt() -> str:
    """Return the default English evaluation prompt.

    We reuse the LLaVA-OneVision prompt so that Gemini receives
    exactly the same instructions as other backends.
    """
    return llava_onevision.default_prompt()



def generate_evaluation(
    video_path: str,
    prompt: Optional[str] = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> Dict[str, Any]:
    """Run a Gemini video evaluation on the given presentation video.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    prompt:
        Evaluation prompt. If None, uses `default_prompt()`.
    model_id:
        Gemini model name, e.g. "gemini-2.5-flash".
    gen_kwargs:
        Extra keyword arguments forwarded to `client.models.generate_content`,
        e.g. temperature, max_output_tokens, etc.

    Returns
    -------
    dict
        JSON-serializable result with the same structure as other backends,
        containing model metadata, video info, prompt, and Gemini's response.
    """
    if prompt is None:
        prompt = default_prompt()
    print(f"Using Gemini model: {model_id}")
    # print(f"Using prompt: {prompt}")

    abs_video_path = os.path.abspath(video_path)
    client = genai.Client()

    # Upload the video file to Gemini, then call the multimodal API
    print(f"Uploading video file {abs_video_path} to Gemini...")
    video_file = client.files.upload(file=abs_video_path)
    while video_file.state != "ACTIVE":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)
        print(f"waiting for upload... current state: {video_file.state}")
    print("Upload complete.")
    response = client.models.generate_content(
        model=model_id,
        contents=[video_file, prompt],
        )

    # `response.text` is the standard way to access the generated text
    text = getattr(response, "text", None)
    if text is None:
        # Fallback for future/alternate response shapes
        text = str(response)

    result: Dict[str, Any] = {
        "model": model_id,
        "language": "en",
        "video": os.path.basename(abs_video_path),
        "video_path": abs_video_path,
        "prompt": prompt,
        "generation": {
            # Let callers control details via gen_kwargs if needed
        },
        "response": text,
    }

    return result
