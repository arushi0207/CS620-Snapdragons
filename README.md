# CS620-Snapdragons

## TODOs
- integrate an actual scoring model to replace the dummy scorer
- validate the FaceMap-only pipeline on representative speech videos

## Setup from Scratch

### 1. System Environment (for Windows)
Install Microsoft C++ Build Tools at https://visualstudio.microsoft.com/visual-cpp-build-tools/
![alt text](/assets/image.png)

### 2. Python Environment

Install Miniconda, and then set up a Python environment and install the required packages:
```shell
conda create -n quic_speechmodel python=3.12
conda activate quic_speechmodel
```

Install quantumaihub:
```shell
pip install qai_hub_models
```

For running Facemap 3DMM demo, do the following (you don't need it if using ONNX exported model):
```shell
pip install "qai-hub-models[facemap-3dmm]"
```

For running draft VLM:
```shell
pip install transformers
```

For LLaVA-OneVision (text evaluation):
```shell
pip install torch accelerate pillow
# transformers already installed above; trust_remote_code=True is used.
```

### 3. Configure AI Hub Access

Many features of AI Hub Models _(such as model compilation, on-device profiling, etc.)_ require access to Qualcomm® AI Hub:

-  [Create a Qualcomm® ID](https://myaccount.qualcomm.com/signup), and use it to [login to Qualcomm® AI Hub](https://app.aihub.qualcomm.com/).
-  Configure your [API token](https://app.aihub.qualcomm.com/account/): `qai-hub configure --api_token API_TOKEN`

## Running Commands
### Extract features from a video
```shell
python -m scripts.extract_from_video --video /path/to/input.mp4 --out outputs/run1 --output-video outputs/run1/annotated.mp4
```

The script runs both the FaceMap 3DMM and MediaPipe Face extractors by default. Per-frame outputs are written to `features.jsonl`/`features.npz`, an annotated video is emitted when `--output-video` is provided, and the placeholder speech score (1–10) prints to the terminal.

To pick a subset of extractors, pass `--extractors`:
```shell
python -m scripts.extract_from_video --video input.mp4 --out outputs/run3 --extractors mediapipe_face --output-video outputs/run3/annotated.mp4
python -m scripts.extract_from_video --video input.mp4 --out outputs/run3 --extractors facemap_3dmm --output-video outputs/run3/annotated.mp4
```

### LLaVA-OneVision Text Evaluation (Visual-only)
Run a visual-only assessment of a presentation video (no audio analysis). Outputs English JSON at `--out/evaluation_llava.json`.

```shell
python -m scripts.llm_eval \
  --video input.mp4 \
  --out outputs/run4 \
  --model lmms-lab/LLaVA-OneVision-1.5-4B-stage0 \
  --max-new-tokens 512 \
  --temperature 0.2 \
  --max-frames 1
```

Optional arguments:
- `--prompt` or `--prompt-file` to override the default public-speaking coach prompt.
- `--sample-fps` and `--max-frames` to control frame usage. Default `--max-frames 64` is applied for safety to avoid context overflow.
- `--use-video-mode` to use the model's video pathway (still capped by `--max-frames`); image pathway is default for robustness.

Notes:
- Default path uses multiple images (first N frames) to prevent token-length overflow. Increase `--max-frames` cautiously; too many frames can exceed the model's maximum sequence length.
- `--use-video-mode` may produce longer sequences and extra model kwargs; it is provided for experimentation and remains safety-capped.
- Requires GPU for reasonable performance; `device_map="auto"` and `trust_remote_code=True` are used.
- This component is separate from the numerical `DummyScoreModel` (kept as-is). The JSON includes the raw LLM response.

### Using an ONNX FaceMap model
1. Download the optimized package from [Qualcomm® AI Hub](https://aihub.qualcomm.com/models/facemap_3dmm?domain=Computer+Vision&useCase=Pose+Estimation) (e.g., `job_jgnm2o7vp_optimized_onnx`) and copy it under `assets/models/facemap_3dmm/` so that `model.onnx` and its companion `model.data` sit side-by-side. The project automatically picks up the first `.onnx` file it finds there.
2. (Optional) Override the location by setting `FACEMAP_ONNX_DIR=/path/to/model/folder` or passing `onnx_model_path`/`onnx_model_dir` when instantiating `FaceMap3DMMExtractor` in code.
3. When an ONNX model is found, `FaceMap3DMMExtractor` uses `onnxruntime` instead of the PyTorch/QAI Hub helper; if no ONNX model is present it falls back to the original workflow.
