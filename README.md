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
Update Version
Just run 
```shell
pip install -r requirements.txt
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
pip install qwen-vl-utils
pip install torch accelerate pillow
# transformers already installed above; trust_remote_code=True is used.
```

For Qwen-3 VL Text Evaluation, also install Decord:
```shell
pip install qwen-vl-utils[decord]
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

### Run Gemini Video Evaluation (Visual + Audio)
```shell
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
python -m scripts.llm_eval   --video input.mp4   --out outputs/gemini_eval   --model gemini-2.5-flash
```

### LLaVA-OneVision Text Evaluation (Visual-only)
Run a visual-only assessment of a presentation video (no audio analysis). Outputs English JSON at `--out/evaluation_llava.json`.

```shell
python -m scripts.llm_eval \
  --video test.mp4 \
  --out outputs/run4 \
  --model lmms-lab/LLaVA-OneVision-1.5-4B-Instruct \
  --max-new-tokens 512 \
  --temperature 0.2 \
  --max-frames 1
```
### Qwen-3 VL Text Evaluation (Visual-only)
```shell
python -m scripts.llm_eval --video test.mp4 --out outputs/run_qwen --model Qwen/Qwen3-VL-2B-Instruct --num-frames 16 --output-json evaluation_qwen.json
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

### Qwen3-VL ONNX Evaluation
```shell
python -m scripts.llm_eval --video test.mp4 --out outputs/run_qwen_onnx --onnx-path /home/nick12138/projects/CS620-Snapdragons/qwen3_vl_onnx/qwen3_vl_2b_instruct.onnx
```

### Export Qwen3-VL (vision + decoder) to ONNX
```shell
python -m scripts.export_qwen3_vl_onnx \
  --video test.mp4 \
  --output-dir assets/models/qwen3_vl \
  --num-frames 8 \
  --dtype fp16 \
  --offline
```
This writes `vision_encoder.onnx` and `decoder.onnx` (split for Snapdragon deployment with dynamic frame/sequence axes). Use a small `--num-frames` if you want a lighter tracing sample; increase if you need to stress larger frame counts when exporting.

To mirror the existing PyTorch eval command but with ONNX (once an ORT runner is wired up), the invocation would look like:
```shell
python -m scripts.llm_eval_onnx \
  --video test.mp4 \
  --out outputs/run_qwen \
  --vision-onnx assets/models/qwen3_vl/vision_encoder.onnx \
  --decoder-onnx assets/models/qwen3_vl/decoder.onnx \
  --num-frames 128 \
  --output-json evaluation_qwen.json
```
Note: `scripts.llm_eval` today is PyTorch-only; an ONNX-capable runner (e.g., `scripts.llm_eval_onnx`) must be added to load the split graphs, run ORT, and manage KV cache during decoding.

### Using an ONNX FaceMap model
1. Download the optimized package from [Qualcomm® AI Hub](https://aihub.qualcomm.com/models/facemap_3dmm?domain=Computer+Vision&useCase=Pose+Estimation) (e.g., `job_jgnm2o7vp_optimized_onnx`) and copy it under `assets/models/facemap_3dmm/` so that `model.onnx` and its companion `model.data` sit side-by-side. The project automatically picks up the first `.onnx` file it finds there.
2. (Optional) Override the location by setting `FACEMAP_ONNX_DIR=/path/to/model/folder` or passing `onnx_model_path`/`onnx_model_dir` when instantiating `FaceMap3DMMExtractor` in code.
3. When an ONNX model is found, `FaceMap3DMMExtractor` uses `onnxruntime` instead of the PyTorch/QAI Hub helper; if no ONNX model is present it falls back to the original workflow.
