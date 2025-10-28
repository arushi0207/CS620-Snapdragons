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

For running Facemap 3DMM demo, do the following:
```shell
pip install "qai-hub-models[facemap-3dmm]"
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
```

### Using an ONNX FaceMap model
1. Download the optimized package from [Qualcomm® AI Hub](https://aihub.qualcomm.com/models/facemap_3dmm?domain=Computer+Vision&useCase=Pose+Estimation) (e.g., `job_jgnm2o7vp_optimized_onnx`) and copy it under `assets/models/facemap_3dmm/` so that `model.onnx` and its companion `model.data` sit side-by-side. The project automatically picks up the first `.onnx` file it finds there.
2. (Optional) Override the location by setting `FACEMAP_ONNX_DIR=/path/to/model/folder` or passing `onnx_model_path`/`onnx_model_dir` when instantiating `FaceMap3DMMExtractor` in code.
3. When an ONNX model is found, `FaceMap3DMMExtractor` uses `onnxruntime` instead of the PyTorch/QAI Hub helper; if no ONNX model is present it falls back to the original workflow.
