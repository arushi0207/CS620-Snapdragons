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
python -m scripts.extract_from_video --video input.mp4 --out outputs/run2 --extractors mediapipe_face
```
