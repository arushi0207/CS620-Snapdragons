# CS620-Snapdragons

## TODOs
- setup integrated environment that supports the hrnet_pose_extractor
- fix errors and complete the pipeline

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
### Run webcam demo
```shell
python -m scripts.webcam_demo
```