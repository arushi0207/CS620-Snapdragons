# CS620-Snapdragons

## Setup

### 1. Install Python Package

Set up a Python environment and install the required packages using pip:
```shell
conda create -n quic_ai_hub python=3.12
conda activate quic_ai_hub
```

Install quantumaihub:
```shell
pip install qai_hub_models
```

For running YOLO7 demo, do the following:
```shell
pip install "qai-hub-models[yolov7]"
pip install seaborn
python -m qai_hub_models.models.yolov7.demo
```