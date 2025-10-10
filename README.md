# CS620-Snapdragons

## Setup

### 1. Python Environment

Set up a Python environment and install the required packages using pip:
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

### 2. Configure AI Hub Access

Many features of AI Hub Models _(such as model compilation, on-device profiling, etc.)_ require access to Qualcomm® AI Hub:

-  [Create a Qualcomm® ID](https://myaccount.qualcomm.com/signup), and use it to [login to Qualcomm® AI Hub](https://app.aihub.qualcomm.com/).
-  Configure your [API token](https://app.aihub.qualcomm.com/account/): `qai-hub configure --api_token API_TOKEN`
