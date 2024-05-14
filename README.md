# ComfyUI LayerDivider
**ComfyUI LayerDivider** is custom nodes that generating layered psd files inside ComfyUI, original implement is 

## Environment
I only tested the following environment, it might work on other environment, but I don't test:
1. Windows 10
2. conda
3. Python 3.10
4. GTX 3090
5. Cuda 12.1

## Installation - ComfyUI
First, this plugin depends on **Python 3.10**, which means we cannot use the default Python that comes with ComfyUI, as it is Python 3.11. For this reason, it is recommended to use conda to manage and create the ComfyUI runtime environment:
- use cmd/terminal to enter the comfyui root folder (which includes run_cpu.bat and run_nvidia_gpu.bat) 
- `conda create --name comfy-py-310 python=3.10`
- `conda activate comfy-py-310`
- `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121`
- `pip install -r ComfyUI\requirements.txt`

Then you can run `python -s ComfyUI\main.py --windows-standalone-build` to check ComfyUI running properly. 

## Installation - CUDA & cuDNN
Next, this repo also requires specific versions of CUDA and cuDNN to be installed locally:
- For CUDA, I only install and test CUDA 12.1, you can find it from https://developer.nvidia.com/cuda-12-1-0-download-archive
- For cuDNN, it MUST be v8.9.2 - CUDA 12.x (according to https://github.com/mdboom/pytoshop/issues/9), you can find it from https://developer.nvidia.com/rdp/cudnn-archive
- After install and unzip, make sure you configure the PATH of your system variable

## Installation - ComfyUI LayerDivider
Then we can clone and configure this repo for ComfyUI:
- `cd ComfyUI\custom_nodes`
- `pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`
- `pip install Cython`
- `pip install pytoshop -I --no-cache-dir`
- `pip install psd-tools --no-deps`
- `git clone https://github.com/jtydhr88/ComfyUI-LayerDivider.git`
- `cd ComfyUI-LayerDivider`
- `pip install -r requirements.txt`

Congratulation! You complete all installation!

## Node Introduction
Currently, this extension includes two modes with three custom nodes, plus two layer modes(normal and composite) for each mode:

### Mode
- Color Base
- Segment Mask

### Layer Mode
- normal
- composite