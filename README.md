[EN](README.md) | [中文](README_ZH_CN.md)
# ComfyUI LayerDivider
**ComfyUI LayerDivider** is custom nodes that generating layered psd files inside ComfyUI, original implement is [mattyamonaca/layerdivider](https://github.com/mattyamonaca/layerdivider)

![image1](docs/layerdivider-color-base.png)
![image2](docs/layerdivider-seg-mask.png)
https://github.com/jtydhr88/ComfyUI-LayerDivider/assets/860985/3ceb0638-1ed7-4e01-b231-03c4408c95e3

## Environment
I tested the following environment, it might work on other environment, but I don't test:
### Common
1. Windows 10/Ubuntu
2. GTX 3090
3. Cuda 12.1

### Env 1 - see Method 1
1. ComfyUI embedded python (python 3.11) and ComfyUI Manager

### Env 2 - see Method 2
1. conda
2. Python 3.11

### Env 3 - see Method 3
1. conda
2. Python 3.11

### Env 4 - see Method 4
1. Ubuntu
2. conda/Python 3.11
3. cuda 12.1

## (Common) Installation - CUDA & cuDNN
This repo requires specific versions of CUDA and cuDNN to be installed locally:
- For CUDA, I only install and test CUDA 12.1, you can find it from https://developer.nvidia.com/cuda-12-1-0-download-archive
- For cuDNN, it MUST be v8.9.2 - CUDA 12.x (according to https://github.com/mdboom/pytoshop/issues/9), you can find it from https://developer.nvidia.com/rdp/cudnn-archive
- After install and unzip, make sure you configure the PATH of your system variable ![Path](docs/paths.png)

## (Common) Installation - Visual Studio Build Tools
It might also require Visual Studio Build Tools.
However, I am not sure because my local already installed previously. 
If it needs, you can find from [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools).

## (Method 1) Installation - ComfyUI Embedded Plugin & ComfyUI Manager 
1. You could clone this repo inside **comfyUI/custom_notes** directly `git clone https://github.com/jtydhr88/ComfyUI-LayerDivider.git`
2. or use ComfyUI Manager ![manager](docs/comfyui-manager.png)
3. However, no matter which way you choose, it will fail at first time ![error](docs/error.png)
4. Stop ComfyUI
5. Then go to **custom_nodes\ComfyUI-LayerDivider**, and run **install_windows_portable_win_py311_cu121.bat**

Done!

(If you prefer to use conda and python 3.10, you could follow the next)
## (Method 2) Installation - ComfyUI
You could use conda to manage and create the ComfyUI runtime environment:
- use cmd/terminal to enter the comfyui root folder (which includes run_cpu.bat and run_nvidia_gpu.bat) 
- `conda create --name comfy-py-310 python=3.10`
- `conda activate comfy-py-310`
- `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121`
- `pip install -r ComfyUI\requirements.txt`

Then you can run `python -s ComfyUI\main.py --windows-standalone-build` to check ComfyUI running properly. 

## (Method 2) Installation - ComfyUI LayerDivider
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

## (Method 3) Installation - ComfyUI LayerDivider
Assume you already have a conda python3.11 env
- activate your env
- go into this folder and run install_conda_win_py311_cu121.bat

Congratulation! You complete all installation!

## (Method 4) Ubuntu Installation - ComfyUI LayerDivider
Assume you already have a python3.11 env + cuda 12.1
- clone this repo inside custom_nodes folder
- cd ComfyUI-LayerDivider/
- pip install -r requirements.txt

Then make sure run them one by one:
- pip install cython
- pip install pytoshop -I --no-cache-dir
- pip install psd_tools
- pip install onnxruntime-gpu==1.17.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

Congratulation! You complete all installation!


## Node Introduction
Currently, this extension includes two modes with four custom nodes, plus two layer modes(normal and composite) for each mode:

### Mode
There are two main layered segmentation modes:
- Color Base - Layers based on similar colors, with parameters:
  - loops 
  - init_cluster 
  - ciede_threshold 
  - blur_size
- Segment Mask - First, the image is divided into segments using [SAM - segment anything](https://segment-anything.com/) to generate corresponding masks, then layers are created based on these masks.
  - Load SAM Mask Generator, with parameters (These come from segment anything, please refer to [here](https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/automatic_mask_generator.py#L61) for more details):
    - pred_iou_thresh 
    - stability_score_thresh 
    - min_mask_region_area
  - LayerDivider - Segment Mask, with parameters:
    - area_th: determines the number of partitions. The smaller the value, the more partitions there will be; the larger the value, the fewer partitions there will be.

### Layer Mode
Using in Divide Layer node to decide the layer mode:
- normal - Generates three layers for each region:
  - base - The base layer is the starting point for image processing
  - bright - The bright layer focuses on the brightest parts of the image, enhancing the brightness and gloss of these areas
  - shadow - The shadow layer deals with the darker parts of the image, emphasizing the details of shadows and dark areas.
- composite - Generates five layers for each region:
  - base - The base layer is the starting point of the image
  - screen - The screen layer simulates the effect of light overlay. It multiplies the color values of the image with the color values of the layer above it and then inverts the result, producing a brighter effect than the original image
  - multiply - The multiply layer simulates the effect of multiple images being overlaid. It directly multiplies the color values of the image with the color values of the layer above it, resulting in a darker effect than the original image.
  - subtract - The subtract layer subtracts the color values of the layer above from the base image, resulting in an image with lower color values.
  - addition - The addition layer adds the color values of the layer above to the base image, resulting in an image with higher color values.

## Example workflows
Here are two workflows for reference:
- [layerdivider-color-base.json](workflows/layerdivider-color-base.json) ![color-base](docs/layerdivider-color-base.png)
- [layerdivider-seg-mask.json](workflows/layerdivider-seg-mask-workflow.json) ![color-base](docs/layerdivider-seg-mask.png)

## Example outputs
- [output_color_base_composite.psd](docs/output_color_base_composite.psd)
- [output_color_base_normal.psd](docs/output_color_base_normal.psd)
- [output_seg_mask_composite.psd](docs/output_seg_mask_composite.psd)
- [output_seg_mask_normal.psd](docs/output_seg_mask_normal.psd)

## Known issues
Sometimes, composite mode will fail on some images, such as ComfyUI example image, still under invesgating the cause

## Credit & Thanks
- [mattyamonaca/layerdivider](https://github.com/mattyamonaca/layerdivider) - Original implement
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

## My extensions for ComfyUI
- [ComfyUI-Unique3D](https://github.com/jtydhr88/ComfyUI-Unique3D) - ComfyUI Unique3D is custom nodes that running Unique3D into ComfyUI
- [ComfyUI-LayerDivider](https://github.com/jtydhr88/ComfyUI-LayerDivider) - ComfyUI InstantMesh is custom nodes that generating layered psd files inside ComfyUI
- [ComfyUI-InstantMesh](https://github.com/jtydhr88/ComfyUI-InstantMesh) - ComfyUI InstantMesh is custom nodes that running InstantMesh into ComfyUI
- [ComfyUI-ImageMagick](https://github.com/jtydhr88/ComfyUI-ImageMagick) - This extension implements custom nodes that integreated ImageMagick into ComfyUI
- [ComfyUI-Workflow-Encrypt](https://github.com/jtydhr88/ComfyUI-Workflow-Encrypt) - Encrypt your comfyui workflow with key

## My extensions for stable diffusion webui
- [3D Model/pose loader](https://github.com/jtydhr88/sd-3dmodel-loader) A custom extension for AUTOMATIC1111/stable-diffusion-webui that allows you to load your local 3D model/animation inside webui, or edit pose as well, then send screenshot to txt2img or img2img as your ControlNet's reference image.
- [Canvas Editor](https://github.com/jtydhr88/sd-canvas-editor) A custom extension for AUTOMATIC1111/stable-diffusion-webui that integrated a full capability canvas editor which you can use layer, text, image, elements and so on, then send to ControlNet, basing on Polotno.
- [StableStudio Adapter](https://github.com/jtydhr88/sd-webui-StableStudio) A custom extension for AUTOMATIC1111/stable-diffusion-webui to extend rest APIs to do some local operations, using in StableStudio.
- [Txt/Img to 3D Model](https://github.com/jtydhr88/sd-webui-txt-img-to-3d-model) A custom extension for sd-webui that allow you to generate 3D model from txt or image, basing on OpenAI Shap-E.
- [3D Editor](https://github.com/jtydhr88/sd-webui-3d-editor) A custom extension for sd-webui that with 3D modeling features (add/edit basic elements, load your custom model, modify scene and so on), then send screenshot to txt2img or img2img as your ControlNet's reference image, basing on ThreeJS editor.


