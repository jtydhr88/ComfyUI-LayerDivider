from PIL import Image
import numpy as np
import torch
import os
import folder_paths
from .ldivider.ld_utils import save_psd, load_masks, divide_folder, load_seg_model
from .ldivider.ld_convertor import pil2cv, cv2pil, df2bgra
from .ldivider.ld_processor import get_base, get_normal_layer, get_composite_layer, get_seg_base
from .ldivider.ld_segment import get_mask_generator, get_masks, show_anns
from pytoshop.enums import BlendMode

comfy_path = os.path.dirname(folder_paths.__file__)

layer_divider_path = f'{comfy_path}/custom_nodes/ComfyUI-LayerDivider'

output_dir = f"{layer_divider_path}/output"
input_dir = f"{layer_divider_path}/input"
model_dir = f"{layer_divider_path}/segment_model"

import uuid

import cv2


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def to_comfy_img(np_img):
    out_imgs = []

    out_imgs.append(HWC3(np_img))

    out_imgs = np.stack(out_imgs)

    out_imgs = torch.from_numpy(out_imgs.astype(np.float32) / 255.)

    return out_imgs


def to_comfy_imgs(np_imgs):
    out_imgs = []

    for np_img in np_imgs:
        out_imgs.append(HWC3(np_img))

    out_imgs = np.stack(out_imgs)

    out_imgs = torch.from_numpy(out_imgs.astype(np.float32) / 255.)

    return out_imgs


def load_seg_model(model_dir):
    folder = model_dir
    file_name = 'sam_vit_h_4b8939.pth'
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    file_path = os.path.join(folder, file_name)
    if not os.path.exists(file_path):
        response = requests.get(url, stream=True)

        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(
                desc=file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)


def generate_layers(input_image, cv_image, df, layer_mode, divide_mode):
    base_image = to_comfy_img(df2bgra(df))
    comfy_image = to_comfy_img(cv_image)

    if layer_mode == "composite":
        base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = (
            get_composite_layer(input_image, df))

        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
            ["base", "screen", "multiply", "subtract", "addition"],
            [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
            output_dir,
            layer_mode,
            divide_mode
        )

        # base_layer_list = [cv2pil(layer) for layer in base_layer_list]

        divide_folder(filename, input_dir, layer_mode)

        base_layer_list = to_comfy_imgs(base_layer_list)
        bright_layer_list = to_comfy_imgs(bright_layer_list)
        shadow_layer_list = to_comfy_imgs(shadow_layer_list)

        return (comfy_image, base_image, base_layer_list,
                bright_layer_list, shadow_layer_list, filename)

    elif layer_mode == "normal":
        base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(input_image, df)
        filename = save_psd(
            input_image,
            [base_layer_list, bright_layer_list, shadow_layer_list],
            ["base", "bright", "shadow"],
            [BlendMode.normal, BlendMode.normal, BlendMode.normal],
            output_dir,
            layer_mode,
            divide_mode
        )

        divide_folder(filename, input_dir, layer_mode)

        return (comfy_image, base_image, to_comfy_imgs(base_layer_list), to_comfy_imgs(bright_layer_list),
                to_comfy_imgs(shadow_layer_list), filename)
    else:
        return None


class LayerDividerLoadImagePath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "LayerDivider"

    RETURN_TYPES = ("IMAGE_PATH",)
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        return (image_path,)


class LayerDividerColorBase:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "loops": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "init_cluster": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "ciede_threshold": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "blur_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "layer_mode": (["composite", "normal"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "base_image", "base", "bright", "shadow", "filepath")

    FUNCTION = "execute"

    # OUTPUT_NODE = False

    CATEGORY = "LayerDivider"

    def execute(self,
                image1,
                loops, init_cluster, ciede_threshold, blur_size, layer_mode):

        # Disable bg remove for now

        split_bg = False
        h_split = -1
        v_split = -1
        n_cluster = -1
        alpha = -1
        th_rate = 0

        img_batch_np = image1.cpu().detach().numpy().__mul__(255.).astype(np.uint8)

        input_image = Image.fromarray(img_batch_np[0])

        image = pil2cv(input_image)

        self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        df = get_base(self.input_image, loops, init_cluster, ciede_threshold, blur_size, h_split, v_split, n_cluster,
                      alpha, th_rate, split_bg, False)

        if layer_mode == "composite":
            base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(
                self.input_image, df)
            filename = save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
                output_dir,
                layer_mode,
                "color_base"
            )
            #base_layer_list = [cv2pil(layer) for layer in base_layer_list]
        elif layer_mode == "normal":
            base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(self.input_image, df)
            filename = save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal],
                output_dir,
                layer_mode,
                "color_base"
            )

        print("filename:" + filename)

        divide_folder(filename, input_dir, layer_mode)

        return (image1,
                to_comfy_img(self.input_image),
                to_comfy_imgs(base_layer_list),
                to_comfy_imgs(bright_layer_list),
                to_comfy_imgs(shadow_layer_list),
                filename)


class LayerDividerLoadMaskGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pred_iou_thresh": ("FLOAT", {
                    "default": 0.8,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider"
                }),
                "stability_score_thresh": ("FLOAT", {
                    "default": 0.8,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider"
                }),
                "min_mask_region_area": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("MASK_GENERATOR",)
    RETURN_NAMES = ("mask_generator",)

    FUNCTION = "execute"

    CATEGORY = "LayerDivider"

    def execute(self, pred_iou_thresh, stability_score_thresh, min_mask_region_area):
        load_seg_model(model_dir)

        mask_generator = get_mask_generator(pred_iou_thresh, stability_score_thresh, min_mask_region_area, model_dir)

        return (mask_generator,)


class LayerDividerSegmentMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "mask_generator": ("MASK_GENERATOR",),
                "area_th": ("INT", {
                    "default": 20000,
                    "min": 1,
                    "max": 100000,
                    "step": 100,
                    "display": "slider"
                }),
                "layer_mode": (["composite", "normal"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "base_image", "base", "bright", "shadow", "filepath")

    FUNCTION = "execute"

    # OUTPUT_NODE = False

    CATEGORY = "LayerDivider"

    def execute(self, image1, mask_generator, area_th, layer_mode):
        img_batch_np = image1.cpu().detach().numpy().__mul__(255.).astype(np.uint8)

        input_image = Image.fromarray(img_batch_np[0])

        masks = get_masks(pil2cv(input_image), mask_generator)

        show_anns(input_image, masks, output_dir)

        input_image.putalpha(255)

        input_image = Image.fromarray(img_batch_np[0])

        image = pil2cv(input_image)

        self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        masks = load_masks(output_dir)

        df = get_seg_base(self.input_image, masks, area_th)

        if layer_mode == "composite":
            base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(
                self.input_image, df)
            filename = save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
                output_dir,
                layer_mode,
                "seg_mask"
            )

        elif layer_mode == "normal":
            base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(self.input_image, df)
            filename = save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal],
                output_dir,
                layer_mode,
                "seg_mask"
            )

        print("filename:" + filename)

        divide_folder(filename, input_dir, layer_mode)

        return (image1,
                to_comfy_img(self.input_image),
                to_comfy_imgs(base_layer_list),
                to_comfy_imgs(bright_layer_list),
                to_comfy_imgs(shadow_layer_list),
                filename)


NODE_CLASS_MAPPINGS = {
    "LayerDivider - Color Base": LayerDividerColorBase,
    "LayerDivider - Load SAM Mask Generator": LayerDividerLoadMaskGenerator,
    "LayerDivider - Segment Mask": LayerDividerSegmentMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerDivider - Color Base": LayerDividerColorBase,
    "LayerDivider - Load SAM Mask Generator": LayerDividerLoadMaskGenerator,
    "LayerDivider - Segment Mask": LayerDividerSegmentMask
}