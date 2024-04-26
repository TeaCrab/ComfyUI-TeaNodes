import torch, random
from torchvision.transforms.functional import center_crop
import numpy as np
import comfy.utils

from colorsys import hsv_to_rgb

from kornia.enhance import equalize_clahe, adjust_gamma, add_weighted
from ._func import pixel_approx, po2, Color, byte
from PIL import Image

class CropTo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_src": ("IMAGE", ),
                "image_ref": ("IMAGE", ),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "weightedadd"
    CATEGORY = "TeaNodes/Image"

    def weightedadd(self, image_src, image_ref):
        R = image_ref.movedim(-1, 1)
        S = image_src.movedim(-1, 1)

        _image = center_crop(S, R.shape[2:])
        result = _image.movedim(1, -1)

        return (result,)

class KorniaGamma:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "gamma": ("FLOAT", {"default": 1, "min": 0.0, "max": 100, "step": 0.01}),
                "gain": ("FLOAT", {"default": 1.0, "min": -100, "max": 100, "step": 0.01}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gamma"
    CATEGORY = "TeaNodes/Image"

    def gamma(self, image, gamma, gain):
        _image = image.movedim(-1, 1)
        _image = adjust_gamma(_image, gamma, gain)
        result = _image.movedim(1, -1)

        return (result,)

class EqualizeCLAHE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "size": ("TUPLE", {"default": (1024, 1024)}),
                "clip_limit": ("FLOAT", {"default": 64, "min": 0.0, "max": 255, "step": 0.1}),
                "grid_size": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "equalize"
    CATEGORY = "TeaNodes/Image"

    def equalize(self, image, size, clip_limit, grid_size):
        _image = image.movedim(-1, 1)
        if size != (1024, 1024):
            grid_ratio = grid_size / clip_limit
            clip_limit = int(max(8, po2(max(size)) * (clip_limit / 1024)))
            grid_size = int(max(2, clip_limit * grid_ratio))
            print(clip_limit, grid_size)

        _image = equalize_clahe(_image, float(clip_limit), (grid_size, grid_size))
        result = _image.movedim(1, -1)

        return (result,)

class SizeApproximation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "square": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32}),
        }
    }

    RETURN_TYPES = ("TUPLE", "INT", "INT")
    FUNCTION = "calculate"
    CATEGORY = "TeaNodes/Image"

    def calculate(self, image, square):
        _image = image.movedim(-1, 1)
        height, width = image.shape[1:3]

        # print(width, height)

        if width >= height:
            width, height = pixel_approx(width, height, square)
        else:
            height, width = pixel_approx(height, width, square)
        return ((width, height), width, height)

class ImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("TUPLE", {"default": (1024, 1024)}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32}),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "TeaNodes/Image"

    def resize(self, image, size, width, height):
        _image = image.movedim(-1, 1)

        if size != (1024, 1024): width, height = size

        result = comfy.utils.common_upscale(_image, int( width ), int( height ), 'lanczos', 'center')
        result = result.movedim(1, -1)
        return (result,)

class ImageScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.01}),
                # "step" of 0.00 breaks the node graph engine...
        }
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "TeaNodes/Image"

    def resize(self, image, factor):
        _image = image.movedim(-1, 1)

        height, width = image.shape[1:3]

        result = comfy.utils.common_upscale(_image, int(width * factor), int(height * factor), 'lanczos', 'center')
        result = result.movedim(1, -1)

        return (result,)

class RandomColorFill():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": '#7f7f7fff'}),
                "hue_range": ("FLOAT", {"default": 0.005}),
                "sat_range": ("FLOAT", {"default": 0.005}),
                "val_range": ("FLOAT", {"default": 0.005})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "randomize"
    CATEGORY = "TeaNodes/Input"

    def randomize(self, image, color, hue_range, sat_range, val_range):
        h, s, v, _ = Color(color).hsv()
        h += hue_range * (random.random() - 0.5)
        s += sat_range * (random.random() - 0.5)
        v += val_range * (random.random() - 0.5)

        color_rgba = Color()
        color_rgba.RGBA = hsv_to_rgb(h, s, v) + (1.0,)
        print(f"Random Color:\t{color_rgba.hex()}")

        print(type(image))
        height, width = image.shape[1:3]

        _color = tuple(byte(e) for e in color_rgba.RGBA)
        _image = Image.new('RGBA', (width, height), _color)
        _image = np.array(_image.convert('RGBA')).astype(np.float32) / 255.0
        result = torch.from_numpy(_image).unsqueeze(0)
        print(type(result))
        return (result,)

class ColorFill():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": '#7f7f7fff'}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill"
    CATEGORY = "TeaNodes/Input"

    def fill(self, image, color):
        if color.startswith('#'):
            _color = color.lstrip('#')
            try: color_rgba = tuple(int(_color[i:i+2], 16) for i in (0, 2, 4, 6))
            except: color_rgba = tuple(int(_color[i:i+2], 16) for i in (0, 2, 4))
            else: print(f"Something went wrong here: {color}")
        else:
            _color = color.split(',')
            try: _color = tuple(int(e) for e in _color if 255>int(e)>0)
            except: print(f"Something went wrong here: {color}")
            color_rgba = _color
        color_mode = 'RGBA' if len(color_rgba) > 3 else 'RGB'

        height, width = image.shape[1:3]

        _image = Image.new('RGBA', (width, height), color_rgba)
        _image = np.array(_image.convert(color_mode)).astype(np.float32) / 255.0
        result = torch.from_numpy(_image).unsqueeze(0)

        return (result,)

NODE_CLASS_MAPPINGS = {
    "TC_CropTo": CropTo,
    "TC_KorniaGamma": KorniaGamma,
    "TC_EqualizeCLAHE": EqualizeCLAHE,
    "TC_SizeApproximation": SizeApproximation,
    "TC_ImageResize": ImageResize,
    "TC_ImageScale": ImageScale,
    "TC_ColorFill": ColorFill,
    "TC_RandomColorFill": RandomColorFill,
}