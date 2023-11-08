import os, torch, random
import numpy as np
import comfy.utils

from colorsys import rgb_to_hsv, hsv_to_rgb

from kornia.enhance import equalize_clahe
from ._func import pixel_approx, po2, Color, byte
from .isnet import dis_process
from PIL import Image

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
            # size_ratio = min(size) / max(size)
            clip_limit = int(max(8, po2(max(size)) * (clip_limit / 1024)))
            grid_size = int(max(2, clip_limit * grid_ratio))
            print(clip_limit, grid_size)
            # grid_x = max(2, int(clip_limit * grid_ratio) // 2 * 2)
            # grid_y = max(2, int(grid_x * size_ratio) // 2 * 2)
            # if size[0] < size[1]: grid_x, grid_y = grid_y, grid_x
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

        result = comfy.utils.common_upscale(_image, int( width ), int( height ), "area", 'center')
        result = result.movedim(1, -1)
        return (result,)

class ImageScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.01}),
        }
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "TeaNodes/Image"

    def resize(self, image, factor):
        _image = image.movedim(-1, 1)

        height, width = image.shape[1:3]

        result = comfy.utils.common_upscale(_image, int( width * factor ), int( height * factor ), "area", 'center')
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

        height, width = image.shape[1:3]

        _color = tuple(byte(e) for e in color_rgba.RGBA)
        _image = Image.new('RGBA', (width, height), _color)
        _image = np.array(_image.convert('RGBA')).astype(np.float32) / 255.0
        result = torch.from_numpy(_image).unsqueeze(0)

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

# class MaskBG_DIS:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "image": ("IMAGE", ),
#             }
#         }

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "process"
#     CATEGORY = "TeaNodes/Image"

#     def process(self, image):
#         i = 255. * image[-1].numpy()
#         img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

#         img.save("..\__temp__.png", format='png', pnginfo=None, compress_level=4)

#         images_transformed = dis_process("..\__temp__.png")

#         os.remove("..\__temp__.png")

#         _image = Image.fromarray(images_transformed)

#         _image = _image.image_to_tensor()
#         _image.unsqueeze_(0)
#         result = _image.repeat(1,1,1,3)

#         return (result,)

NODE_CLASS_MAPPINGS = {
    "TC_EqualizeCLAHE": EqualizeCLAHE,
    "TC_SizeApproximation": SizeApproximation,
    "TC_ImageResize": ImageResize,
    "TC_ImageScale": ImageScale,
    "TC_ColorFill": ColorFill,
    "TC_RandomColorFill": RandomColorFill,
    "TC_MaskBG_DIS": MaskBG_DIS,
}