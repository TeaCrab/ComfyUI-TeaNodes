import random
from colorsys import hsv_to_rgb

import numpy as np
import torch
from kornia.enhance import adjust_gamma, equalize_clahe
from PIL import Image
from torchvision.transforms.functional import center_crop

import comfy.utils
import folder_paths

from ._func import Color, byte, pixel_approx, po2


class CropTo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_src": ("IMAGE",),
                "image_ref": ("IMAGE",),
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
                "image": ("IMAGE",),
                "gamma": (
                    "FLOAT",
                    {"default": 1, "min": 0.0, "max": 100, "step": 0.01},
                ),
                "gain": (
                    "FLOAT",
                    {"default": 1.0, "min": -100, "max": 100, "step": 0.01},
                ),
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
                "image": ("IMAGE",),
                "size": ("TUPLE", {"default": (1024, 1024)}),
                "clip_limit": (
                    "FLOAT",
                    {"default": 64, "min": 0.0, "max": 255, "step": 0.1},
                ),
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
                "square": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 4096, "step": 32},
                ),
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
                "width": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 4096, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 4096, "step": 32},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "TeaNodes/Image"

    def resize(self, image, size, width, height):
        _image = image.movedim(-1, 1)

        if size != (1024, 1024):
            width, height = size

        result = comfy.utils.common_upscale(
            _image, int(width), int(height), "lanczos", "center"
        )
        result = result.movedim(1, -1)
        return (result,)


class ImageScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.01},
                ),
                # "step" of 0.00 breaks the node graph engine...
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "TeaNodes/Image"

    def resize(self, image, factor):
        _image = image.movedim(-1, 1)

        height, width = image.shape[1:3]

        result = comfy.utils.common_upscale(
            _image, int(width * factor), int(height * factor), "lanczos", "center"
        )
        result = result.movedim(1, -1)

        return (result,)


class RandomColorFill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": "#7f7f7fff"}),
                "hue_range": ("FLOAT", {"default": 0.005}),
                "sat_range": ("FLOAT", {"default": 0.005}),
                "val_range": ("FLOAT", {"default": 0.005}),
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
        _image = Image.new("RGBA", (width, height), _color)
        _image = np.array(_image.convert("RGBA")).astype(np.float32) / 255.0
        result = torch.from_numpy(_image).unsqueeze(0)
        print(type(result))
        return (result,)


class ColorFill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": "#7f7f7fff"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill"
    CATEGORY = "TeaNodes/Input"

    def fill(self, image, color):
        if color.startswith("#"):
            _color = color.lstrip("#")
            try:
                color_rgba = tuple(int(_color[i : i + 2], 16) for i in (0, 2, 4, 6))
            except:
                color_rgba = tuple(int(_color[i : i + 2], 16) for i in (0, 2, 4))
            else:
                print(f"Something went wrong here: {color}")
        else:
            _color = color.split(",")
            try:
                _color = tuple(int(e) for e in _color if 255 > int(e) > 0)
            except:
                print(f"Something went wrong here: {color}")
            color_rgba = _color
        color_mode = "RGBA" if len(color_rgba) > 3 else "RGB"

        height, width = image.shape[1:3]

        _image = Image.new("RGBA", (width, height), color_rgba)
        _image = np.array(_image.convert(color_mode)).astype(np.float32) / 255.0
        result = torch.from_numpy(_image).unsqueeze(0)

        return (result,)


import re
import os

RETYPE = type(re.compile(''))

class Pool:
    def __init__(self, content):
        self.regex = re.compile('', re.I)
        self.content = content
        self.filtered = []

    def sieve(self, pattern):
        if self.regex.pattern!=pattern or not self.filtered:
            try: self.regex = re.compile(pattern, re.I)
            except Exception: return None
            self.filtered = [name for name in self.content if self.regex.search(name)]
        return True

class LoraPool(Pool):
    name = 'Loras'
    loaded = dict()
    history = dict()
    def __init__(self):
        super().__init__([os.path.join(p, e) for p, _, f in os.walk('ComfyUI\\models\\loras', followlinks=True) for e in f if e.endswith('.safetensors')])

class ModelPool(Pool):
    name = "Models"
    loaded = dict()
    history = dict()
    def __init__(self):
        super().__init__([os.path.join(p, e) for p, _, f in os.walk('ComfyUI\\models\\checkpoints', followlinks=True) for e in f if e.endswith('.safetensors')])


class Randomizer():
    def __init__(self, pool):
        self.path = ''
        self.pool = pool
        self.loop = 0
        self.every = 1
        self.error = ''
        self.pause = False

    def out(self):
        if not self.pause: self.loop += 1
        warn = ': '.join(e for e in (
            f"{self.error}" if self.error else '',
            f"No {self.pool.name} Found! Using the last available.\n" if not self.pool.filtered else ''
        ) if e)
        return '\n'.join((
            f"{warn}Loop-<{self.loop}/{self.every}>: {self.path}",
            "\nHistory (has generated with):",
            "\n".join(f"{v:>4d}: {os.path.basename(k)}" for k, v in sorted(self.pool.history.items(), key=lambda e: e[-1], reverse=True) if k in self.pool.filtered),
            "\nPool (hasn't generated with):",
            "\n".join(os.path.basename(e) for e in self.pool.filtered if e not in self.pool.history),
        ))

    def yet(self):
        self.loop %= self.every
        if not self.path: self.path = random.choice(self.pool.content)
        return self.pause or not self.loop

    def run(self, pattern, force=False):
        self.loop %= self.every
        # Execute the randomization only after every # generations
        if not force and (self.pause or self.loop != 0): return self.path
        if self.pool.sieve(pattern) is None:
            self.error = "Invalid Pattern"
        else:
            if self.path not in self.pool.history: self.pool.history[self.path] = 0
            self.pool.history[self.path] += 1
        if self.pool.filtered: self.path = random.choice(self.pool.filtered)

    def get(self):
        return self.pool.loaded[self.path]

    def put(self, content):
        self.pool.loaded[self.path] = content


class RandomLora:
    def __init__(self):
        self.randomizer = Randomizer(LoraPool())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "pattern": ("STRING", {"default": "", "multiline": True, "tooltip": "Regular Expression"}),
                "every": ("INT", {"default": 7, "min": 1, "max": 99, "tooltip": "Change only takes effect every N generations."}),
                "pause": ("BOOLEAN", {"default": False, "tooltip": "Pause the randomization and counting, keep generating with current lora."}),
                "skip": ("BOOLEAN", {"default": False, "tooltip": "Skip curent model."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "RESULT")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.", "List of lora names that matches the pattern.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "Randomly apply a different Lora that matches the search pattern after every N generations."
    SEARCH_ALIASES = ["lora", "load lora", "apply lora", "lora loader", "lora model"]

    def try_get(self, model, clip, strength_model, strength_clip):
        try:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, self.randomizer.get(), strength_model, strength_clip)
            return (model_lora, clip_lora, self.randomizer.out())
        except:
            self.randomizer.put(comfy.utils.load_torch_file(self.randomizer.path))
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, self.randomizer.get(), strength_model, strength_clip)
            return (model_lora, clip_lora, self.randomizer.out())

    def load_lora(self, model, clip, pattern, every, pause, skip, strength_model, strength_clip, **kwargs):
        self.randomizer.every = every
        self.randomizer.pause = pause
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        if not skip and not self.randomizer.yet():
            return self.try_get(model, clip, strength_clip, strength_model)
        else:
            self.randomizer.run(pattern, skip)
            return self.try_get(model, clip, strength_clip, strength_model)

class RandomModel:
    def __init__(self):
        self.randomizer = Randomizer(ModelPool())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pattern": ("STRING", {"default": "", "multiline": True, "tooltip": "Regular Expression"}),
                "every": ("INT", {"default": 7, "min": 1, "max": 99, "tooltip": "Change only takes effect every N generations."}),
                "pause": ("BOOLEAN", {"default": False, "tooltip": "Pause the randomization and counting, keep generating with current model."}),
                "skip": ("BOOLEAN", {"default": False, "tooltip": "Skip curent model."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "RESULT")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.",
                       "List of model names that matches the pattern.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"
    DESCRIPTION = "Randomly switch to a different Model which matches the search pattern after every N generations."
    SEARCH_ALIASES = ["load model", "checkpoint", "model loader", "load checkpoint", "ckpt", "model"]

    def try_get(self):
        try:
            return *self.randomizer.get(), self.randomizer.out()
        except:
            self.randomizer.put(comfy.sd.load_checkpoint_guess_config(self.randomizer.path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))[:3])
            return *self.randomizer.get(), self.randomizer.out()

    def load_checkpoint(self, pattern, every, pause, skip, **kwargs):
        self.randomizer.every = every
        self.randomizer.pause = pause
        if not skip and not self.randomizer.yet():
            return self.try_get()
        else:
            self.randomizer.run(pattern, skip)
            return self.try_get()


NODE_CLASS_MAPPINGS = {
    "TC_CropTo": CropTo,
    "TC_KorniaGamma": KorniaGamma,
    "TC_EqualizeCLAHE": EqualizeCLAHE,
    "TC_SizeApproximation": SizeApproximation,
    "TC_ImageResize": ImageResize,
    "TC_ImageScale": ImageScale,
    "TC_ColorFill": ColorFill,
    "TC_RandomColorFill": RandomColorFill,
    "TC_RandomLora": RandomLora,
    "TC_RandomModel": RandomModel,
}
