import regex_spm, re, code
import numpy as np
from math import ceil, floor, log
from enum import Enum

from colorsys import (
    rgb_to_hls,
    rgb_to_hsv,
    rgb_to_yiq,
    hls_to_rgb,
    hsv_to_rgb,
    yiq_to_rgb,
)

debouncer = set()

def po2(value, fill=False) -> int:
    func = ceil if fill else floor
    return pow(2, func(log(value)/log(2)))

def pixel_approx(primary, secondary, total=1024, ratio=1.0,
                 threshold=0.125, _inc=1024, calculate_inc=True) -> tuple:
    global debouncer
    if calculate_inc:
        debouncer.clear()
        total *= total
        ratio = secondary / primary
        primary = po2(primary)
        secondary = primary * ratio
        _inc = primary / 4
    elif total * 3 > int(primary * secondary) > total * 0.333:
        if abs(_inc) >= 64: _inc /= 2

    # print(f"Action: {_inc}")

    count = round(primary) * round(secondary)
    # print(round(primary), round(secondary), count)

    _recurse = False
    if count > total:
        _inc = abs(_inc) * -1
    elif count < total:
        _inc = abs(_inc)

    if not (total * (1+threshold) > count > total * (1-threshold)) and total not in debouncer:
        debouncer.add(count)
        _recurse = True

    if _recurse:
        primary += _inc
        secondary = primary * ratio
        primary, secondary = pixel_approx(primary, secondary, total, ratio, threshold, _inc, False)

    return (int(primary), int(secondary))

class RE:
    HEX = re.compile(r"^\#[\da-fA-F]{1,8}$", re.IGNORECASE)
    RGB = re.compile(r"^rgba?\s+(?:\d{1,3}[, ](?=\d)|\d{1,3}(?=$)){1,4}$", re.IGNORECASE)
    HSV = re.compile(r"^hsva?\s+(?:\d{1,3}[, ](?=\d)|\d{1,3}(?=$)){1,4}$", re.IGNORECASE)
    HLS = re.compile(r"^hlsa?\s+(?:\d{1,3}[, ](?=\d)|\d{1,3}(?=$)){1,4}$", re.IGNORECASE)
    YIQ = re.compile(r"^yiqa?\s+(?:\d{1,3}[, ](?=\d)|\d{1,3}(?=$)){1,4}$", re.IGNORECASE)

class color_format_constant(Enum):
    HEX,RGB,HSV,HLS,YIQ = range(5)

C = color_format_constant

def partition(array, step=2, indices=None):
    if not indices: indices = np.array(range(len(array) // step))*2
    return [array[i:i+step] for i in tuple(indices)]

def ispowof(value:int, exp=2):
    return ceil(log(value, 2)) == floor(log(value, 2))

def iclamp(value:int, low=0, high=255) -> int:
    return int(max(min(value, high), low))

def fclamp(value:float, low=0.0, high=1.0) -> float:
    return float(max(min(value, high), low))

def baser(value:int) -> int:
    '''
    Convert a NYBL integer (0~16) into a BYTE integer (0~255)
    Absolutely no error proofing whatsoever.
    '''
    return int(value * 16 + value)

def byte(value:float) -> int:
    '''
    Convert FLOAT scaler value to BYTE integer (0~255)
    Absolutely no error proofing whatsoever.
    '''
    return iclamp(round(fclamp(value) * 255))

def scale(value: int) -> float:
    '''
    Convert a BYTE integer (0~255) to a FLOAT scaler (0.0~1.0)
    Absolutely no error proofing whatsoever.
    '''
    return iclamp(value) / 255

def normalize(nparray, low=0.0, high=1.0):
    return np.array([(((i - min(nparray)) * (high - low)) / (max(nparray) - min(nparray))) + low for i in nparray])

class Color():
    default = C.HEX
    alfault = False
    byfault = False
    def __init__(self, color="#0", style=None):
        if isinstance(color, type(self)):
            self.RGBA = color.RGBA
        elif isinstance(color, type(tuple)):
            match len(color):
                case 1:
                    self.RGBA = color * 4
                case 2:
                    self.RGBA = (color[0],) * 3 + (color[1],)
                case 3:
                    self.RGBA = color + (1.0,)
                case 4:
                    self.RGBA = color
                case _:
                    self.RGBA = color[:4]
        elif isinstance(color, str):
            match regex_spm.fullmatch_in(color):
                case RE.HEX:
                    _color = color.lstrip('#')
                    if not _color: print(f"Unhandled Case - {color}"); self.RGBA = self.fallback()
                    match len(_color):
                        case 1:
                            self.RGBA = (scale(baser(int(_color, 16))),) * 3 + (1.0,)
                        case 2:
                            self.RGBA = (scale(int(_color, 16)),) * 3 + (1.0,)
                        case 3:
                            self.RGBA = tuple(scale(baser(int(e, 16))) for e in _color) + (1.0,)
                        case 4:
                            self.RGBA = tuple(scale(baser(int(e, 16))) for e in _color)
                        case 6:
                            # _Color needs to be sliced in 2 char
                            self.RGBA = tuple(scale(int(e, 16)) for e in partition(_color)) + (1.0,)
                        case 8:
                            # _Color needs to be sliced in 2 char
                            self.RGBA = tuple(scale(int(e, 16)) for e in partition(_color))
                        case _:
                            print(f"Unhandled Case - {color}")
                            self.RGBA = self.fallback()
                case RE.RGB:
                    _color = re.split(r" |,", color)
                    print(_color)
                    match len(_color[1:]):
                        case 1:
                            self.RGBA = tuple(scale(_color[1])) * 3 + (1.0,)
                        case 2:
                            self.RGBA = tuple(scale(_color[1])) * 3 + tuple(scale(_color[2]))
                        case 3:
                            self.RGBA = tuple(scale(int(e)) for e in _color[1:]) + (1.0,)
                        case 4:
                            self.RGBA = tuple(scale(int(e)) for e in _color[1:])
                        case _:
                            print(f"Unhandled Case - {color}")
                            self.RGBA = self.fallback()
                case _:
                    print(f"Unhandled Case - {color}")
                    return None
        else:
            print(f"Unhandled Case - {color}")
            return None

    def __str__(self, form=default, alpha=alfault, byte=byfault):
        _result = None
        match form:
            case C.RGB:
                _result = self.RGBA
            case C.HSV:
                _result = self.hsv()
            case C.HLS:
                _result = self.hls()
            case C.YIQ:
                _result = self.yiq()
            case _:
                return self.hex() if alpha else self.hex()[:-2]
        if byte: _result = (f"{byte(e):3d}" for e in _result)
        else: _result = (f"{round(e, 3):.3f}" for e in _result)
        if alpha: _result = _result[:3]
        return f'({",".join(_result)})'

    def __repr__(self, form=default):
        match form:
            case C.RGB:
                return self.RGBA
            case C.HSV:
                return self.hsv()
            case C.HLS:
                return self.hls()
            case C.YIQ:
                return self.yiq()
            case _:
                return self.hex()

    def fallback(self):
        try: return self.RGBA
        except: return (0.0, 0.0, 0.0, 1.0)

    def hex(self):
        R, G, B, A = (byte(e) for e in self.RGBA)
        return f"#{R:02x}{G:02x}{B:02x}{A:02x}"

    def hls(self):
        R, G, B, A = self.RGBA
        return rgb_to_hls(R, G, B) + (A,)

    def hsv(self):
        R, G, B, A = self.RGBA
        return rgb_to_hsv(R, G, B) + (A,)

    def yiq(self):
        R, G, B, A = self.RGBA
        return rgb_to_yiq(R, G, B) + (A,)