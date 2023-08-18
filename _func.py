
from math import ceil, floor, log

debouncer = set()

def po2(value, fill=False):
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
