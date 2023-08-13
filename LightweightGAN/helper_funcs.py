import math
import torch
from math import log2


def exists(val):
    return val is not None


def is_power_of_two(val):
    return log2(val).is_integer()


def default(val, d):
    return val if exists(val) else d


def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))


def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part : (i + 1) * c_part, x, y] = coor_value

    return dct_weights
