from torch import nn


def Conv2dSame(dim_in, dim_out, kernel_size, bias=True):
    pad_left = kernel_size // 2
    pad_right = (pad_left - 1) if (kernel_size % 2) == 0 else pad_left

    return nn.Sequential(
        nn.ZeroPad2d((pad_left, pad_right, pad_left, pad_right)),
        nn.Conv2d(dim_in, dim_out, kernel_size, bias=bias),
    )
