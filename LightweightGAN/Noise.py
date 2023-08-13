import torch
from torch import nn
from .helper_funcs import exists


class Noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        b, _, h, w, device = *x.shape, x.device

        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device=device)

        return x + self.weight * noise
