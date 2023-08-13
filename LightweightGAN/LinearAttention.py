import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange

from .DepthWiseConv2d import DepthWiseConv2d


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, kernel_size=3):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.kernel_size = kernel_size
        self.nonlin = nn.GELU()

        self.to_lin_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_lin_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding=1, bias=False)

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.to_out = nn.Conv2d(inner_dim * 2, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        # linear attention

        lin_q, lin_k, lin_v = (
            self.to_lin_q(fmap),
            *self.to_lin_kv(fmap).chunk(2, dim=1),
        )
        lin_q, lin_k, lin_v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) (x y) c", h=h),
            (lin_q, lin_k, lin_v),
        )

        lin_q = lin_q.softmax(dim=-1)
        lin_k = lin_k.softmax(dim=-2)

        lin_q = lin_q * self.scale

        context = einsum("b n d, b n e -> b d e", lin_k, lin_v)
        lin_out = einsum("b n d, b d e -> b n e", lin_q, context)
        lin_out = rearrange(lin_out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        # conv-like full attention

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1))
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) c x y", h=h), (q, k, v)
        )

        k = F.unfold(k, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        v = F.unfold(v, kernel_size=self.kernel_size, padding=self.kernel_size // 2)

        k, v = map(
            lambda t: rearrange(t, "b (d j) n -> b n j d", d=self.dim_head), (k, v)
        )

        q = rearrange(q, "b c ... -> b (...) c") * self.scale

        sim = einsum("b i d, b i j d -> b i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        attn = sim.softmax(dim=-1)

        full_out = einsum("b i j, b i j d -> b i d", attn, v)
        full_out = rearrange(full_out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        # add outputs of linear attention + conv like full attention

        lin_out = self.nonlin(lin_out)
        out = torch.cat((lin_out, full_out), dim=1)
        return self.to_out(out)
