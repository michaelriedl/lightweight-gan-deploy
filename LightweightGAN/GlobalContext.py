from torch import nn, einsum


class GlobalContext(nn.Module):
    def __init__(self, *, chan_in, chan_out):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim=-1)
        out = einsum("b i n, b c n -> b c i", context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)
