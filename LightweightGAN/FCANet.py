from torch import nn
from einops import reduce
from .helper_funcs import get_dct_weights


class FCANet(nn.Module):
    def __init__(self, *, chan_in, chan_out, reduction=4, width):
        super().__init__()

        freq_w, freq_h = ([0] * 8), list(
            range(8)
        )  # in paper, it seems 16 frequencies was ideal
        dct_weights = get_dct_weights(
            width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w]
        )
        self.register_buffer("dct_weights", dct_weights)

        chan_intermediate = max(3, chan_out // reduction)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = reduce(
            x * self.dct_weights, "b c (h h1) (w w1) -> b c h1 w1", "sum", h1=1, w1=1
        )
        return self.net(x)
