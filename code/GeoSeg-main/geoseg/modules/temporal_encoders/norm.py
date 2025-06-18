from typing import Union

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Size, Tensor

_shape_t = Union[int, list[int], Size]


class AdaptedLayerNorm(nn.LayerNorm):
    """Adapt Layer Norm to our issue normalize over the channel dimension only"""

    def __init__(self, normalized_shape: _shape_t, change_shape=True):
        super().__init__(normalized_shape)
        self.change_shape = change_shape

    def forward(self, input: Tensor) -> Tensor:
        # input dim (b,n,c,h,w)
        b, n, c, h, w = input.shape
        if self.change_shape:  # set the channel dimensioon  the last one
            x = rearrange(input, "b n c h w -> b n h w c")
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = rearrange(x, " b n h w c -> b n c h w ", c=c)
        else:  # In this case normalization is over c,h,w
            x = input
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x
