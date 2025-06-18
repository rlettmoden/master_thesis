# shallow classifiers for classification of representation
import numpy as np
import torch
from einops import rearrange
from torch import nn


# the class develop forward method should have output of size 1, h, w, c


class MeanPoolTemporalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    # TODO implement masking
    def forward(self, x, doy, mask):
        # inputs of shape # B T H W C
        # x[mask] = -1e6
        temporal_out_dict = {
            "features":None,
            "attention_mask":None
        }
        x, *_ = torch.mean(x, dim=1)  # input dim  n c h w output dim  c h w
        temporal_out_dict["features"] = x
        return temporal_out_dict