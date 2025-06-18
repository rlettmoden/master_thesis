# Taken from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py

import math
import torch
from einops import rearrange
from torch import nn as nn



class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        self.denom = self.denom.to(batch_positions.device)
        # if not self.updated_location:
        #     # self.denom = self.denom.to(batch_positions.device)
        #     # self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table

class PositionalTemporalEncoding(nn.Module):
    """
    Modified DOY (Day of year) encoding with group encoding using sine and cosine
    """

    def __init__(self, d_model, max_len=3000, T=10000, num_groups=3):
        super().__init__()
        self.out_features = d_model
        self.num_groups = num_groups
        
        # Compute the positional encodings for DOY
        doy_dim = d_model // 2
        pe_doy = torch.zeros(max_len + 1, doy_dim).float()
        pe_doy.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, doy_dim, 2).float() * -(math.log(T) / doy_dim)).exp()

        pe_doy[1:, 0::2] = torch.sin(position * div_term)
        pe_doy[1:, 1::2] = torch.cos(position * div_term)
        pe_doy = torch.unsqueeze(pe_doy, 0)

        self.register_buffer("pe_doy", pe_doy)

        # Compute the group encodings using sine and cosine
        group_dim = d_model - doy_dim
        pe_group = torch.zeros(num_groups, group_dim).float()
        pe_group.require_grad = False

        group_positions = torch.arange(num_groups).float().unsqueeze(1)
        group_div_term = (torch.arange(0, group_dim, 2).float() * -(math.log(T) / group_dim)).exp()

        pe_group[:, 0::2] = torch.sin(group_positions * group_div_term)
        pe_group[:, 1::2] = torch.cos(group_positions * group_div_term)

        self.register_buffer("pe_group", pe_group)

    def forward(self, doy, group):
        b, *_ = doy.shape
        
        # DOY encoding
        pe_doy_batch = torch.repeat_interleave(self.pe_doy, repeats=b, dim=0)
        doy = doy.to(torch.long)
        res_doy = torch.stack([pe_doy_batch[b, i, :] for b, i in enumerate(doy)])

        # Group encoding
        group = group.to(torch.long)
        res_group = self.pe_group[group]

        # Combine DOY and group encodings
        res = torch.cat([res_doy, res_group], dim=-1)
        res = rearrange(res, "b n c -> b n c 1 1")
        
        return res

class PositionalEncoding(nn.Module):
    """DOY (Day of year type of encoding taken from
    https://github.com/linlei1214/SITS-BERT/blob/master/code/model/embedding/position.py
    """

    def __init__(self, d_model, max_len=3000, T=10000):
        super().__init__()
        self.out_features = d_model
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len + 1, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(T) / d_model)
        ).exp()  # [d_model/2,]

        pe[1:, 0::2] = torch.sin(
            position * div_term
        )  # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(
            position * div_term
        )  # broadcasting to [max_len, d_model/2]
        pe = torch.unsqueeze(pe, 0)

        self.register_buffer("pe", pe)

    def forward(self, doy):
        # print("doy {} pe {}".format(doy.shape, self.pe.shape))

        b, *_ = doy.shape
        pe_batch = torch.repeat_interleave(self.pe, repeats=b, dim=0)
        doy = doy.to(torch.long)
        res = torch.stack([pe_batch[b, i, :] for b, i in enumerate(doy)])
        res = rearrange(res, " b n c-> b n c 1 1")
        return res