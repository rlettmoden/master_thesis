"""
Adapted from Lightweight Temporal Attention Encoder module

Credits:
https://github.com/VSainteuf/lightweight-temporal-attention-pytorch

MIT License

Copyright (c) 2020 VSainteuf (Vivien Sainte Fare Garnot)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""


import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # my_logger.info("query shape {} key shape {} value shape {} ".format(q.unsqueeze(1).shape,k.transpose(1, 2).shape,v.shape))
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        # my_logger.info("attn shape {}".format(attn.shape))
        if mask is not None:
            attn = attn.masked_fill(
                mask.type_as(v).bool(), 1e-9
            )  # mask attention matrix so that the zero padded elements are
            # not taken into account during matmul
        attn = self.softmax(attn)
        # my_logger.info("Attn after mask {}".format(attn.shape))
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn
