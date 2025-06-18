from typing import Literal

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, SimpleTemporalAttention
from .convolutionalblock import ConvBlock
from .norm import AdaptedLayerNorm


def get_subsequent_mask(temporal_size):
    """For masking out the subsequent info."""
    len_s = temporal_size
    # print('tempsize',len_s)
    subsequent_mask = (
        1 - torch.triu(torch.ones((1, len_s, len_s)), diagonal=1)
    ).bool()  # device=temporal_size.device #TODO see what device do ...
    return subsequent_mask


class EncoderTransformerLayer(nn.Module):
    """Transformer encoder layer"""

    def __init__(
        self,
        d_model: int,
        d_in: int,
        nhead: int = 1,
        norm_first: bool = False,
        input_depth=13,
        block_name: Literal["pff", "se", "basicconv"] = "se",
        dropout=0.1,
        attn_dropout: float = 0,
        encode_value=True,
    ):
        super().__init__()
        if nhead >= 1:
            self.attn = MultiHeadAttention(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=attn_dropout,
                encode_value=encode_value,
            )
        else:
            self.attn = SimpleTemporalAttention()
        self.conv_block = ConvBlock(
            inplanes=d_model,
            planes=d_in,
            block_name=block_name,
            dropout=dropout,
        )
        self.norm = AdaptedLayerNorm(d_model, change_shape=True)
        self.norm2 = AdaptedLayerNorm(d_in, change_shape=True)
        self.norm_first = norm_first
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, src_mask=None, key_padding_mask=None):
        b, n, c, h, w = input.shape
        # input is (b,n,c,h,w)
        if self.norm_first:
            x = input

            output, enc_slf_attn = self._sa_block(
                self.norm(input),
                mask=src_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + output
            x = x + self._conv_block(self.norm(x))
        else:
            x = input
            output, enc_slf_attn = self._sa_block(
                self.norm(input),
                mask=src_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm(x + output)
            x = self.norm(x + self._conv_block(x))

        return x, enc_slf_attn

    def _sa_block(self, x, mask, key_padding_mask):
        x, p_attn = self.attn(x, x, x, mask=mask, key_padding_mask=key_padding_mask)
        return self.dropout(x), p_attn

    def _conv_block(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    """Inspired from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob
    /132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Models.py#L48"""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_in: int,
        dropout=0.1,
        block_name: Literal["se", "pff", "basicconv"] = "se",
        norm_first: bool = False,
        nhead: int = 1,
        attn_dropout: float = 0,
        encode_value: bool = True,
    ):
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [
                EncoderTransformerLayer(
                    d_model=d_model,
                    d_in=d_in,
                    nhead=nhead,
                    norm_first=norm_first,
                    block_name=block_name,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    encode_value=encode_value,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        input: torch.Tensor,
        return_attns: bool = False,
        key_padding_mask: None | torch.Tensor = None,
        src_mask: None | torch.Tensor = None,
    ):
        enc_slf_attn_list = []
        # enc_output = input
        enc_output = self.dropout(input)
        # some values are not there (b,n enc_output=self.layer_norm(ind_modelput) #not sure layer norm is adapted to our
        # issue ...
        for i, enc_layer in enumerate(self.layer_stack):
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                key_padding_mask=key_padding_mask,
                src_mask=src_mask,
            )
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        temporal_out_dict = {
            "features":enc_output,
            "attention_mask":enc_slf_attn_list,
        }
        return temporal_out_dict
