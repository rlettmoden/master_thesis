# Inspiration taken from https://github.com/linlei1214/SITS-BERT
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SimpleTemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        k: torch.Tensor,
        mask: None | torch.Tensor = None,
    ):
        """
        input of dimension (b,n,c,h,w) where n is the temporal dimension and will be used as batch dimension as well
        h,w spatial dim and c the spectral dimension and b the batch dimension
        mask should be (n,n) dimension
        """
        # TODO adapt for multi-head attention mechanism
        assert (
            len(q.shape) == 5
        ), f"Wrong dimension of input should be (b,n,c,h,w) but is {q.shape}"

        b, n, c, h, w = q.shape
        # Consider it as a usual transformer case we transform the array so that we end up with arrays where the last
        # two dimensions are things we compare spectral info with temporal
        q = rearrange(q, " b n c h w  -> b h w n c")
        k = rearrange(k, "b n c h w -> b h w c n ")
        # scores
        scores = torch.matmul(q, k) / math.sqrt(c)  # of shape b,h,w,n,n
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e-9)
        p_attn = F.softmax(
            scores, dim=-2
        )  # softmax over the temporal dimension dimension b h w n n
        v = rearrange(v, "b n c h w -> b h w n c")
        Ouput = torch.matmul(p_attn, v)
        output = rearrange(Ouput, " b h w n c-> b n c h w ", n=n, c=c)

        return output, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, encode_value=True
    ):
        """`embed_dim`` will be split
        across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_heads
        self.dropout = dropout
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )  # average_attn_weights=False

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask=None,
        key_padding_mask=None,
    ):
        """"""
        assert (
            len(Q.shape) == 5
        ), f"Wrong dimension of input should be (b,n,c,h,w) but is {Q.shape}"

        b, nq, c, h, w = torch.tensor(Q.shape).to(
            device=Q.device, dtype=torch.int32
        )  # TODO correct

        Q = rearrange(Q, " b n c h w  -> (b h w) n c")
        K = rearrange(K, "b n c h w -> (b h w ) n c ")
        V = rearrange(V, " b n c h w  -> (b h w) n c")
        if key_padding_mask is not None:
            key_padding_mask = torch.unsqueeze(key_padding_mask, dim=1)
            key_padding_mask = torch.repeat_interleave(
                key_padding_mask, repeats=h * w, dim=1
            )
            key_padding_mask = rearrange(key_padding_mask, "b hw n -> (b hw ) n")
        if mask is not None:
            assert (
                len(mask.shape) == 3
            ), f"Wrong attention mask dim should be B,n, not {mask.shape}"

            mask = torch.unsqueeze(mask, -1)
            mask = torch.repeat_interleave(mask, h * w, -1)
            mask = rearrange(mask, " b n1 n2 hw-> (b hw) n1 n2")
            mask = torch.repeat_interleave(
                mask, self.num_head, dim=0
            )  # (b * h * w * head, T, T)

        # my_logger.debug("query tmp {}".format(Q.shape))
        attn_output, attn_output_weights = self.mha(
            Q,
            K,
            V,
            need_weights=True,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            average_attn_weights=False,
        )
        attn_output = rearrange(
            attn_output, "(b h w) n c -> b n c h w ", b=b, h=h, w=w, n=nq, c=c
        )
        attn_output_weights = rearrange(
            attn_output_weights,
            "(b h w) nh nq nv -> nh b h w nq nv ",
            b=b,
            h=h,
            w=w,
        )  # nh is number of head

        return attn_output, attn_output_weights
