# shallow classifiers for classification of representation
import numpy as np
import torch
from einops import rearrange
from torch import nn

from .attention import MultiHeadAttention
from .ltae import ScaledDotProductAttention
from .lightweight_convolution import LightweightConv1d
# the class develop forward method should have output of size 1, h, w, c


class MaxPoolDecoding(nn.Module):
    def __init__(self, inplanes, num_classes, last_activ=None):
        super().__init__()
        self.linear = nn.Linear(inplanes, num_classes)

    def forward(self, x, mask):
        # print("classif")
        # x is b n c h w
        x[mask] = -1e6
        x, *_ = torch.max(x, dim=1)  # input dim  n c h w output dim  c h w

        x = rearrange(x, "b c h w -> b h w c")

        x = self.linear(x)  # 1 h w c perfect for one hot

        return x


class MasterQueryDecoding(nn.Module):
    def __init__(self, inplanes, nhead=1, attn_dropout=0, last_activ=None):
        super().__init__()
        # self.linear = nn.Linear(inplanes, num_classes)
        # print("d_model {}".format(d_model))
        self.attn = MultiHeadMasterQueryAttention(
            n_head=nhead, d_k=inplanes // nhead, d_in=inplanes
        )  #
        self.n_head = nhead

    def forward(self, x, key_padding_mask, return_attns=False):
        assert (
            len(x.shape) == 5
        ), "Wrong input size, should be 5, batch dimension is required {}".format(
            x.shape
        )
        b, n, c, h, w = x.shape
        #        x = rearrange(x, 'b  n c h w -> 1 n c h w')
        x = rearrange(x, " b n c h w -> (b h w ) n c")
        # print("INPUT", x.shape)
        # print("MAKS", key_padding_mask.shape)
        if key_padding_mask is not None:
            # my_logger.info("init padding mask shape {}".format(key_padding_mask.shape))
            key_padding_mask = torch.unsqueeze(key_padding_mask, dim=1)
            key_padding_mask = torch.repeat_interleave(
                key_padding_mask, repeats=h * w, dim=1
            )
            key_padding_mask = rearrange(key_padding_mask, "b hw n -> (b hw ) n 1")
            key_padding_mask = torch.repeat_interleave(
                key_padding_mask, repeats=self.n_head, dim=2
            )
            key_padding_mask = rearrange(key_padding_mask, "bs n c -> (bs c) 1 n ")
        # print("KEY MASK", key_padding_mask.shape)
        # print("query", torch.mean(x, dim=1, keepdim=True).shape)
        # my_logger.info("before attn shape {}".format(x.shape))
        x, p_attn = self.attn(
            x, x, x, key_padding_mask
        )  # torch.mean(x, dim=1, keepdim=True)
        # my_logger.info("after attn {}".format(x.shape))
        x = rearrange(x, "H bhw dk-> bhw (H dk)")
        #        x = x.permute(1, 0, 2).contiguous().view(h * w, -1)  #merge heads
        x = rearrange(x, " (b h w )  c -> b c h w ", b=b, c=c, h=h, w=w)
        #        x = rearrange(x, ' 1 c h w -> 1 h w c')
        # x = self.linear(x)
        # print("out classif", x.shape)

        temporal_out_dict = {
            "features":x,
            "attention_mask":p_attn,
        }
        return temporal_out_dict
    
class GLTaeDecoding(nn.Module):
    def __init__(self, inplanes, nhead=1, attn_dropout=0, last_activ=None):
        super().__init__()
        # self.linear = nn.Linear(inplanes, num_classes)
        # print("d_model {}".format(d_model))
        self.attn = MultiHeadMasterQueryAttention(
            # n_head=nhead, d_k=inplanes // nhead, d_in=inplanes
            n_head=nhead, d_k=inplanes // nhead, d_in=inplanes // 2
        )  #
        self.lconv = LightweightConv1d(input_size=inplanes // 2, kernel_size=3, weight_softmax=True, num_heads=nhead, padding=1)
        self.n_head = nhead

    def forward(self, x, key_padding_mask, return_attns=False):
        assert (
            len(x.shape) == 5
        ), "Wrong input size, should be 5, batch dimension is required {}".format(
            x.shape
        )
        b, n, c, h, w = x.shape
        #        x = rearrange(x, 'b  n c h w -> 1 n c h w')
        x = rearrange(x, " b n c h w -> (b h w ) n c")
        # print("INPUT", x.shape)
        # print("MAKS", key_padding_mask.shape)
        if key_padding_mask is not None:
            # my_logger.info("init padding mask shape {}".format(key_padding_mask.shape))
            key_padding_mask = torch.unsqueeze(key_padding_mask, dim=1)
            key_padding_mask = torch.repeat_interleave(
                key_padding_mask, repeats=h * w, dim=1
            )
            key_padding_mask = rearrange(key_padding_mask, "b hw n -> (b hw ) n 1")
            key_padding_mask = torch.repeat_interleave(
                key_padding_mask, repeats=self.n_head, dim=2
            )
            key_padding_mask = rearrange(key_padding_mask, "bs n c -> (bs c) 1 n ")
        
        x_ltae = x[:, :, :x.shape[2]//2]
        x_convs = x[:, :, x.shape[2]//2:]
        # print("KEY MASK", key_padding_mask.shape)
        # print("query", torch.mean(x, dim=1, keepdim=True).shape)
        # my_logger.info("before attn shape {}".format(x.shape))
        x_ltae, p_attn = self.attn(
            x_ltae, x_ltae, x_ltae, key_padding_mask
        )  # torch.mean(x, dim=1, keepdim=True)
        x_ltae = rearrange(x_ltae, "H bhw dk-> bhw (H dk)")

        # requires input of B x C x T
        # x_convs = self.lconv(x_convs.reshape(seq_len, sz_b * h * w, out.shape[2]//2))
        x_convs = self.lconv(x_convs.reshape(b * h * w, x.shape[2]//2, n)).permute(2, 0, 1) # B T C -> B C T -> T B C
        x_convs = torch.mean(x_convs, dim=0) # average pool T dimension

        x = torch.concat([x_ltae, x_convs], dim=1)

        # my_logger.info("after attn {}".format(x.shape))
        #        x = x.permute(1, 0, 2).contiguous().view(h * w, -1)  #merge heads
        x = rearrange(x, " (b h w )  c -> b c h w ", b=b, c=c, h=h, w=w)
        #        x = rearrange(x, ' 1 c h w -> 1 h w c')
        # x = self.linear(x)
        # print("out classif", x.shape)

        temporal_out_dict = {
            "features":x,
            "attention_mask":p_attn,
        }
        return temporal_out_dict


class AttnDecodingMaxpool(nn.Module):
    def __init__(self, inplanes, num_classes, nhead=1, attn_dropout=0, last_activ=None):
        super().__init__()
        self.linear = nn.Linear(inplanes, num_classes)
        # print("d_model {}".format(d_model))
        self.attn = MultiHeadAttention(
            num_heads=nhead,
            embed_dim=inplanes,
            dropout=attn_dropout,
            encode_value=False,
        )

    def forward(self, x, return_attns=False):
        assert (
            len(x.shape) == 4
        ), "Wrong input size, should be 4, no batch dimension is required {}".format(
            x.shape
        )
        x = rearrange(x, " n c h w -> 1 n c h w")
        x, p_attn = self.attn(x, x, x)
        x = rearrange(x, " 1 n c h w -> n h w c")
        x = self.linear(x)
        x, *_ = torch.max(x, dim=0, keepdim=True)
        if return_attns:
            return x, p_attn

        return x


class MultiHeadMasterQueryAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(n_head * d_k), nn.Linear(n_head * d_k, n_head * d_k)
        )

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v, key_padding_mask):
        # print("q,k,v", q.shape, k.shape, v.shape)
        # print(key_padding_mask.shape)
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(q).view(sz_b, seq_len, n_head, d_k)
        q = q.mean(dim=1).squeeze()  # MEAN query sz_b,n_head,d_k
        q = self.fc2(q.view(sz_b, n_head * d_k)).view(sz_b, n_head, d_k)
        q = q.permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)

        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk
        # my_logger.info("{} vlue ".format(v.shape))
        # my_logger.info("{} key".format(k.shape))
        # my_logger.info("query {}".format(q.shape))
        # v = v.repeat(n_head, 1, 1)  # (n*b) x lv x d_in, correction
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        # my_logger.info("{} vlue ".format(v.shape))
        # print("q,k,v", q.shape, k.shape, v.shape)
        output, attn = self.attention(q, k, v, mask=key_padding_mask)
        # my_logger.info("{} vlue ".format(v.shape))

        output = output.view(
            n_head, sz_b, 1, d_in // n_head
        )  # correction seems that there is an errorr
        output = output.squeeze(dim=2)

        return output, attn
