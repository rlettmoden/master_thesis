# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from torchvision.transforms.functional import resize, InterpolationMode

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, norm=nn.BatchNorm2d):#, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        # self.norm = LayerNorm(out_channels, data_format="channels_first")#norm_cfg() if norm_cfg else None
        self.norm = norm(out_channels)#, data_format="channels_first")#norm_cfg() if norm_cfg else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        return x

class SegformerDecoder(nn.Module):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, out_shape, in_channels=[768, 768, 768, 768], out_channels=1024, norm_cl=nn.BatchNorm2d):
        super().__init__()

        num_inputs = len(in_channels)
        
        self.out_shape = out_shape

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=in_channels[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    norm=norm_cl))

        self.fusion_conv = ConvModule(
            in_channels=out_channels * num_inputs,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                norm=norm_cl)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    img=conv(x),
                    size=self.out_shape,
                    interpolation=InterpolationMode.BILINEAR))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        return out
