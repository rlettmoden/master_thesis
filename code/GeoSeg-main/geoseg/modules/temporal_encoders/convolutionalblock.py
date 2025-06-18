from typing import Literal

import torch
from einops import rearrange
from torch import nn

BLOCK_NAME = ["se", "pff", "basicconv", "last", "cnn11"]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True,
    )


class ConvBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        reduction=16,
        block_name: Literal["se", "basicconv", "pff", "last"] = "se",
        dropout=0,
    ):
        super().__init__()
        assert block_name in BLOCK_NAME, "{} should be in {}".format(
            block_name, BLOCK_NAME
        )
        self.block_name = block_name
        if self.block_name == "se":
            self.block = SEBasicBlock(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                reduction=reduction,
            )
        elif self.block_name == "basicconv":
            self.block = ConvBasicBlock(
                inplanes=inplanes, planes=planes, dropout=dropout
            )
        elif self.block_name == "pff":
            self.block = PFFBlock(inplanes=inplanes, planes=planes, dropout=dropout)
        elif self.block_name == "last":
            self.block = LastBlock(inplanes=inplanes, planes=planes)
        elif self.block_name == "cnn11":
            self.block = nn.Linear(in_features=inplanes, out_features=planes)

        else:
            raise NotImplementedError(f"{block_name} is not define")

    def forward(self, x: torch.Tensor):
        b, n, c, h, w = x.shape
        if self.block_name in ("pff", "last", "cnn11"):
            x = rearrange(x, "b n c h w -> b n h w c")
        else:
            x = rearrange(x, "b n c h w -> (b n) c h w")
        x = self.block(x)
        if self.block_name in ("pff", "last", "cnn11"):
            x = rearrange(x, " b n h w c -> b n c h w", n=n)
        else:
            x = rearrange(x, " (b n ) c h w -> b n c h w ", n=n)
        return x


class ConvBasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, dropout: float = 0):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.conv2 = conv3x3(planes, inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.dropout(self.relu(x))
        return self.dropout(self.conv2(x))


class PFFBlock(nn.Module):
    """pointwise feed forward layer"""

    def __init__(self, inplanes: int, planes: int, dropout: float = 0):
        super().__init__()
        self.linear1 = nn.Linear(in_features=inplanes, out_features=planes)
        self.linear2 = nn.Linear(in_features=planes, out_features=inplanes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        return self.dropout(self.linear2(x))


class LastBlock(nn.Module):
    def __init__(self, inplanes, planes, last_activ=None):
        super().__init__()
        self.linear1 = nn.Linear(in_features=inplanes, out_features=planes)
        self.activ = last_activ

    def forward(self, x):
        x = self.linear1(x)
        if self.activ is not None:
            x = self.activ(x)

        return x


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        *,
        reduction=16,
        dropout=0,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, inplanes, 1)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.se = SELayer(inplanes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return self.dropout(out)


class SELayer(nn.Module):
    """
    from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
