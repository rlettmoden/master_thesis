import torch
import torch.nn as nn
import torch.nn.functional as F
        
# TODO check layernorm & batchnorm
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, norm=nn.BatchNorm2d):#, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.norm = norm(out_channels)#, data_format="channels_first")#norm_cfg() if norm_cfg else None
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/fcn_head.py
# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/decode_head.py#L157
class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=1,
                 kernel_size=3,
                 in_channels=512,
                 channels=256,
                 num_classes=8,
                 dropout_ratio=0.1,
                 norm=nn.BatchNorm2d):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        super().__init__()
        if num_convs == 0:
            assert in_channels == channels

        convs = []
        convs.append(
            ConvModule(
                in_channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                norm=norm))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    norm=norm))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    
    def forward(self, x):
        """Forward function."""
        output = self.convs(x)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        return output