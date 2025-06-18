import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torchvision.transforms.functional import resize

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):#, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        # self.norm = norm_cfg() if norm_cfg else None
        # self.activation = act_cfg() if act_cfg else None

    def forward(self, x):
        x = self.conv(x)
        # if self.norm:
        #     x = self.norm(x)
        # if self.activation:
        #     x = self.activation(x)
        return x

class MultiLevelNeck(nn.Module):
    def __init__(self, in_channels, out_channels, adjust_channels=False, scales=[0.5, 1, 2, 4]):#, norm_cfg=None, act_cfg=None):
        super().__init__()
        assert isinstance(in_channels, list), "in_channels must be a list of integers"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        if adjust_channels:
            for in_channel in in_channels:
                self.lateral_convs.append(ConvModule(in_channel, out_channels, kernel_size=1))#, norm_cfg=norm_cfg, act_cfg=act_cfg))
        else:
            self.lateral_convs = None
        
        for _ in range(self.num_outs):
            self.convs.append(ConvModule(out_channels, out_channels, kernel_size=3, padding=1))#, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels), "The number of inputs must match the number of in_channels"
        if self.lateral_convs:
            inputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        if len(inputs) != self.num_outs:
            inputs = [inputs[0]] * self.num_outs
        
        outs = []
        for i in range(self.num_outs):
            x_resize = resize(inputs[i], (int(inputs[i].shape[2]*self.scales[i]), int(inputs[i].shape[3]*self.scales[i])))
            outs.append(self.convs[i](x_resize))
        return outs
