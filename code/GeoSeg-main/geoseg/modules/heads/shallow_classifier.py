import torch.nn as nn

# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/decode_head.py#L157
class ShallowClassifier(nn.Module):

    def __init__(self,
                 in_channels=512*3,
                 num_classes=8,
                 dropout_ratio=0.1):
        super().__init__()
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    
    def forward(self, x):
        """Forward function."""
        output = x
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        return output