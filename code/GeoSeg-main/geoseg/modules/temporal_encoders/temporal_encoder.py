from .encoding import PositionalEncoding
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, temporal_head, down_conv=None, encoding=None, feature_extractor=None):
        super().__init__()
        self.encoding = encoding
        self.feature_extractor = feature_extractor
        self.temporal_head = temporal_head
        self.down_conv = down_conv

    # TODO implement masking
    def forward(self, x, doy, padding_mask):
        if self.down_conv:
            # TODO debug !!!
            out = self.down_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        # inputs of shape # B T H W C
        sz_b, seq_len, d, h, w = x.shape
        if self.encoding:
            # bp = (
            #     doy.unsqueeze(-1)
            #     .repeat((1, 1, h))
            #     .unsqueeze(-1)
            #     .repeat((1, 1, 1, w))
            # )  # BxTxHxW
            # bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            # pe = self.encoding(bp).reshape(sz_b, seq_len, d, h, w)
            pe = self.encoding(doy.reshape(sz_b, seq_len))
            x = x + pe
        temporal_out_dict = {
            "features":None,
            "attention_mask_features":None,
            "attention_mask_head":None,
        }
        temporal_out_dict["features"] = x

        if self.feature_extractor:
            feature_dict = self.feature_extractor(x)
            temporal_out_dict["features"] = feature_dict["features"]
            temporal_out_dict["attention_mask_features"] = feature_dict["attention_mask"]

        head_feature_dict = self.temporal_head(temporal_out_dict["features"], None)
        temporal_out_dict["features"] = head_feature_dict["features"]
        temporal_out_dict["attention_mask_head"] = head_feature_dict["attention_mask"]
        return temporal_out_dict
    
    # TODO implement masking
    def forward_group(self, x, doy, group):
        if self.down_conv:
            # TODO debug !!!
            out = self.down_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        # inputs of shape # B T H W C
        sz_b, seq_len, d, h, w = x.shape
        if self.encoding:
            # bp = (
            #     doy.unsqueeze(-1)
            #     .repeat((1, 1, h))
            #     .unsqueeze(-1)
            #     .repeat((1, 1, 1, w))
            # )  # BxTxHxW
            # bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            # pe = self.encoding(bp).reshape(sz_b, seq_len, d, h, w)
            pe = self.encoding(doy.reshape(sz_b, seq_len), group.reshape(sz_b, seq_len))
            x = x + pe
        temporal_out_dict = {
            "features":None,
            "attention_mask_features":None,
            "attention_mask_head":None,
        }
        temporal_out_dict["features"] = x

        if self.feature_extractor:
            feature_dict = self.feature_extractor(x)
            temporal_out_dict["features"] = feature_dict["features"]
            temporal_out_dict["attention_mask_features"] = feature_dict["attention_mask"]

        head_feature_dict = self.temporal_head(temporal_out_dict["features"], None)
        temporal_out_dict["features"] = head_feature_dict["features"]
        temporal_out_dict["attention_mask_head"] = head_feature_dict["attention_mask"]
        return temporal_out_dict