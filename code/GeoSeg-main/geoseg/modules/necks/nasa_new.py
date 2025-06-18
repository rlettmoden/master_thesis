import torch.nn as nn

def _convTranspose2dOutput(
    input_size: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    output_padding: int,
):
    """
    Calculate the output size of a ConvTranspose2d.
    Taken from: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    return (
        (input_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )

class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_embed_dim: int,
        Hp: int = 8,
        Wp: int = 8,
        drop_cls_token: bool = False,
        num_layers: int = 4,
        use_separate_necks: bool = False,
        norm_cl = nn.SyncBatchNorm,
        only_last_idx = False,
        only_first_block = True,
    ):
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = Hp
        self.W_out = Wp
        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.use_separate_necks = use_separate_necks
        self.only_first_block = only_first_block

        # Calculate output dimensions
        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        self.norm_cl = norm_cl
        for _ in range(4):
            self.H_out = _convTranspose2dOutput(
                self.H_out, stride, padding, dilation, kernel_size, output_padding
            )
            self.W_out = _convTranspose2dOutput(
                self.W_out, stride, padding, dilation, kernel_size, output_padding
            )

        if use_separate_necks:
            # Create separate neck for each layer
            self.necks = nn.ModuleList([
                self._create_neck() for _ in range(num_layers)
            ])
        else:
            # Create a single shared neck
            self.shared_neck = self._create_neck()
        self.only_last_idx = only_last_idx

    def _create_neck(self):
        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0

        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            self.norm_cl(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            self.norm_cl(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        if self.only_first_block:
            return fpn1
        else:
            return nn.Sequential(fpn1, fpn2) 

    # def forward(self, x_layers):
    #     outputs = []
    #     if self.only_last_idx:
    #         x = x_layers[-1]
    #         # x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp)

    #         x = self.shared_neck(x)

    #         # x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))
    #         outputs = x
    #     else:
    #         for i, x in enumerate(x_layers):
    #             if self.drop_cls_token:
    #                 x = x[:, 1:, :]
    #             # x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp)

    #             if self.use_separate_necks:
    #                 x = self.necks[i](x)
    #             else:
    #                 x = self.shared_neck(x)

    #             # x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))
    #             outputs.append(x)

    #     return outputs

    def forward(self, x_layers):
        outputs = []
        x = x_layers[-1]
        # x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp)

        x = self.shared_neck(x)

        # x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))
        outputs = x

        return outputs
