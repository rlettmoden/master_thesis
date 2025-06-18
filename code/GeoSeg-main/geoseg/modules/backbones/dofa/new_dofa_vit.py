# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from .wave_dynamic_layer import Dynamic_MLP_OFA, Dynamic_MLP_Decoder
from mmengine.logging import print_log
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple
from timm.models.vision_transformer import Block

import torch.nn.functional as F

WAVE_LIST_S2 = [
    2.19,   # B12
    1.61,   # B11
    # 1.375,  # B10
    # 0.94,   # B9 SWIR
    0.865,  # B8a
    0.842,  # B8
    0.783,  # B7
    0.74,   # B6
    0.705,  # B5
    0.665,  # R 
    0.56,   # G
    0.49,   # B
    # 0.443,  # Ultra Blue
]

WAVE_LIST_S1 = [
    3.75,
    3.75
]

WAVE_LIST_PLANET = [
    0.867,  # B4
    0.666,  # R 
    0.565,   # G
    0.49,   # B
]

WAVE_LIST_FUSE = [
    # S1
    3.75,
    3.75,

    # S2
    2.19,   # B12
    1.61,   # B11
    # 1.375,  # B10
    # 0.94,   # B9 SWIR
    0.865,  # B8a
    0.842,  # B8
    0.783,  # B7
    0.74,   # B6
    0.705,  # B5
    0.665,  # R 
    0.56,   # G
    0.49,   # B
    # 0.443,  # Ultra Blue

    # planet
    0.867,  # B4
    0.666,  # R 
    0.565,   # G
    0.49,   # B
]

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

import torch

def convert_pytorch_imagenet_dict(source_dict):
    new_state_dict = {}
    
    # Mapping for encoder layers
    for i in range(12):
        source_prefix = f"encoder.layers.encoder_layer_{i}"
        target_prefix = f"blocks.{i}"
        
        # Attention layers
        new_state_dict[f"{target_prefix}.norm1.weight"] = source_dict[f"{source_prefix}.ln_1.weight"]
        new_state_dict[f"{target_prefix}.norm1.bias"] = source_dict[f"{source_prefix}.ln_1.bias"]
        new_state_dict[f"{target_prefix}.attn.qkv.weight"] = source_dict[f"{source_prefix}.self_attention.in_proj_weight"]
        new_state_dict[f"{target_prefix}.attn.qkv.bias"] = source_dict[f"{source_prefix}.self_attention.in_proj_bias"]
        new_state_dict[f"{target_prefix}.attn.proj.weight"] = source_dict[f"{source_prefix}.self_attention.out_proj.weight"]
        new_state_dict[f"{target_prefix}.attn.proj.bias"] = source_dict[f"{source_prefix}.self_attention.out_proj.bias"]
        
        # MLP layers
        new_state_dict[f"{target_prefix}.norm2.weight"] = source_dict[f"{source_prefix}.ln_2.weight"]
        new_state_dict[f"{target_prefix}.norm2.bias"] = source_dict[f"{source_prefix}.ln_2.bias"]
        new_state_dict[f"{target_prefix}.mlp.fc1.weight"] = source_dict[f"{source_prefix}.mlp.linear_1.weight"]
        new_state_dict[f"{target_prefix}.mlp.fc1.bias"] = source_dict[f"{source_prefix}.mlp.linear_1.bias"]
        new_state_dict[f"{target_prefix}.mlp.fc2.weight"] = source_dict[f"{source_prefix}.mlp.linear_2.weight"]
        new_state_dict[f"{target_prefix}.mlp.fc2.bias"] = source_dict[f"{source_prefix}.mlp.linear_2.bias"]
    
    # Other layers
    new_state_dict["cls_token"] = source_dict["class_token"]
    new_state_dict["pos_embed"] = source_dict["encoder.pos_embedding"]
    new_state_dict["patch_embed.weight_generator.weight_tokens"] = source_dict["conv_proj.weight"]
    new_state_dict["patch_embed.weight_generator.bias_token"] = source_dict["conv_proj.bias"]
    
    # Final layer norm and head
    new_state_dict["norm.weight"] = source_dict["encoder.ln.weight"]
    new_state_dict["norm.bias"] = source_dict["encoder.ln.bias"]
    new_state_dict["head.weight"] = source_dict["heads.head.weight"]
    new_state_dict["head.bias"] = source_dict["heads.head.bias"]
    
    return new_state_dict


class DOFAViTMM(nn.Module):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size_lr (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        patch_pad  (str | int | None): The padding method in patch embedding.
            Default: 'corner'.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_origin (bool): Whether to output the original input embedding.
            Default: False
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_bias (dict): Whether use bias in convolution of PatchEmbed Block.
            Default: True.
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        pre_norm (bool): Whether to add a norm before Transformer Layers.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        frozen_exclude (List): List of parameters that are not to be frozen.
            Default: ["all"], "all" means there are no frozen parameters.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size_lr=128, # s1 & s2

                 img_size_hr=384, # planet should be img_size_lr * 3
                 wave_list_s1=WAVE_LIST_S1,
                 wave_list_s2=WAVE_LIST_S2,
                 wave_list_planet=WAVE_LIST_PLANET, # TODO

                 patch_size=16,
                 patch_pad='corner',
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_origin=False,
                 out_indices=[2, 5, 8, 11],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 patch_bias=False,
                 pre_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 frozen_exclude=['all'],
                 pretrained=None,
                 init_cfg=None):
        super().__init__()#init_cfg=init_cfg

        if isinstance(img_size_lr, int):
            img_size_lr = to_2tuple(img_size_lr)
        elif isinstance(img_size_lr, tuple):
            if len(img_size_lr) == 1:
                img_size_lr = to_2tuple(img_size_lr[0])
            assert len(img_size_lr) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size_lr)}'
        if isinstance(img_size_hr, int):
            img_size_hr = to_2tuple(img_size_hr)
        elif isinstance(img_size_hr, tuple):
            if len(img_size_hr) == 1:
                img_size_hr = to_2tuple(img_size_hr[0])
            assert len(img_size_hr) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size_hr)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')


        # temporal properties
        self.img_size_hr = img_size_hr
        self.modalities = {
            's1': {
                'wave_list': wave_list_s1,
                'img_size': img_size_lr,
            },
            's2': {
                'wave_list': wave_list_s2,
                'img_size': img_size_lr,
            },
            'planet': {
                'wave_list': wave_list_planet,
                'img_size': img_size_hr,
            },
            'fused': {
                'wave_list': WAVE_LIST_FUSE,
                'img_size': img_size_hr,
            }
        }


        self.img_size_lr = img_size_lr
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained
        self.out_origin = out_origin
        self.frozen_exclude = frozen_exclude

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dims)
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, num_patches_lr + 1, embed_dims)) # TODO KEEP FOR COPYING

        # patches lr & hr
        num_patches_lr = (img_size_lr[0] // patch_size) * \
            (img_size_lr[1] // patch_size)
        num_patches_hr = (img_size_hr[0] // patch_size) * \
            (img_size_hr[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        # position embeds lr & hr
        self.pos_embed_lr = nn.Parameter(
            torch.zeros(1, num_patches_lr + 1, embed_dims))
        self.pos_embed_hr = nn.Parameter(
            torch.zeros(1, num_patches_hr + 1, embed_dims))
            
        num_patches_vanilla = (224 // 16) * (224 // 16)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (num_patches_vanilla) + 1, embed_dims))
        
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.pre_norm = pre_norm

        if self.pre_norm:
            self.pre_ln_name, pre_ln = build_norm_layer(
                norm_cfg, embed_dims, postfix='_pre')
            self.add_module(self.pre_ln_name, pre_ln)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dims, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=nn.LayerNorm)
            for i in range(num_layers)])

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self._freeze()

    @property
    def pre_ln(self):
        return getattr(self, self.pre_ln_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    def init_weights_new(self, checkpoint_path, transform_imagenet=False):

        def init_pos_emb(pos_embed_module, img_size):
            if pos_embed_module.shape != state_dict['pos_embed'].shape:
                print_log(msg=f'Resize the pos_embed shape from '
                            f'{state_dict["pos_embed"].shape} to '
                            f'{pos_embed_module.shape}')
                h, w = img_size
                pos_size = int(\
                    math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                params = self.resize_pos_embed(
                    state_dict['pos_embed'],
                    (h // self.patch_size, w // self.patch_size),
                    (pos_size, pos_size), self.interpolate_mode)
                pos_embed_module.parameters = params.detach().clone()

        
        checkpoint = CheckpointLoader.load_checkpoint(checkpoint_path, logger=None, map_location='cpu')
        if transform_imagenet:
            checkpoint = convert_pytorch_imagenet_dict(checkpoint)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

            # if self.init_cfg.get('type') == 'Pretrained':
            # elif self.init_cfg.get('type') == 'Pretrained_Part':
            #     state_dict = checkpoint.copy()
            #     para_prefix = 'image_encoder'
            #     prefix_len = len(para_prefix) + 1
            #     for k, v in checkpoint.items():
            #         state_dict.pop(k)
            #         if para_prefix in k:
            #             state_dict[k[prefix_len:]] = v

        if 'pos_embed' in state_dict.keys():
            init_pos_emb(img_size=self.img_size_hr, pos_embed_module=self.pos_embed_hr)
            init_pos_emb(img_size=self.img_size_lr, pos_embed_module=self.pos_embed_lr)

        # for name, param in self.named_parameters():
        #     if name in ['specific.layer.weight', 'specific.layer.bias']:  # Specify parameter names of interest
        #         print(f"{name}: {param}")
        #         break
        load_state_dict(self, state_dict, strict=False, logger=None)
        # for name, param in self.named_parameters():
        #     print(name)
        #     if name in ['specific.layer.weight', 'specific.layer.bias']:  # Specify parameter names of interest
        #         print(f"{name}: {param}")
        #         break

        # elif self.init_cfg is not None:
        #     super().init_weights()
        # else:
        #     # We only implement the 'jax_impl' initialization implemented at
        #     # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
        #     trunc_normal_(self.pos_embed_lr, std=.02)
        #     trunc_normal_(self.cls_token, std=.02)
        #     for n, m in self.named_modules():
        #         if isinstance(m, nn.Linear):
        #             trunc_normal_(m.weight, std=.02)
        #             if m.bias is not None:
        #                 if 'ffn' in n:
        #                     nn.init.normal_(m.bias, mean=0., std=1e-6)
        #                 else:
        #                     nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.Conv2d):
        #             kaiming_init(m, mode='fan_in', bias=0.)
        #         elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
        #             constant_init(m, val=1.0, bias=0.)

    def init_weights(self):

        def init_pos_emb(pos_embed_module, img_size):
            if pos_embed_module.shape != state_dict['pos_embed'].shape:
                print_log(msg=f'Resize the pos_embed shape from '
                            f'{state_dict["pos_embed"].shape} to '
                            f'{pos_embed_module.shape}')
                h, w = img_size
                pos_size = int(\
                    math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                params = self.resize_pos_embed(
                    state_dict['pos_embed'],
                    (h // self.patch_size, w // self.patch_size),
                    (pos_size, pos_size), self.interpolate_mode)
                pos_embed_module.parameters = params.detach().clone()

        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') in ['Pretrained', 'Pretrained_Part']:
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if self.init_cfg.get('type') == 'Pretrained':
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

            elif self.init_cfg.get('type') == 'Pretrained_Part':
                state_dict = checkpoint.copy()
                para_prefix = 'image_encoder'
                prefix_len = len(para_prefix) + 1
                for k, v in checkpoint.items():
                    state_dict.pop(k)
                    if para_prefix in k:
                        state_dict[k[prefix_len:]] = v

            if 'pos_embed' in state_dict.keys():
                init_pos_emb(img_size=self.img_size_hr, pos_embed_module=self.pos_embed_hr)
                init_pos_emb(img_size=self.img_size_lr, pos_embed_module=self.pos_embed_lr)

            load_state_dict(self, state_dict, strict=False, logger=None)
        elif self.init_cfg is not None:
            super().init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed_lr, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def _freeze(self):
        if 'all' in self.frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in self.frozen_exclude]):
                param.requires_grad = False

    def _pos_embeding(self, patched_img, hw_shape, img_size, pos_embed):
        """Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (img_size[0] // self.patch_size) * (
                    img_size[1] // self.patch_size) + 1:
                pos_h = img_size[0] // self.patch_size
                pos_w = img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
    
    # def forward_all(self, x):
    #     x_s1 = x["s1_img"]
    #     x_s2 = x["s2_img"]
    #     x_planet = x["planet_img"]

    #     # spatial encoder
    #     x_s1 = self.forward_modality(x_s1, "s1")
    #     x_s2 = self.forward_modality(x_s2, "s2")
    #     x_planet = self.forward_modality(x_planet, "planet")
    #     # TODO planet pooling vs Max pooling ||| use reflective padding? vs specific neck resize
    #     x_planet = nn.AvgPool2d(kernel_size=3, stride=3)

    #     if self.temporal_fusion_location == "ubarn":
    #         # TODO spatial decoder head


    #         # temporal encoding
    #         x_s1 = self.modalities["s1"]["temporal_encoder"](x_s1)
    #         x_s2 = self.modalities["s2"]["temporal_encoder"](x_s2)
    #         x_planet = self.modalities["planet"]["temporal_encoder"](x_planet)

    #         # concat features TODO: unsqueeze necessary?
    #         x = torch.concat([x_s1, x_s2, x_planet])

    #         # TODO: classifier head

    #     elif self.temporal_fusion_location == "utae":
    #         # TODO first get attention scores at lowest resolution
    #         pass


    def forward(self, x, modality_key):
        modality = self.modalities[modality_key]

        # Example of using modality specific configurations
        wave_list = modality['wave_list']
        B = x.shape[0]
        if wave_list is None:
            raise ValueError
        
        
        waves = torch.tensor(wave_list, device=x.device).float()

        x, _ = self.patch_embed(x, waves)

        # hw shapes
        hw = modality["img_size"][0] // self.patch_embed.kernel_size
        pos_embed = self.pos_embed_lr if modality_key in ("s1", "s2") else self.pos_embed_hr
        hw_shape = (hw,hw)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, modality["img_size"], pos_embed)
        

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        if self.pre_norm:
            x = self.pre_ln(x)

        outs = []
        if self.out_origin:
            if self.with_cls_token:
                # Remove class token and reshape token for decoder head
                out = x[:, 1:]
            else:
                out = x
            B, _, C = out.shape
            out = out.reshape(B, hw_shape[0], hw_shape[1],
                              C).permute(0, 3, 1, 2).contiguous()
            if self.output_cls_token:
                out = [out, x[:, 0]]
            outs.append(out)

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i == len(self.blocks) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)
        # outs = tuple(outs)
        return outs, None

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()


if __name__ == "__main__":
    path = "/beegfs/work/y0092788/pre_train_weights/DOFA_ViT_base_e100.pth"
    model = DOFAViTMM()
    model.init_weights_new(path)
    pass