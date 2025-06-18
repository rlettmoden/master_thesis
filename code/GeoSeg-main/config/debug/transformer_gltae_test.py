"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.mm_harz import MultiModalHarzDataset

from geoseg.modules.backbones.dofa.new_dofa_vit import DOFAViTMM
from geoseg.modules.decoders.segformer import SegformerDecoder
from geoseg.modules.heads.shallow_classifier import ShallowClassifier
from geoseg.modules.heads.fcn_head import FCNHead
from geoseg.modules.temporal_encoders.shallow_classifiers import GLTaeDecoding
from geoseg.modules.temporal_encoders.encoding import PositionalEncoding
from geoseg.modules.temporal_encoders.transformer import Encoder
from geoseg.modules.temporal_encoders.temporal_encoder import TemporalEncoder

from geoseg.modules.tmm_seg import MultiTempoModalSegmentation

from tools.utils import Lookahead
from tools.utils import process_model_params

from geoseg.modules.norm.layer import LayerNorm


from os.path import basename



ignore_index = 255
num_classes = 8
## MM configurations
active_modalities = ["s2", "s1"]
modal_fusion = "concat"
# norm_cl=nn.SyncBatchNorm#nn.BatchNorm2d
norm_cl=nn.BatchNorm2d

# first data properties
image_size = 128
planet_ratio = 3
planet_hw = [image_size*3, image_size*3]
label_hw = [image_size, image_size]


architecture = "ubarn"
modality_config = {
    "s1":{},
    "s2":{},
    "planet":{},
}

# backbone
weights_path_backbone = "/beegfs/work/y0092788/pre_train_weights/DOFA_ViT_base_e100.pth"
backbone = DOFAViTMM(img_size_lr=image_size)
backbone.init_weights_new(weights_path_backbone)


# decoder 
shared_decoder = True
in_channel_spatial_decoder = [768, 768, 768, 768]
out_channels_decoder = 512
out_shape_decoder = [image_size//4, image_size//4]
if shared_decoder:
    spatial_decoder = SegformerDecoder(out_shape_decoder, in_channels=in_channel_spatial_decoder, 
                            out_channels=out_channels_decoder, norm_cl=norm_cl)
    for modality in active_modalities:
        modality_config[modality]["spatial_decoder"] = spatial_decoder

else:
    for modality in active_modalities:
        modality_config[modality]["spatial_decoder"] = SegformerDecoder(out_shape_decoder, in_channels=in_channel_spatial_decoder, 
                            out_channels=out_channels_decoder, norm_cl=norm_cl)

# temporal encoder
single_temporal = False
shared_temporal = False
d_model_temporal_encoder = 512
out_channels_temporal_encoder = 512
n_layers = 3
n_head_temporal_encoder = 4
n_head_decoding = 32
d_k = 8
max_len_pe = 3000
pe_cst = 10000
mlp_channes_temporal_encoder = [d_model_temporal_encoder, out_channels_temporal_encoder]
# inconv = nn.Conv1d(out_channels_decoder, d_model_temporal_encoder, 1)
inconv = None
if shared_temporal:
    p_encoder = PositionalEncoding(d_model=d_model_temporal_encoder, max_len=max_len_pe, T=pe_cst)
    mt_feature_extractor = Encoder(n_layers=n_layers, d_in=d_model_temporal_encoder, d_model=out_channels_temporal_encoder, nhead=n_head_temporal_encoder)
    mt_head = GLTaeDecoding(inplanes=out_channels_temporal_encoder, nhead=n_head_temporal_encoder)
    mt_encoder = TemporalEncoder(encoding=p_encoder, feature_extractor=mt_feature_extractor, temporal_head=mt_head, down_conv=inconv)

    for modality in active_modalities:
        modality_config[modality]["temporal_encoder"] = mt_encoder
else:
    for modality in active_modalities:
        p_encoder = PositionalEncoding(d_model=d_model_temporal_encoder, max_len=max_len_pe, T=pe_cst)
        mt_feature_extractor = Encoder(n_layers=n_layers, d_in=d_model_temporal_encoder, d_model=out_channels_temporal_encoder, nhead=n_head_temporal_encoder)
        mt_head = GLTaeDecoding(inplanes=out_channels_temporal_encoder, nhead=n_head_temporal_encoder)
        mt_encoder = TemporalEncoder(encoding=p_encoder, feature_extractor=mt_feature_extractor, temporal_head=mt_head, down_conv=inconv)
        modality_config[modality]["temporal_encoder"] = mt_encoder


# auxiliary head
use_aux_head = True
for modality in active_modalities:
    kernel_size_aux_head=1
    modality_config[modality]["aux_head"] = FCNHead(in_channels=out_channels_temporal_encoder, 
                    kernel_size=kernel_size_aux_head, 
                    num_classes=num_classes, norm=norm_cl)
        
# head 
head = ShallowClassifier(out_channels_temporal_encoder*len(active_modalities), num_classes)

# whole model
net = MultiTempoModalSegmentation(backbone=backbone, modality_config=modality_config, 
                                  head=head, active_modalities=active_modalities, 
                                  single_temporal=single_temporal, patch_size=image_size, 
                                  use_aux_head=use_aux_head)


# loss
use_aux_loss = use_aux_head
smooth_factor=0.1
loss = TempoModalLoss(ignore_index=ignore_index, smooth_factor=smooth_factor, active_modalities=active_modalities, use_auxiliary_losses=use_aux_head)
# training hparam
max_epoch = 1
train_batch_size = 1
val_batch_size = 1

# TODO adapt
lr = 6e-5
weight_decay = 0.01
backbone_lr = 6e-6
backbone_weight_decay = 0.01
weights_name = basename(__file__).replace(".py", "")
weights_path = "model_weights/testing/{}".format(weights_name)
test_weights_name = "last"
log_name = 'testing/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = "auto"#4  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None


# define the dataloader
data_root = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
invalid_indices_path = "/beegfs/work/y0092788/invalid_paths.yaml"
splits_path = "/beegfs/work/y0092788/indices_config_p128.yaml"
train_dataset = MultiModalHarzDataset(data_root, patch_size=128, mode="train", invalid_patch_policy="drop", 
                                      invalid_indices_path=invalid_indices_path, indices_path=splits_path,
                                      active_modalities=active_modalities, single_temporal=single_temporal)
val_dataset = MultiModalHarzDataset(data_root, patch_size=128, mode="validation", invalid_patch_policy="drop", 
                                      invalid_indices_path=invalid_indices_path, indices_path=splits_path,
                                      active_modalities=active_modalities, single_temporal=single_temporal)

num_workers = 5
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=num_workers,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)