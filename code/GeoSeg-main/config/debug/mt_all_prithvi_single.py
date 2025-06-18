"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.mm_harz import MultiModalHarzDataset, TEMP_DROPOUT_PAPER

from geoseg.modules.backbones.dofa.new_dofa_vit import DOFAViTMM
from geoseg.modules.necks.multi_level_neck import MultiLevelNeck
from geoseg.modules.decoders.segformer import SegformerDecoder
from geoseg.modules.decoders.UPerNet import UperNet
from geoseg.modules.necks.nasa_new import ConvTransformerTokensToEmbeddingNeck
from geoseg.modules.heads.shallow_classifier import ShallowClassifier
from geoseg.modules.heads.fcn_head import FCNHead
from geoseg.modules.temporal_encoders.max_pool import MaxPoolTemporalEncoding
from geoseg.modules.temporal_encoders.shallow_classifiers import MasterQueryDecoding, GLTaeDecoding
from geoseg.modules.temporal_encoders.encoding import PositionalEncoding
from geoseg.modules.temporal_encoders.transformer import Encoder
from geoseg.modules.temporal_encoders.temporal_encoder import TemporalEncoder

from geoseg.modules.tmm_seg import MultiTempoModalSegmentation

from tools.utils import Lookahead
from tools.utils import process_model_params

from geoseg.modules.norm.layer import LayerNorm


from geoseg.datasets.tm_transform import ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomCropResize
from torchvision.transforms import Compose


from os.path import basename



ignore_index = 0
num_classes = 8
## MM configurations
active_modalities = ["s2", "planet", "s1"]
modal_fusion = "sum"
# norm_cl=nn.SyncBatchNorm#nn.BatchNorm2d
norm_cl=nn.BatchNorm2d

# first data properties
image_size = 128
planet_ratio = 3
planet_hw = [image_size*3, image_size*3]
label_hw = [image_size, image_size]


architecture = "utae"
modality_config = {
    "s1":{},
    "s2":{},
    "planet":{},
}

# backbone
out_indices = [11]
weights_path_backbone = "/beegfs/work/y0092788/pre_train_weights/DOFA_ViT_base_e100.pth"
backbone = DOFAViTMM(img_size_lr=image_size, out_indices=out_indices)
backbone.init_weights_new(weights_path_backbone)

# decoder 
shared_decoder = False
only_last_idx = True
# in_channel_spatial_decoder = [768, 768, 768, 768]
out_channels_decoder = 768
use_separate_necks = False
# out_shape_decoder = [image_size//4, image_size//4]
if shared_decoder:
    spatial_decoder = ConvTransformerTokensToEmbeddingNeck(768, out_channels_decoder, norm_cl=norm_cl, 
                                                           only_last_idx = only_last_idx, use_separate_necks=use_separate_necks)
    for modality in active_modalities:
        modality_config[modality]["spatial_decoder"] = spatial_decoder

else:
    for modality in active_modalities:
        modality_config[modality]["spatial_decoder"] = ConvTransformerTokensToEmbeddingNeck(768, out_channels_decoder, norm_cl=norm_cl, 
                                                           only_last_idx = only_last_idx, use_separate_necks=use_separate_necks)

# temporal encoder
single_temporal = False
for modality in active_modalities:
    modality_config[modality]["temporal_encoder"] = MaxPoolTemporalEncoding()
# shared_temporal = False
# d_model_temporal_encoder = 512
# out_channels_temporal_encoder = 512
# n_layers = 3
# n_head_temporal_encoder = 4
# # n_head_decoding = 32
# d_k = 8
# max_len_pe = 3000
# pe_cst = 10000
# mlp_channes_temporal_encoder = [d_model_temporal_encoder, out_channels_temporal_encoder]
# # inconv = nn.Conv1d(out_channels_decoder, d_model_temporal_encoder, 1)
# inconv = None

# for modality in ("planet", "s2"):
#     p_encoder = PositionalEncoding(d_model=d_model_temporal_encoder, max_len=max_len_pe, T=pe_cst)
#     mt_feature_extractor = Encoder(n_layers=n_layers, d_in=d_model_temporal_encoder, d_model=out_channels_temporal_encoder, nhead=n_head_temporal_encoder)
#     mt_head = MasterQueryDecoding(inplanes=out_channels_temporal_encoder, nhead=n_head_temporal_encoder)
#     mt_encoder = TemporalEncoder(encoding=p_encoder, feature_extractor=mt_feature_extractor, temporal_head=mt_head, down_conv=inconv)
#     modality_config[modality]["temporal_encoder"] = mt_encoder

# p_encoder = PositionalEncoding(d_model=d_model_temporal_encoder, max_len=max_len_pe, T=pe_cst)
# mt_feature_extractor = Encoder(n_layers=n_layers, d_in=d_model_temporal_encoder, d_model=out_channels_temporal_encoder, nhead=n_head_temporal_encoder)
# mt_head = GLTaeDecoding(inplanes=out_channels_temporal_encoder, nhead=n_head_temporal_encoder)
# mt_encoder = TemporalEncoder(encoding=p_encoder, feature_extractor=mt_feature_extractor, temporal_head=mt_head, down_conv=inconv)
# modality_config["s1"]["temporal_encoder"] = mt_encoder


# auxiliary head
use_aux_head = False
# for modality in active_modalities:
#     kernel_size_aux_head=1
#     modality_config[modality]["aux_head"] = FCNHead(in_channels=out_channels_decoder, 
#                     kernel_size=kernel_size_aux_head, 
#                     num_classes=num_classes)
        
# head 
head = FCNHead(in_channels=out_channels_decoder, num_classes=num_classes, norm=norm_cl)

# whole model
net = MultiTempoModalSegmentation(backbone=backbone, modality_config=modality_config, 
                                  head=head, active_modalities=active_modalities, 
                                  single_temporal=single_temporal, patch_size=image_size, 
                                  use_aux_head=use_aux_head, modal_fusion=modal_fusion, architecture=architecture)


# loss
# class_weights = torch.tensor([ 1.80854371,  1.12056942,  0.31253058,  3.03685606, 17.14859262, 0.70857039,  0.67901063, 11.99709964], dtype=torch.float)
use_aux_loss = use_aux_head
smooth_factor=0.1
main_loss = nn.CrossEntropyLoss(label_smoothing=smooth_factor, ignore_index=ignore_index)
# aux_loss = None
# aux_losses = {}
# for modality in active_modalities:
#     aux_losses[modality] = nn.CrossEntropyLoss(label_smoothing=smooth_factor, ignore_index=ignore_index)
loss = TempoModalLoss(main_loss)

# training hparam
max_epoch = 100
train_batch_size = 1
val_batch_size = 1

# TODO adapt
lr = 6e-5
weight_decay = 0.01
backbone_lr = 6e-5
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
invalid_indices_path = "/beegfs/work/y0092788/invalid_paths_planet.yaml"
splits_path = "/beegfs/work/y0092788/indices_config_p128.yaml"
crop_ratio = 3/4
def get_training_transform(modality_config):
    train_transform = [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        RandomCropResize(crop_size=(int(image_size*crop_ratio), int(image_size*crop_ratio)), p=0.5),
        Normalize(modality_config)
    ]
    return Compose(train_transform)
train_temporal_dropout = TEMP_DROPOUT_PAPER
train_dataset = MultiModalHarzDataset(data_root, patch_size=128, mode="train", invalid_patch_policy="drop", 
                                      invalid_indices_path=invalid_indices_path, indices_path=splits_path,
                                      active_modalities=active_modalities, single_temporal=single_temporal, train_transform=get_training_transform, temporal_dropout=train_temporal_dropout)
val_dataset = MultiModalHarzDataset(data_root, patch_size=128, mode="validation", invalid_patch_policy="drop", 
                                      invalid_indices_path=invalid_indices_path, indices_path=splits_path,
                                      active_modalities=active_modalities, single_temporal=single_temporal)
test_dataset = MultiModalHarzDataset(data_root, patch_size=128, mode="test", invalid_patch_policy="drop", 
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
test_loader = DataLoader(dataset=test_dataset,
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