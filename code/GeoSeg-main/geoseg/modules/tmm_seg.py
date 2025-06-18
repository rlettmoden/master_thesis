import torch
import torch.nn as nn
import torch
import numpy as np
from torchvision.transforms.functional import resize, InterpolationMode
    
def create_identifier_array(arrays):
    result = []
    for i, arr in enumerate(arrays, start=1):
        result.extend([i] * len(arr))
    return result

def sort_and_fuse_patches(patches_dict):
    modalities = ["s1", "s2", "planet"]
    
    #### Fuse patches
    reference_mod = min(modalities, key=lambda mod: patches_dict[mod].shape[1])
    min_samples = min(patches_dict[mod].shape[1] for mod in modalities)
    device = patches_dict["planet"][0].device
    spatial_dims = patches_dict["planet"][0].shape[2:]
    
    fused_patches = []
    # used_indices = {mod: set() for mod in modalities}
    
    for i in range(min_samples):
        patch_list = []
        # reference_doy = patches_dict[f"{reference_mod}_doy"][0, i]
        reference_doy = patches_dict[f"{reference_mod}_doy"][0][i]
        
        for mod in modalities:
            # available_indices = set(range(patches_dict[mod].shape[1])) - used_indices[mod]
            # if not available_indices:
            #     break
            
            # closest_idx = torch.argmin(abs(reference_doy - patches_dict[f"{mod}_doy"][0 ,np.array(list(available_indices), dtype=int)]))
            closest_idx = torch.argmin(abs(reference_doy - patches_dict[f"{mod}_doy"][0]))
            
            patch = patches_dict[mod][0, closest_idx]
            if mod != "planet" and spatial_dims is not None:
                patch = torch.nn.functional.interpolate(patch.unsqueeze(0), size=spatial_dims, mode='nearest').squeeze(0)
            patch_list.append(patch)
            # used_indices[mod].add(closest_idx)
        
        if len(patch_list) == len(modalities):
            fused_patch = torch.cat(patch_list, dim=0)
            fused_patches.append(fused_patch)
    
    patches_dict["fused"] = torch.stack(fused_patches).to(device).unsqueeze(0)
    
    return patches_dict

class MultiTempoModalSegmentation(nn.Module):
    # Implementing only the object path
    # TODO outsource into config!
    # head, backbone, out_channels, out_indices, 
    # modality_config: temporal encoder, loss weight, aux_head, (neck?), spatial_decoder
    # 
    # backbone, temporal_encoders={"s1":None, "s2":None, "planet": None}, aux_loss_weights={""}, aux_head_position="late", architecture="ubarn", neck=None, aux_head=None, active_modalities=["s1", "s2", "planet"], num_classes=8
    def __init__(self, backbone, modality_config, head, apply_padding=False, architecture="utae", active_modalities=["s1", "s2", "planet"], single_temporal=False,
                 modal_fusion="concat", patch_size=128, planet_patch_size=3*128, use_aux_head=True, has_full_resolution_decoder=False):
        super().__init__()
        self.backbone = backbone
        # self.modality_config = modality_config
        self.head = head
        self.architecture = architecture
        self.active_modalities = active_modalities
        self.single_temporal = single_temporal
        self.modal_fusion = modal_fusion
        self.label_hw = [patch_size, patch_size]
        self.planet_hw = [planet_patch_size, planet_patch_size]
        self.use_aux_head = use_aux_head
        self.apply_padding = apply_padding
        self.has_full_resolution_decoder = has_full_resolution_decoder

        # Initialize modules for each modality
        self.modules_by_modality = nn.ModuleDict()
        if not single_temporal:
            for modality in active_modalities:
                self.modules_by_modality[modality] = nn.ModuleDict({
                    'neck': modality_config[modality].get('neck', None),
                    'temporal_encoder': modality_config[modality].get('temporal_encoder', None),
                    'spatial_decoder': modality_config[modality].get('spatial_decoder', None),
                    'aux_head': modality_config[modality].get('aux_head', None)
                })
        else:
            modality = "s1"
            self.modules_by_modality[modality] = nn.ModuleDict({
                'neck': modality_config[modality].get('neck', None),
                'temporal_encoder': modality_config[modality].get('temporal_encoder', None),
                'spatial_decoder': modality_config[modality].get('spatial_decoder', None),
                'aux_head': modality_config[modality].get('aux_head', None)
            })

        # pooling
        ratio = np.array(self.planet_hw) / np.array(self.label_hw)

        if np.count_nonzero(~np.equal(np.mod(ratio, 1), 0)):
            raise NotImplementedError()
        
        # TODO max vs average pooling
        ratio = ratio.astype(int).tolist()
        self.pooling_kernel = nn.MaxPool2d(kernel_size=ratio, stride=ratio)

    def forward_modality(self, input_dict, modality):
        x = input_dict[modality] if not self.modal_fusion == "early" else input_dict["fused"]
        doy = input_dict[f"{modality}_doy"]
        mask = input_dict[f"{modality}_padding_mask"]

        modules = self.modules_by_modality[modality] if not self.single_temporal and not self.modal_fusion == "early" else self.modules_by_modality["s1"]
        neck = modules['neck']
        temporal_encoder = modules['temporal_encoder']
        aux_head = modules['aux_head']
        spatial_decoder = modules['spatial_decoder']

        apply_reshape = len(x.shape) == 5
        if apply_reshape:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
        
        spatial_encoder_modality = modality if not self.modal_fusion == "early" else "fused"
        x, backbone_aux = self.backbone(x, spatial_encoder_modality)

        if self.architecture == "utae":
            for i in range(len(x)):
                if apply_reshape:
                    x_block = x[i]
                    _, c, h, w = x_block.shape
                    x_block = x_block.view(b, t, c, h, w)

                if self.modal_fusion != "mid" and not self.single_temporal:
                    temporal_out_dict = temporal_encoder(x_block, doy, mask)
                    x_block = temporal_out_dict["features"]
                x_block = x_block.squeeze()
                if len(x_block.shape) == 3:  # single temporal & single modal
                    x_block = x_block.unsqueeze(0)
                x[i] = x_block
            if self.modal_fusion != "mid":
                x = neck(x) if neck else x
                x = spatial_decoder(x)

        elif self.architecture == "ubarn":
            x = neck(x) if neck else x
            x = spatial_decoder(x)
            if apply_reshape:
                _, c, h, w = x.shape
                x = x.view(b, t, c, h, w)
            
            if self.modal_fusion != "mid" and not self.single_temporal:
                temporal_out_dict = temporal_encoder(x, doy, mask)
                x = temporal_out_dict["features"]
            x = x.squeeze()
            if len(x.shape) == 3:  # single temporal & single modal
                x = x.unsqueeze(0)
        
        if self.use_aux_head:
            x_aux = aux_head(x)
            x_aux = resize(x_aux,
                    size=self.label_hw,
                    interpolation=InterpolationMode.BILINEAR)
        else:
            x_aux = None

        return (x, x_aux, backbone_aux)
    

    def forward(self, input_dict):
        out = {}
        x_forward_modality = {} # each element of dim: BxCxHxW
        # TODO solve & discuss
        non_empty_modalities = np.array(input_dict["non_empty_modalities"]).flatten()

        if self.modal_fusion == "early":
            # FUSE
            input_dict = sort_and_fuse_patches(input_dict)

            (x, x_aux, backbone_aux) = self.forward_modality(input_dict, "s1")
            x = self.head(x)
            x = resize(x,
                size=self.label_hw,
                interpolation=InterpolationMode.BILINEAR)
            out["logits"] = x
            out["non_empty_modalities"] = non_empty_modalities
            return out
        
        for modality in non_empty_modalities:
            (x, x_aux, backbone_aux) = self.forward_modality(input_dict, modality)
                    
            out[f"{modality}_aux"] = x_aux
            out[f"{modality}_backbone_aux"] = backbone_aux
            x_forward_modality[modality] = x

            x_forward_modality[modality] = x

        if "planet" in non_empty_modalities:#input_dict["non_empty_modalities"]:
            if self.architecture == "utae" and self.modal_fusion == "mid":
                planet_shape = x_forward_modality["planet"][0].shape[2:]
            else:
                planet_shape = x_forward_modality["planet"].shape[2:]
            for modality in self.active_modalities:
                label_shape = input_dict["labels"].shape[1:]

                if self.has_full_resolution_decoder:
                    min_shape = label_shape
                else:
                    min_shape = [min(label_shape[0], planet_shape[0]), min(label_shape[1], planet_shape[1])]

                if self.architecture == "utae" and self.modal_fusion == "mid":
                    # FIXME: THIS ASSUMES, THAT THERE IS A SINGLE ENCODER OUTPUT!
                    x_forward_modality[modality] = resize(x_forward_modality[modality][0], size=min_shape, interpolation=InterpolationMode.BILINEAR)
                else:
                    x_forward_modality[modality] = resize(x_forward_modality[modality], size=min_shape, interpolation=InterpolationMode.BILINEAR)
        if self.modal_fusion == "mid":
            x_concat = [x_forward_modality[key] for key in sorted(x_forward_modality.keys())]
            group_array = create_identifier_array(x_concat)
            x_concat = torch.concat(x_concat).unsqueeze(0)

            doy = [input_dict[f"{key}_doy"][0] for key in sorted(x_forward_modality.keys())]
            doy_concat = torch.concat(doy).unsqueeze(0)

            mask = [input_dict[f"{key}_padding_mask"][0] for key in sorted(x_forward_modality.keys())]
            mask_concat = torch.concat(mask).unsqueeze(0)

            # TODO FIXME!!!!
            # temporal_out_dict = self.s1_temporal_encoder.forward_group(x_concat, doy_concat, torch.tensor(group_array, device=doy_concat.device))
            temporal_out_dict = self.modules_by_modality["s1"]["temporal_encoder"](x_concat, doy_concat, torch.tensor(group_array, device=doy_concat.device))
            x = temporal_out_dict["features"] # BxTxCxHxW -> BxCxHxW

            if self.architecture == "utae":
                x = self.modules_by_modality["s1"]["neck"](x) if self.modules_by_modality["s1"]["neck"] else x
                x = self.modules_by_modality["s1"]["spatial_decoder"]([x])
        
        if self.single_temporal:
            sample_size = sum([input_dict[modality].shape[1] for modality in self.active_modalities])
            input_dict["labels"] = input_dict["labels"].repeat([sample_size, 1, 1])
            x_concat = [x_forward_modality[key] for key in sorted(x_forward_modality.keys())]
            x_concat = torch.concat(x_concat, dim=0)
        else:
            # TODO if fusion==global -> temporal forward + spatial forward else below
            if len(self.active_modalities) > 1:
                # TODO discuss & solve
                if isinstance(modality, tuple) or isinstance(modality, list):
                    modality = modality[0] 
                empty_modalities = np.setdiff1d(np.array(self.active_modalities), non_empty_modalities)
                if len(empty_modalities) >= 1:
                    feature_shape = x_forward_modality[non_empty_modalities[0]].shape[1:]
                    B, C, H, W = input_dict["labels"].shape[0], feature_shape[0], feature_shape[1], feature_shape[2]
                    for modality in empty_modalities:
                        x_forward_modality[modality] = torch.zeros(B, C, H, W, device=x_forward_modality[non_empty_modalities[0]].device)
                    
                if self.modal_fusion == "concat":
                    x_concat = [x_forward_modality[key] for key in sorted(x_forward_modality.keys())]
                    x_concat = torch.concat(x_concat, dim=1)
                elif self.modal_fusion == "sum":
                    x_concat = [x_forward_modality[key] for key in sorted(x_forward_modality.keys())]
                    x_concat = torch.sum(torch.vstack(x_concat), dim=0).unsqueeze(0)
                elif self.modal_fusion == "mid":
                    x_concat = x
            else:
                x_concat = x
        x = self.head(x_concat)
        # if self.single_temporal:
        #     x = torch.max(x, dim=0)[0].unsqueeze(0)
        x = resize(x,
            size=self.label_hw,
            interpolation=InterpolationMode.BILINEAR)
        out["logits"] = x
        out["non_empty_modalities"] = non_empty_modalities
        return out

    def get_backbone_params(self):
        return self.backbone.parameters()

    # def get_decoder_params(self):
    #     return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())
    
    def set_eval(self, mode=True):
        pass # TODO !

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    
if __name__ == "__main__":
    # Creating the tensors
    img_size = 196 
    tensor1 = torch.randn(1, 3*3*3, 4, img_size, img_size) # 3 x 3 x 4 x 196 x 196
    tensor2 = torch.randn(1, 3*2, 4, img_size, img_size) # 3 x 3 x 8 x 64 x 64
    tensor3 = torch.randn(1, 5, 4, img_size, img_size) # 3 x 10 x 2 x 64 x 64

    neck = MultiLevelNeck(in_channels=[768, 768, 768, 768], scales=[4, 2, 1, 0.5], out_channels=768)
    model = UperNet(num_classes=9, in_channels=4, neck=neck, img_size=img_size)
    loss = nn.CrossEntropyLoss()
    labels = torch.ones((1, img_size, img_size), dtype=torch.long)
    # label = nn.functional.one_hot(torch.tensor(8), num_classes=9)
    # Check if CUDA is available and move tensors to GPU if possible
    if torch.cuda.is_available():

        # Track memory usage
        torch.cuda.reset_max_memory_allocated()  # Reset peak memory metrics
        initial_memory = torch.cuda.memory_allocated()

        device = torch.device("cuda")
        tensor1 = tensor1.to(device)
        tensor2 = tensor2.to(device)
        tensor3 = tensor3.to(device)
        model = model.to(device)
        labels = labels.to(device)
        print("Tensors are moved to CUDA.")

        # basic usage -> params & model
        basic_usage = torch.cuda.memory_allocated()
    else:
        print("CUDA is not available. Tensors are on CPU.")
    model.set_input(tensor1, tensor2, tensor3)
    out = model.forward_all()
    error = loss(out, labels)
    error.backward()
    
    # Memory usage details
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Initial VRAM usage: {initial_memory / (1024 ** 2):.2f} MB")
        print(f"Basic VRAM usage: {basic_usage / (1024 ** 2):.2f} MB")
        print(f"Peak VRAM usage during computation: {peak_memory / (1024 ** 2):.2f} MB")

    pass

