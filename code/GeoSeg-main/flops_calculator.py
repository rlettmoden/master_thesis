import torch
import argparse
from fvcore.nn import FlopCountAnalysis
from utils.sample_visualization import visualize_sample_with_predictions
from train_supervision import Supervision_Train
from tools.cfg import py2cfg
import pandas as pd

def calculate_flops_and_params(config, name):
    # Load the model from checkpoint
    model = Supervision_Train(config=config, out_dir="/beegfs/work/y0092788/debug_out")
    model.eval()

    # Get the test dataset
    test_dataset = config.test_dataset

    # Get a sample from the dataset
    sample = test_dataset[0]

    # Prepare the input
    for modality in config.active_modalities:
        # if config.single_temporal:
        #     sample[modality] = sample[modality][:1]
        sample[modality] = sample[modality].unsqueeze(0)
        if config.modal_fusion == "mid" or config.modal_fusion == "early":
            sample[f"{modality}_doy"] = [sample[f"{modality}_doy"]]
            sample[f"{modality}_padding_mask"] = [sample[f"{modality}_padding_mask"]]
    sample["labels"] = sample["labels"].unsqueeze(0)

    # Calculate FLOPs
    flops = FlopCountAnalysis(model, sample)
    flops_by_module = flops.by_module()

    # Extract FLOPs for specific modules
    flops_encoder = flops_by_module["net.backbone"]
    flops_head = flops_by_module["net.head"]
    flops_decoder_s1 = flops_by_module["net.modules_by_modality.s1.spatial_decoder"]
    flops_decoder_s2 = flops_by_module["net.modules_by_modality.s2.spatial_decoder"]
    flops_decoder_planet = flops_by_module["net.modules_by_modality.planet.spatial_decoder"]
    flops_neck_s1 = flops_by_module["net.modules_by_modality.s1.neck"]
    flops_neck_s2 = flops_by_module["net.modules_by_modality.s2.neck"]
    flops_neck_planet = flops_by_module["net.modules_by_modality.planet.neck"]
    flops_temporal_encoder_s1 = flops_by_module["net.modules_by_modality.s1.temporal_encoder"]
    flops_temporal_encoder_s2 = flops_by_module["net.modules_by_modality.s2.temporal_encoder"]
    flops_temporal_encoder_planet = flops_by_module["net.modules_by_modality.planet.temporal_encoder"]
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.net.parameters())
    modality = "s1" if "s1" in config.active_modalities or config.single_temporal or config.modal_fusion == "early" else config.active_modalities[0]
    params_decoder = sum(p.numel() for p in model.net.modules_by_modality[modality].spatial_decoder.parameters())
    params_temporal_encoder = sum(p.numel() for p in model.net.modules_by_modality[modality].temporal_encoder.parameters()) if not config.single_temporal else 0
    params_head = sum(p.numel() for p in model.net.head.parameters())
    params_backbone = sum(p.numel() for p in model.net.backbone.parameters())
    neck_params = sum(p.numel() for p in model.net.modules_by_modality[modality].neck.parameters()) if model.net.modules_by_modality[modality].neck else 0


    # Calculate trainable parameters
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "Config": name,
        "Total FLOPs": flops.total(),
        "Head FLOPs": flops_head,
        "Backbone FLOPs": flops_encoder,
        "Decoder S1 FLOPs": flops_decoder_s1 + flops_neck_s1,
        "Decoder S2 FLOPs": flops_decoder_s2 + flops_neck_s2,
        "Temporal Encoder Planet FLOPs": flops_temporal_encoder_planet,
        "Temporal Encoder S1 FLOPs": flops_temporal_encoder_s1,
        "Temporal Encoder S2 FLOPs": flops_temporal_encoder_s2,
        "Decoder Planet FLOPs": flops_decoder_planet + flops_neck_planet,
        "Total Parameters": total_params,
        "Parameters Head": params_head,
        "Parameters Decoder": params_decoder + neck_params,
        "Parameters Backbone": params_backbone,
        "Parameters Temporal Encoder": params_temporal_encoder
    }

def main():
    configs = [
        "st_s1",
        "st_s2",
        "st_planet",
        "st_all",
        # "ltae_s1_mt_s2_planet_concat_vit_ti",
        # "ltae_s1_mt_s2_planet_concat_vit_s",
        # "ltae_s2",
        # "ltae_planet",
        # "ltae_all",
        # "ltae_s1_mt_s2_planet_concat",
        # "mt_all_segformer",
        # "mt_all_dofa_drop",
        # "mt_all_prithvi_single",
        # "mt_all_prithvi_partial_double",
        # "mt_all_prithvi_double",
        # "mt_all_ubarn",
        # "ltae_all",
        # "mt_all_mid",
        # "mt_all_concat",
        # "mt_planet_vit_l",
        # "mt_s2_planet_vit_l",
        # "mt_planet",
        # "mt_s1",
        # "mt_s2",
        # "mt_s1_planet",
        # "mt_s2_planet",
        # "mt_s1_s2",
    ]
    
    results = []
    for config_name in configs:
        full_config = f"/beegfs/work/y0092788/thesis/GeoSeg-main/config/{config_name}.py"
        config = py2cfg(full_config)
        result = calculate_flops_and_params(config, config_name)
        results.append(result)

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Write the DataFrame to an Excel file

    df.to_excel("model_flops_and_params_appendix.xlsx", index=False)

if __name__ == "__main__":
    main()
