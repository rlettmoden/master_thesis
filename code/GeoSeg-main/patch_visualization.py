import os
import matplotlib.pyplot as plt
import numpy as np
from train_supervision import Supervision_Train

def norm_0_255(x):
    x = x.numpy().astype(np.float32)
    x_normed = (x - x.min()) / (x.max() - x.min())*255
    return x_normed.astype(np.uint8)

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def save_modality_patches(input_dict, output_dir, sample_id):
    """
    Save patches from each modality (Planet, Sentinel-2, Sentinel-1) as images for all timesteps.
    
    Args:
        input_dict (dict): Dictionary containing data for each modality
        output_dir (str): Directory to save the patches
        sample_id (str/int): Identifier for the sample
        
    Returns:
        None (saves images to disk)
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dictionary of modalities and their corresponding band selections
    modality_bands = {
        "planet": 3,  # RGB bands
        "s2": 3,     # RGB bands
        "s1": 2      # VV, VH bands
    }
    
    for modality, num_bands in modality_bands.items():
        if modality in input_dict["non_empty_modalities"]:
            # Get number of timesteps for this modality
            num_timesteps = input_dict[modality].shape[1]
            
            for t in range(num_timesteps):
                # Get the patch data for this timestep
                patch = input_dict[modality][0, t][:num_bands].squeeze().permute(1, 2, 0)
                
                # Normalize the data to 0-255 range
                patch_normalized = norm_0_255(patch)
                
                # Create figure
                plt.figure(figsize=(5, 5))
                
                if num_bands == 2:  # For Sentinel-1
                    # For SAR data, display as grayscale using first band
                    plt.imshow(patch_normalized[:,:,0], cmap='gray')
                else:  # For optical data (Planet and Sentinel-2)
                    plt.imshow(patch_normalized)
                
                plt.axis('off')
                
                # Get the date if available
                date_str = ""
                if f"{modality}_dates" in input_dict:
                    date_str = f"_{input_dict[f'{modality}_dates'][t]}"
                
                # Save the patch
                output_path = os.path.join(output_dir, f"{sample_id}_{modality}_t{t}{date_str}.png")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()

def adapt_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("net.", "")
        if new_key.startswith("s1_spatial_decoder.shared_neck"):
            new_key = new_key.replace("s1_spatial_decoder", "modules_by_modality.s1.spatial_decoder")
            new_state_dict[new_key] = value
        elif new_key.startswith("s2_spatial_decoder.shared_neck"):
            new_key = new_key.replace("s2_spatial_decoder", "modules_by_modality.s2.spatial_decoder")
            new_state_dict[new_key] = value
        elif new_key.startswith("planet_spatial_decoder.shared_neck"):
            new_key = new_key.replace("planet_spatial_decoder", "modules_by_modality.planet.spatial_decoder")
            new_state_dict[new_key] = value
        else:
            new_state_dict[new_key] = value
    return new_state_dict


def main():
    import os
    import torch
    from tools.cfg import py2cfg
    
    # Set paths
    root_config_dir = "/beegfs/work/y0092788/thesis/GeoSeg-main/config/"
    out_dir = "/beegfs/work/y0092788/visuals"  # Replace with your output directory
    name = "ltae_s1_mt_s2_planet_concat"  # Replace with your model name
    exp_dir = "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55"
    
    # Construct config path
    config_path = os.path.join(root_config_dir, f"{name}.py")
    checkpoint_path = os.path.join(exp_dir, "model_weights", "testing", name, f"{name}.ckpt")

    # Load config and model
    config = py2cfg(config_path)
    
    # Load model from checkpoint
    weights = torch.load(checkpoint_path, map_location="cpu")
    adapted_weights = adapt_state_dict(weights["state_dict"])
    model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config, out_dir=out_dir, strict=False)
    model.net.load_state_dict(adapted_weights, strict=False)
    model.eval()

    # Load config
    config = py2cfg(config_path)
    
    # Get number of samples to process
    num_samples = 5  # Change this to process more or fewer samples
    
    # Process each sample
    for sample_idx in range(num_samples):
        # Get sample from test set
        input_dict = config.test_dataset[sample_idx]

         # Prepare input data for model
        # model_input = {}
        for modality in config.active_modalities:
            input_dict[modality] = input_dict[modality].unsqueeze(0)
            input_dict[f"{modality}_doy"] = torch.tensor(input_dict[f"{modality}_doy"])
            input_dict[f"{modality}_padding_mask"] = [input_dict[f"{modality}_padding_mask"]]
        input_dict["labels"] = input_dict["labels"].unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            prediction = model.forward(input_dict)
            pre_mask = torch.nn.Softmax(dim=1)(prediction["logits"])
            pre_mask = pre_mask.argmax(dim=1).detach().cpu()
        
        # Prepare input data
        for modality in config.active_modalities:
            # input_dict[modality] = input_dict[modality].unsqueeze(0)
            
            # Denormalize the data if transforms were applied
            if config.test_dataset.apply_transforms:
                mean = config.test_dataset.modalities[modality]["mean"]
                std = config.test_dataset.modalities[modality]["std"]
                input_dict[modality] = denormalize(input_dict[modality].detach().cpu(), mean=mean, std=std)

        # Create directory for this specific sample
        sample_dir = os.path.join(out_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save patches for each modality in the sample-specific directory
        save_modality_patches(
            input_dict=input_dict,
            output_dir=sample_dir,
            sample_id=f"sample_{sample_idx}"
        )

        # Save prediction
        label_colors = {
            1: (233,255,190),   # tree
            2: (255,115,223),   # grass
            3: (255,255,0),     # crop
            4: (255,0,0),       # built-up
            5: (0,77,168),      # water
            6: (38,115,0),      # conifer
            7: (56,168,0),      # Deciduous
            8: (0, 0, 0)        # Dead tree
        }

        
        pred_rgb = np.zeros((*pre_mask.shape[1:], 3), dtype=np.uint8)
        for label, color in label_colors.items():
            pred_rgb[pre_mask[0] == label-1] = color

        # Save prediction
        plt.figure(figsize=(5, 5))
        plt.imshow(pred_rgb)
        plt.axis('off')
        plt.savefig(os.path.join(sample_dir, f"sample_{sample_idx}_prediction.png"), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()

        labels = input_dict["labels"][0].squeeze()  # Remove batch dimension
         # Create RGB image for labels
        label_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for label, color in label_colors.items():
            label_rgb[labels == label-1] = color
        
        # Create figure for labels
        plt.figure(figsize=(5, 5))
        plt.imshow(label_rgb)
        plt.axis('off')
        
        # Save the labels
        output_path = os.path.join(sample_dir, f"{sample_idx}_ground_truth.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        
        print(f"Processed sample {sample_idx}")

if __name__ == "__main__":
    main()
