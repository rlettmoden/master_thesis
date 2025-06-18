import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import re
from tools.cfg import py2cfg
from train_supervision import Supervision_Train
import pytorch_lightning as pl
import numpy as np

import matplotlib.pyplot as plt
import rasterio
from PIL import Image

from torchvision.transforms import Compose


from geoseg.datasets.tm_transform import ToTensor, Normalize

def create_color_map(predictions, label_colors):
    """Convert predictions to RGB image using label colors"""
    # Create empty RGB array
    height, width = predictions.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill in colors for each class
    for class_idx, color in label_colors.items():
        mask = (predictions == class_idx)
        rgb_image[mask] = color
        
    return rgb_image

def visualize_predictions(results, config, output_dir):
    """Visualize and save the prediction maps for both halves separately"""
    
    label_colors = {
        1: (233,255,190),   # tree
        2: (255,115,223),   # grass
        3: (255,255,0),     # crop
        4: (255,0,0),       # built-up
        5: (0,77,168),      # water
        6: (38,115,0),      # conifer
        7: (56,168,0),      # Decidiuous
        8: (0, 0, 0)        # Dead tree
    }

    # Process first half
    first_half = results['first_half']
    
    # Process second half
    second_half = results['second_half']
    # Create figures for both halves
    for idx, (half_data, half_name) in enumerate([
        (first_half, '2020'),
        (second_half, '2021')
    ]):
        
        # Get dimensions
        n_vertical, n_horizontal, patch_h, patch_w = half_data.shape
        
        # Create full image array
        full_image = np.zeros((n_vertical * patch_h, n_horizontal * patch_w), dtype=np.uint8)
        
        # Stitch patches together
        for i in range(n_vertical):
            for j in range(n_horizontal):
                y_start = i * patch_h
                y_end = (i + 1) * patch_h
                x_start = j * patch_w
                x_end = (j + 1) * patch_w
                full_image[y_start:y_end, x_start:x_end] = half_data[i, j]

        # Create RGB image
        rgb_image = create_color_map(full_image + 1, label_colors)
        
       # Save RGB visualization as PNG using PIL
        png_path = f"{output_dir}/prediction_map_{half_name}.png"
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(png_path, format='PNG', dpi=(300, 300))
        
        # Save RGB visualization as GeoTIFF
        rgb_tif_path = f"{output_dir}/prediction_map_{half_name}_rgb.tif"
        
        with rasterio.open(rgb_tif_path, 'w',
                          driver='GTiff',
                          height=rgb_image.shape[0],
                          width=rgb_image.shape[1],
                          count=3,  # 3 bands for RGB
                          dtype=rgb_image.dtype) as dst:
            # Write each color channel separately
            for band_idx in range(3):
                dst.write(rgb_image[:, :, band_idx], band_idx + 1)

def inference_dataset(model, dataloader, config, split):
    """
    Perform inference on a dataset
    
    Args:
        model: The trained model
        dataloader: DataLoader for the dataset
        config: Configuration object containing dataset settings
    
    Returns:
        predictions: List of model predictions
        ground_truth: List of ground truth labels
    """
    model.eval()
    predictions = []
    idxs = []
    
    # Create datasets
    datasets = {
        'train': config.train_dataset,
        'val': config.val_dataset,
        'test': config.test_dataset
    }

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            # Prepare input data
            # input_dict = {}
            for modality in config.active_modalities:
                batch[modality] = batch[modality].cuda()
                batch[f"{modality}_doy"] = batch[f"{modality}_doy"].cuda()
                batch[f"{modality}_padding_mask"] = batch[f"{modality}_padding_mask"]
            
            # Forward pass
            prediction = model.forward(batch)
            pre_mask = nn.Softmax(dim=1)(prediction["logits"])
            pre_mask = pre_mask.argmax(dim=1).cpu()
            
            # Store predictions and ground truth
            predictions.append(pre_mask)
            idxs.append(datasets[split].indices[i])
    
    # Convert lists to tensors
    predictions = torch.cat(predictions)
    idxs = torch.tensor(idxs)

    return predictions, idxs

def get_val_transform(modality_config):
    val_transform = [
        ToTensor(),
        Normalize(modality_config)
    ]
    return Compose(val_transform)

def run_inference_all_datasets(model, config):
    results = {}
    all_predictions = []
    all_idxs = []
    
    config.train_dataset.temporal_dropout = None
    config.train_dataset.transform = get_val_transform(config.train_dataset.modalities)
    dataloaders = {
        'train': DataLoader(dataset=config.train_dataset,
                          batch_size=1,
                          num_workers=5,
                          pin_memory=True,
                          shuffle=False,
                          drop_last=True),
        'val': config.val_loader,
        'test': config.test_loader
    }
    
    # Get dimensions from dataset
    num_patches_vertical = config.train_dataset.num_patches_vertical
    num_patches_horizontal = config.train_dataset.num_patches_horizontal
    
    # Collect predictions and indices from all splits
    for dataset_name, dataloader in dataloaders.items():
        print(f"\nRunning inference on {dataset_name} dataset...")
        predictions, idxs = inference_dataset(model, dataloader, config, dataset_name)
        all_predictions.append(predictions)
        all_idxs.append(idxs)
    
    # Concatenate predictions and indices from all splits
    all_predictions = torch.cat(all_predictions)
    all_idxs = torch.cat(all_idxs)
    
    # Get sorting indices and sort predictions
    sorted_indices = torch.argsort(all_idxs)
    sorted_predictions = all_predictions[sorted_indices]
    sorted_idxs = all_idxs[sorted_indices]
    print(sorted_idxs)
    
    # Split predictions into two halves
    total_patches = len(sorted_predictions)
    mid_point = total_patches // 2
    
    first_half_pred = sorted_predictions[:mid_point]
    second_half_pred = sorted_predictions[mid_point:]
    
    # Reshape each half into (num_patches_vertical, num_patches_horizontal, 128, 128)
    first_half_reshaped = first_half_pred.reshape(num_patches_vertical, num_patches_horizontal, 128, 128)
    second_half_reshaped = second_half_pred.reshape(num_patches_vertical, num_patches_horizontal, 128, 128)
    
    results = {
        'first_half': first_half_reshaped,
        'second_half': second_half_reshaped,
        'indices': sorted_idxs
    }
    
    return results

def remove_date(input_string):
    # Regex pattern to match the date and time portion
    pattern = r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'
    
    # Use re.sub to replace the matched pattern with an empty string
    result = re.sub(pattern, '', input_string)
    
    return result

def main():
    # Assuming you have your model and config already loaded
    root_config_dir = "/beegfs/work/y0092788/thesis/GeoSeg-main/config"
    exp_dir = "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55"
    name = os.path.basename(remove_date(exp_dir))
    config_path = os.path.join(root_config_dir, f"{name}.py")
    checkpoint_path = os.path.join(exp_dir, "model_weights", "testing", name, f"{name}.ckpt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = py2cfg(config_path)

    weights = torch.load(checkpoint_path, map_location=device)
    adapted_weights = weights["state_dict"]
    model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config, out_dir=exp_dir, strict=False)
    model.net.load_state_dict(adapted_weights, strict=False)
    model.eval()
    
    # Move model to GPU if available
    model = model.to(device)
    
     # Run inference and get results
    results = run_inference_all_datasets(model, config)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize and save predictions
    visualize_predictions(results, config, output_dir)

if __name__ == "__main__":
    main()
