import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import os
import argparse
from tools.cfg import py2cfg
from train_supervision import Supervision_Train
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import yaml
import pytorch_lightning as pl
import re
import seaborn as sns
import math

sns.set_theme(font_scale=1.5)

def sort_samples_by_metric(samples_data, metric, class_name, reverse=False):
    """
    Sort samples by a given metric, dropping NaN values before sorting.
    
    Args:
        samples_data (dict): Dictionary containing sample data
        metric (str): Name of the metric to sort by
        class_name (str): Name of the class to consider
        reverse (bool): Sort in descending order if True, ascending if False
    
    Returns:
        list: Sorted list of tuples with NaN values removed
    """
    # First, filter out NaN values
    filtered_data = {
        key: value for key, value in samples_data.items() 
        if not math.isnan(value[metric][class_name])
    }
    
    # Then sort the filtered data
    return sorted(
        filtered_data.items(),
        key=lambda x: x[1][metric][class_name],
        reverse=reverse
    )




def get_other_year_idx(idx, dataset):
    num_patches_per_image = dataset.num_patches_per_image
    if idx > num_patches_per_image:
        return idx - num_patches_per_image
    else:
        return idx + num_patches_per_image

def remove_date(input_string):
    # Regex pattern to match the date and time portion
    pattern = r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'
    
    # Use re.sub to replace the matched pattern with an empty string
    result = re.sub(pattern, '', input_string)
    
    return result

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def norm_0_255(x):
    x = x.numpy().astype(np.float32)
    x_normed = (x - x.min()) / (x.max() - x.min())*255
    return x_normed.astype(np.uint8)

def forward_mt(input_dict, net):
    planet_data = input_dict["planet"]  # Shape: 1 x T x C x H x W
    T = planet_data.shape[1]
    
    all_predictions = []
    doy_all = input_dict["planet_doy"]
    padding_mask = input_dict["planet_padding_mask"]
    for t in range(T):
        # Extract single timestep data
        input_dict["planet"] = planet_data[:, t:t+1, :, :, :]  # Shape: 1 x 1 x C x H x W
        
        # Add other necessary data for the single timestep
        input_dict["planet_doy"] = doy_all[0][t].unsqueeze(0)
        input_dict["planet_padding_mask"] = padding_mask[0][t].unsqueeze(0)
        
        # Process single timestep
        prediction = net(input_dict)
        pre_mask = nn.Softmax(dim=1)(prediction["logits"])
        pre_mask = pre_mask.argmax(dim=1).detach().cpu()
        all_predictions.append(pre_mask)

    # Combine predictions from all timesteps
    # combined_prediction = self.combine_predictions(all_predictions)
    input_dict["planet"] = planet_data
    input_dict["planet_doy"] = doy_all
    input_dict["planet_padding_mask"] = padding_mask
    return all_predictions

def visualize_temporal_changes(years_data, class_name, out_dir, name):

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

    for row, (year, input_dict, pred_mask, sample_data, forward_mt) in enumerate(years_data):
        num_timesteps = len(forward_mt)
        fig, axes = plt.subplots(num_timesteps + 1, 3, figsize=(15, 5 * num_timesteps))
        for t in range(num_timesteps):
            # Plot planet image
            planet_img = input_dict["planet"][0, t, :3].squeeze().permute(1, 2, 0)
            axes[t, 0].imshow(norm_0_255(planet_img))
            axes[t, 0].set_title(f"Planet {input_dict[f"planet_dates"][t]}")
            axes[t, 0].axis('off')

            # Plot prediction
            pred_mask_t = forward_mt[t].squeeze().cpu()
            pred_rgb = np.zeros((*pred_mask_t.shape, 3), dtype=np.uint8)
            for label, color in label_colors.items():
                pred_rgb[pred_mask_t == label-1] = color

            axes[t, 1].imshow(pred_rgb)
            axes[t, 1].set_title(f"Prediction")
            axes[t, 1].axis('off')

            # Plot changes from previous timestep
            if t > 0:
                prev_mask = forward_mt[t-1].squeeze().cpu()
                changes = pred_mask_t != prev_mask
                change_rgb = np.zeros((*changes.shape, 3), dtype=np.uint8)
                change_rgb[changes] = (255, 0, 0)  # Red color for changes

                axes[t, 2].imshow(change_rgb)
                axes[t, 2].set_title(f"Timestep {t+1} - Changes")
                axes[t, 2].axis('off')
            else:
                axes[t, 2].axis('off')
        # Plot prediction
        pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for label, color in label_colors.items():
            pred_rgb[pred_mask == label-1] = color
        axes[t+1, 1].imshow(pred_rgb[0])
        axes[t+1, 1].set_title(f"All timestamps - Prediction")
        axes[t+1, 1].axis('off')

        # Plot ground truth and prediction
        true_mask = input_dict['labels'].squeeze().cpu() + 1

        # Create RGB images for true and predicted masks
        true_rgb = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
        
        for label, color in label_colors.items():
            true_rgb[true_mask == label] = color

        # Plot ground truth
        axes[t+1, 0].imshow(true_rgb)
        axes[t+1, 0].set_title(f"{year} - Ground Truth")
        axes[t+1, 0].axis('off')
        fig.delaxes(axes[t+1, 2])

        # Create custom legend
        legend_patches = [plt.Rectangle((0,0),1,1, color=[c/255 for c in color]) for label, color in label_colors.items()]
        legend_labels = ['Tree', 'Grass', 'Crop', 'Built-up', 'Water', 'Conifer', 'Deciduous', 'Dead tree']

        # Add the legend to the figure
        fig.legend(legend_patches, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
            ncol=4, title='Land Cover Classes')

        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save the figure
        out_path = os.path.join(out_dir, f"{name}_{year}_temporal.png")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.5)
        plt.close()

def visualize_samples(model, config, out_dir, num_samples=15):
    samples_data_path = os.path.join(out_dir, "precision_recall_per_sample.yaml")
    with open(samples_data_path, 'r') as file:
        samples_data = yaml.safe_load(file)
    classes_sorted = sorted(config.test_dataset.class_names)

    for i, class_name in enumerate(classes_sorted):
        if class_name == "Tree":
            continue 
        high_precision_samples = sort_samples_by_metric(samples_data, 'precision', class_name, reverse=True)[:num_samples]
        high_f1_samples = sort_samples_by_metric(samples_data, 'f1', class_name, reverse=True)[:num_samples]
        low_recall_samples = sort_samples_by_metric(samples_data, 'recall', class_name)[:num_samples]
        low_precision_samples = sort_samples_by_metric(samples_data, 'precision', class_name)[:num_samples]

        samples_to_visualize = [
            ("high_precision", high_precision_samples),
            ("high_f1", high_f1_samples),
            ("low_recall", low_recall_samples),
            ("low_precision", low_precision_samples)
        ]

        for sample_type, samples in samples_to_visualize:
            for (sample_id, sample) in samples:
                idx_raw_ref = config.test_dataset.indices[sample_id]
                other_year_idx_raw = get_other_year_idx(idx_raw_ref, config.test_dataset)
                other_year = np.where(np.array(config.test_dataset.indices) == other_year_idx_raw)[0][0]
                
                years_data = []
                for (year_idx, idx_raw) in [(sample_id, idx_raw_ref)]:#((sample_id, idx_raw_ref), (other_year, other_year_idx_raw)):
                    input_dict = config.test_dataset[year_idx]
                    for modality in config.active_modalities:
                        input_dict[modality] = input_dict[modality].unsqueeze(0)
                        input_dict[f"{modality}_doy"] = torch.tensor(input_dict[f"{modality}_doy"])
                        input_dict[f"{modality}_padding_mask"] = [input_dict[f"{modality}_padding_mask"]]
                    input_dict["labels"] = input_dict["labels"].unsqueeze(0)
                    
                    input_dict["planet"] = input_dict["planet"][0][0].unsqueeze(0).unsqueeze(0)
                    prediction = model.forward(input_dict)
                    pre_mask = nn.Softmax(dim=1)(prediction["logits"])
                    pre_mask = pre_mask.argmax(dim=1).detach().cpu()

                    # forward_time_steps = forward_mt(input_dict, model)
                    if config.single_temporal:
                        pre_mask = pre_mask[0]
                        input_dict["labels"] = input_dict["labels"][0]

                    for modality in config.active_modalities:
                        if config.test_dataset.apply_transforms:
                            mean, std = config.test_dataset.modalities[modality]["mean"], config.test_dataset.modalities[modality]["std"]
                            input_dict[modality] = denormalize(input_dict[modality].detach().cpu(), mean=mean, std=std)

                    year_str = "2020" if idx_raw < config.test_dataset.num_patches_per_image else "2021"
                    # years_data.append((year_str, input_dict, pre_mask, samples_data[year_idx], forward_time_steps))
                    _ = None
                    years_data.append((year_str, input_dict, pre_mask, samples_data[year_idx], _))

                visualize_sample_comparison(
                    years_data,
                    class_name,
                    os.path.join(out_dir, "qualitative_eval", f"class_{i}_{class_name}_{sample_type}"),
                    sample_id
                )

                # visualize_temporal_changes(
                #     years_data,
                #     class_name,
                #     os.path.join(out_dir, "qualitative_eval", f"class_{i}_{class_name}_{sample_type}"),
                #     sample_id
                # )

def visualize_sample_comparison(years_data, class_name, out_dir, name):
    # Dynamically set number of rows based on years_data length
    num_rows = len(years_data)
    fig, axes = plt.subplots(num_rows, 4, figsize=(15, 5 * num_rows))
    
    # Convert axes to 2D array if only one row
    if num_rows == 1:
        axes = axes.reshape(1, -1)

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

    for row, (year, input_dict, pred_mask, sample_data, _) in enumerate(years_data):
        # Plot input modalities
        for j, modality in enumerate(("planet", "s2")):
            if modality in input_dict["non_empty_modalities"]:
                axes[row, j].imshow(norm_0_255(input_dict[modality][0, 0][:3].squeeze().permute(1, 2, 0)))
                axes[row, j].set_title(f"{year} - {modality}")
                axes[row, j].axis('off')
                # break

        # Plot ground truth and prediction
        true_mask = input_dict['labels'].squeeze().cpu() + 1
        pred_mask = pred_mask.squeeze().cpu()

        # Create RGB images for true and predicted masks
        true_rgb = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        
        for label, color in label_colors.items():
            true_rgb[true_mask == label] = color
            pred_rgb[pred_mask == label-1] = color  # Adjust for 0-based indexing in predictions

        # Plot ground truth
        axes[row, 2].imshow(true_rgb)
        axes[row, 2].set_title(f"{year} - Ground Truth")
        axes[row, 2].axis('off')

        # Plot prediction
        axes[row, 3].imshow(pred_rgb)
        axes[row, 3].set_title(f"{year} - Prediction")
        axes[row, 3].axis('off')

        # # Create and plot mismatch mask
        # mismatch_mask = (true_mask-1) != pred_mask
        # mismatch_rgb = np.zeros((*mismatch_mask.shape, 3), dtype=np.uint8)
        # mismatch_rgb[mismatch_mask] = (255, 0, 0)  # Red color for mismatches
        
        # axes[row, 3].imshow(mismatch_rgb)
        # axes[row, 3].set_title(f"{year} - Prediction Mismatches")
        # axes[row, 3].axis('off')

        # # Calculate pixel-wise accuracy
        # correct_pixels = torch.sum((true_mask-1) == pred_mask).item()
        # total_pixels = true_mask.numel()
        # pixel_accuracy = correct_pixels / total_pixels

        # # Add text annotation for pixel-wise accuracy
        # axes[row, 3].text(0.5, -0.1, f"{class_name} F1: {sample_data['f1'][class_name]:.2%}\n{class_name} Precision: {sample_data['precision'][class_name]:.2%}\n{class_name} Recall: {sample_data['recall'][class_name]:.2%}", 
        #                 color='black', ha='center', va='top', transform=axes[row, 3].transAxes)

        # # Store masks for year comparison only if we have 2 years of data
        # if num_rows == 2:
        #     if row == 0:
        #         first_year_true = true_mask
        #         first_year_pred = pred_mask
        #     else:
        #         # Year comparison visualization (only for second row)
        #         label_diff = first_year_true != true_mask
        #         pred_diff = first_year_pred != pred_mask
                
        #         diff_rgb = np.zeros((*label_diff.shape, 3), dtype=np.uint8)
        #         diff_rgb[label_diff] = (0, 255, 0)  # Green for label differences
        #         diff_rgb[pred_diff] = (0, 0, 255)  # Blue for prediction differences
        #         diff_rgb[label_diff & pred_diff] = (255, 0, 255)  # Magenta for both

        #         axes[1, 4].imshow(diff_rgb)
        #         axes[1, 4].set_title("Year Differences\nGreen: Label, Blue: Pred")
        #         axes[1, 4].axis('off')

    # Create custom legend
    legend_patches = [plt.Rectangle((0,0),1,1, color=[c/255 for c in color]) for label, color in label_colors.items()]
    legend_labels = ['Tree', 'Grass', 'Crop', 'Built-up', 'Water', 'Conifer', 'Deciduous', 'Dead tree']

    # Add the legend to the figure
    # fig.legend(legend_patches, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, title='Land Cover Classes')
    if num_rows == 1:
        fig.legend(legend_patches, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, title='Land Cover Classes')
    else:
        fig.legend(legend_patches, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, title='Land Cover Classes')

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save the figure
    out_path = os.path.join(out_dir, f"{name}_f1_{sample_data['f1'][class_name]:.2}_pre_{sample_data['precision'][class_name]:.2}_rec_{sample_data['recall'][class_name]:.2}.png")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()


# def visualize_sample_comparison(years_data, class_name, out_dir, name):
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))

#     label_colors = {
#         1: (233,255,190),   # tree
#         2: (255,115,223),   # grass
#         3: (255,255,0),     # crop
#         4: (255,0,0),       # built-up
#         5: (0,77,168),      # water
#         6: (38,115,0),      # conifer
#         7: (56,168,0),      # Decidiuous
#         8: (0, 0, 0)        # Dead tree
#     }

#     for row, (year, input_dict, pred_mask, sample_data, _) in enumerate(years_data):
#         # Plot input modalities
#         for j, modality in enumerate(("planet", "s2")):
#             if modality in input_dict["non_empty_modalities"]:
#                 axes[row, 0].imshow(norm_0_255(input_dict[modality][0, 0][:3].squeeze().permute(1, 2, 0)))
#                 axes[row, 0].set_title(f"{year} - {modality}")
#                 axes[row, 0].axis('off')
#                 break

#         # Plot ground truth and prediction
#         true_mask = input_dict['labels'].squeeze().cpu() + 1
#         pred_mask = pred_mask.squeeze().cpu()

#         # Create RGB images for true and predicted masks
#         true_rgb = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
#         pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        
#         for label, color in label_colors.items():
#             true_rgb[true_mask == label] = color
#             pred_rgb[pred_mask == label-1] = color  # Adjust for 0-based indexing in predictions

#         # Plot ground truth
#         axes[row, 1].imshow(true_rgb)
#         axes[row, 1].set_title(f"{year} - Ground Truth")
#         axes[row, 1].axis('off')

#         # Plot prediction
#         axes[row, 2].imshow(pred_rgb)
#         axes[row, 2].set_title(f"{year} - Prediction")
#         axes[row, 2].axis('off')

#         # # Create and plot mismatch mask
#         # mismatch_mask = (true_mask-1) != pred_mask
#         # mismatch_rgb = np.zeros((*mismatch_mask.shape, 3), dtype=np.uint8)
#         # mismatch_rgb[mismatch_mask] = (255, 0, 0)  # Red color for mismatches
        
#         # axes[row, 3].imshow(mismatch_rgb)
#         # axes[row, 3].set_title(f"{year} - Prediction Mismatches")
#         # axes[row, 3].axis('off')

#         # # Calculate pixel-wise accuracy
#         # correct_pixels = torch.sum((true_mask-1) == pred_mask).item()
#         # total_pixels = true_mask.numel()
#         # pixel_accuracy = correct_pixels / total_pixels

#         # # Add text annotation for pixel-wise accuracy
#         # axes[row, 3].text(0.5, -0.1, f"{class_name} F1: {sample_data["f1"][class_name]:.2%}\n{class_name} Precision: {sample_data["precision"][class_name]:.2%}\n{class_name} Recall: {sample_data["recall"][class_name]:.2%}", 
#         #                 color='black', ha='center', va='top', transform=axes[row, 3].transAxes)

#         # # Store masks for year comparison
#         # if row == 0:
#         #     first_year_true = true_mask
#         #     first_year_pred = pred_mask
#         # else:
#         #     # Highlight differences between years
#         #     label_diff = first_year_true != true_mask
#         #     pred_diff = first_year_pred != pred_mask
            
#         #     diff_rgb = np.zeros((*label_diff.shape, 3), dtype=np.uint8)
#         #     diff_rgb[label_diff] = (0, 255, 0)  # Green for label differences
#         #     diff_rgb[pred_diff] = (0, 0, 255)  # Blue for prediction differences
#         #     diff_rgb[label_diff & pred_diff] = (255, 0, 255)  # Magenta for both

#         #     axes[1, 4].imshow(diff_rgb)
#         #     axes[1, 4].set_title("Year Differences\nGreen: Label, Blue: Pred")
#         #     axes[1, 4].axis('off')

#     # Remove unused subplot
#     # fig.delaxes(axes[0, 4])

#     # # Create custom legend
#     # legend_patches = [plt.Rectangle((0,0),1,1, color=[c/255 for c in color]) for label, color in label_colors.items()]
#     # legend_labels = ['Tree', 'Grass', 'Crop', 'Built-up', 'Water', 'Conifer', 'Deciduous', 'Dead tree']

#     # # Add the legend to the figure
#     # fig.legend(legend_patches, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
#     #     ncol=4, title='Land Cover Classes')

#     # Adjust layout to make room for the legend
#     plt.tight_layout(rect=[0, 0.05, 1, 0.95])

#     # Save the figure
#     out_path = os.path.join(out_dir, f"{name}.png")
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     plt.savefig(out_path, bbox_inches='tight', pad_inches=0.5)
#     plt.close()

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
    parser = argparse.ArgumentParser(description="Test and visualize model predictions")
    # parser.add_argument("config_path", type=str, help="Path to the config file")
    # parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("out_dir", type=str, help="Path to theoutput directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()

    root_config_dir = "/beegfs/work/y0092788/thesis/GeoSeg-main/config"
    name = os.path.basename(remove_date(args.out_dir))
    config_path = os.path.join(root_config_dir, f"{name}.py")
    checkpoint_path = os.path.join(args.out_dir, "model_weights", "testing", name, f"{name}.ckpt")

    config = py2cfg(config_path)

    trainer = pl.Trainer(devices=1, max_epochs=config.max_epoch, accelerator='auto')
    # trainer = pl.Trainer(max_epochs=config.max_epoch, accelerator='auto',
    
    # weights = torch.load(checkpoint_path, map_location="cpu")
    weights = torch.load(checkpoint_path)
    adapted_weights = adapt_state_dict(weights["state_dict"])
    model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config, out_dir=args.out_dir, strict=False)
    model.net.load_state_dict(adapted_weights, strict=False)
    model.eval()
    # if not os.path.exists(os.path.join(args.out_dir, "precision_recall_per_sample.yaml")):
    trainer.test(model=model, dataloaders=config.test_loader)
    conf = np.load(os.path.join(args.out_dir, "test_confusion.npz"))["confusion_matrix"]
    # Normalize the confusion matrix
    conf_normalized_horizontal = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    conf_normalized_vertical = conf.astype('float') / (conf.sum(axis=0, keepdims=True) + 1e-10)

    def plot_conf(conf, vertical=True):
        # Define the class labels
        class_labels = ['tree', 'grass', 'crop', 'built-up', 'water', 'conifer', 'Deciduous', 'Dead tree']

        # Round the confusion matrix to 2 decimal places
        conf_rounded = np.round(conf, decimals=2)

        # Plot the confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_rounded, annot=True, cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels,
                    fmt='.2f')  # Format annotations to show 2 decimal places

        # Add labels and title
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Rotate the tick labels and set their alignment
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, ha='right')

        # Adjust the layout to prevent cutting off labels
        plt.tight_layout()

        # Save the figure
        appendix = "vertical" if vertical else "horizontal"
        out_path = os.path.join(args.out_dir, f"confusion_{appendix}.png")

        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.5)
        plt.close()


    plot_conf(conf=conf_normalized_horizontal, vertical=False)
    plot_conf(conf=conf_normalized_vertical, vertical=True)

    # Load the model from checkpoint
    visualize_samples(model, config, args.out_dir, 15)

if __name__ == "__main__":
    main()
