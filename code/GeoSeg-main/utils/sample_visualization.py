# from ..geoseg.datasets.mm_harz import MultiModalHarzDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import os

def denormalize(tensor, mean, std):
    """ Reverses the normalization on a tensor.
    
    Args:
        tensor (Tensor): The normalized tensor.
        mean (list): The mean used for normalization (per channel).
        std (list): The standard deviation used for normalization (per channel).
        
    Returns:
        Tensor: The denormalized tensor.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return tensor * std + mean


def load_and_process_image(path, channel_last, clip_max, clip_sigma_environment=False):
    image = tifffile.imread(path)
    if channel_last:
        image = image.transpose(2, 0, 1)  # Convert to channel-first if needed
    if clip_sigma_environment:
        sigma_environment = 1.64
        image_reshaped = np.reshape(image, (image.shape[0], image.shape[1]*image.shape[2]))
        mean, std = image_reshaped.mean(axis=1), image_reshaped.std(axis=1)
        clip_min, clip_max = image.min(), mean + sigma_environment*std
        image_clipped = image.copy()
        image_clipped[0] = np.clip(image[0], a_min=clip_min, a_max=clip_max[0])
        image_clipped[1] = np.clip(image[1], a_min=clip_min, a_max=clip_max[1])
    else:
        clip_min = 0
        image_clipped = np.clip(image, a_min=clip_min, a_max=clip_max)
    return image_clipped


def norm_0_255(x):
    x = x.astype(np.float32)
    x_normed = (x - x.min()) / (x.max() - x.min())*255
    return x_normed.astype(np.uint8)

def visualize_loaded_sample_s1(samples, mean, std, label_colors):
    """
    Visualizes specified modalities from a sample from a dataset at a given index.
    
    Args:
        dataset (Dataset): A dataset object that supports indexing to get a sample.
        index (int): The index of the sample in the dataset to visualize.
        modalities_to_visualize (list): List of modalities to visualize e.g., ["s1", "planet"].
    
    Each tensor in the dictionary is expected to be in the format T x C x H x W,
    where T is the time series, C is the number of channels, H is the height, and W is the width.
    """
    # Fetch the sample from the dataset

    paths = samples["s1_paths"]
    times = samples["s1_dates"]
    labels = samples["labels"]
    samples = denormalize(samples["s1"], mean, std)
    # Calculate the number of rows needed in the subplot
    n_timesteps = samples.shape[1]+1
    batch_size = samples.shape[0]

    fig, axes = plt.subplots(nrows=n_timesteps+1, ncols=batch_size, figsize=(15, 5 * ((n_timesteps)+1)))
    axes = axes.flatten()

    idx = 0
    for i, sample in enumerate(samples):
        for t in range(sample.shape[0]):
            ax = axes[idx]
            img = sample[0].numpy().transpose(1, 2, 0)  # CHW to HWC
            # If it has more than one channel, display the first three as RGB
            img = np.clip(img, a_min=img.min(), a_max=1540)
            min = np.nanmin(img)
            if np.isnan(min):
                min = 0
            img = np.nan_to_num(img, nan=min)
            if img.min() != img.max():
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1 for display
            ax.imshow(img[:, :, 0], cmap='gray')
            ax.set_title(f"Time {times[t]}")
            ax.axis('off')
            idx += 1
        label_map = labels[i].detach().cpu()
        height, width = label_map.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)

        ax = axes[idx]
        # Plotting the RGB image
        for label, color in label_colors.items():
            mask = label_map == label
            rgb_image[mask] = color
        ax.imshow(rgb_image.astype(np.uint8))

        ax.set_title(f"Labels")
        ax.axis('off')
        idx += 1


    plt.tight_layout()
    plt.savefig(f"sentinel_1_test_{times[0]}.png")
    plt.close(fig)

def visualize_sample_with_predictions(sample, sample_idx, dataset, predictions, epoch, out_dir, mode):
    """
    Visualizes specified modalities from a sample from a dataset at a given index.
    
    Args:
        dataset (Dataset): A dataset object that supports indexing to get a sample.
        index (int): The index of the sample in the dataset to visualize.
        modalities_to_visualize (list): List of modalities to visualize e.g., ["s1", "planet"].
    
    Each tensor in the dictionary is expected to be in the format T x C x H x W,
    where T is the time series, C is the number of channels, H is the height, and W is the width.
    """
    # Fetch the sample from the dataset
    # Fetch the sample from the dataset
    num_modalities = len(dataset.active_modalities)
    if num_modalities == 0:
        print("No modalities specified for visualization.")
        return

    # Calculate the number of rows needed in the subplot
    n_timesteps = sum([sample[modality].shape[1] for modality in dataset.active_modalities if modality in sample])

    fig, axes = plt.subplots(nrows=n_timesteps+2, ncols=1, figsize=(15, 5 * ((n_timesteps//num_modalities)+1)))
    if num_modalities == 1:
        axes = np.array([axes]).transpose()  # Ensure axes is two-dimensional
    axes = axes.flatten()

    idx = 0
    for modality in dataset.active_modalities:
        # if dataset.apply_transforms:
        #     mean, std = dataset.modalities[modality]["mean"], dataset.modalities[modality]["std"]
        #     sample[modality] = denormalize(sample[modality].detach().cpu(), mean=mean, std=std)
        for t in range(sample[modality].shape[1]):
            ax = axes[idx]
            img = sample[modality][0][t].numpy().transpose(1, 2, 0)  # CHW to HWC
            # If it has more than one channel, display the first three as RGB
            img = np.clip(img, a_min=img.min(), a_max=1540)
            if img.shape[2] > 3:
                img = img[:, :, :3]
            else:
                min = np.nanmin(img)
                if np.isnan(min):
                    min = 0
                img = np.nan_to_num(img, nan=min)
            if img.min() != img.max():
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1 for display
            if img.shape[2] == 2:
                ax.imshow(img[:, :, 0], cmap='gray')
            else:
                ax.imshow(img)
            if len(sample[f"{modality}_dates"])> 0:
                ax.set_title(f"{modality} - Time {sample[f"{modality}_dates"][t]}")
            ax.axis('off')
            idx += 1
    if len(sample["labels"].shape) > 2:
        label_map = sample["labels"][0].detach().cpu()
    else:
        label_map = sample["labels"].detach().cpu()
    height, width = label_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    ax = axes[idx]
    visualize_labels_w_legend(label_map, dataset.label_colors, dataset.index_to_class_name, ax, "Labels")
    # # Plotting the RGB image
    # for label, color in dataset.label_colors.items():
    #     mask = label_map == label
    #     rgb_image[mask] = color
    # ax.imshow(rgb_image.astype(np.uint8))

    # ax.set_title(f"Labels")
    # ax.axis('off')
    idx += 1
    label_map = predictions[0] + 1
    height, width = label_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    ax = axes[idx]
    # Plotting the RGB image
    for label, color in dataset.label_colors.items():
        mask = label_map == label
        rgb_image[mask] = color
    ax.imshow(rgb_image.astype(np.uint8))

    ax.set_title(f"Prediction")
    ax.axis('off')
    idx += 1


    plt.tight_layout()
    out_path = os.path.join(out_dir, f"sample_{mode}_i{sample_idx}_prediction_e{epoch}.png")
    plt.savefig(out_path)
    plt.close(fig)

def visualize_predictions_and_auxiliary(sample, sample_idx, dataset, predictions, auxiliary_predictions, epoch, out_dir, mode):
    """
    Visualizes specified modalities from a sample, including auxiliary predictions for every modality.
    
    Args:
        sample (dict): A dictionary containing the sample data for each modality.
        sample_idx (int): The index of the sample in the dataset.
        dataset (Dataset): A dataset object that supports indexing to get a sample.
        predictions (tensor): The main model predictions.
        auxiliary_predictions (dict): A dictionary containing auxiliary predictions for each modality.
        epoch (int): The current epoch number.
        out_dir (str): The output directory for saving the visualization.
        mode (str): The current mode (e.g., 'train', 'val', 'test').
    """
    num_modalities = len(dataset.active_modalities)
    if num_modalities == 0:
        print("No modalities specified for visualization.")
        return

    # Calculate the number of rows needed in the subplot
    n_rows = 3 + len(dataset.active_modalities)

    fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(15, 5 * ((n_rows//num_modalities)+1)))
    axes = axes.flatten()

    idx = 0
    for modality in dataset.active_modalities:
        if modality == "s1":
            continue
        # if dataset.apply_transforms:
        #     mean, std = dataset.modalities[modality]["mean"], dataset.modalities[modality]["std"]
        #     sample[modality] = denormalize(sample[modality].detach().cpu(), mean=mean, std=std)
        for t in range(sample[modality].shape[1]):
            ax = axes[idx]
            img = sample[modality][0][t].numpy().transpose(1, 2, 0)  # CHW to HWC
            img = np.clip(img, a_min=img.min(), a_max=1540)
            if img.shape[2] > 3:
                img = img[:, :, :3]
            else:
                min = np.nanmin(img)
                if np.isnan(min):
                    min = 0
                img = np.nan_to_num(img, nan=min)
            if img.min() != img.max():
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1 for display
            if img.shape[2] == 2:
                ax.imshow(img[:, :, 0], cmap='gray')
            else:
                ax.imshow(img)
            if len(sample[f"{modality}_dates"]) > 0:
                ax.set_title(f"{modality} - Time {sample[f"{modality}_dates"][t]}")
            ax.axis('off')
            idx += 1
            break
        break

    # Visualize ground truth labels
    ax = axes[idx]
    # visualize_labels(sample["labels"][0].detach().cpu(), dataset.label_colors, ax, "Labels")
    visualize_labels_w_legend(sample["labels"][0].detach().cpu(), dataset.label_colors, dataset.index_to_class_name, ax, "Labels")
    idx += 1

    # Visualize main prediction
    ax = axes[idx]
    visualize_labels(predictions[0], dataset.label_colors, ax, "Main Prediction")
    idx += 1

    # Visualize auxiliary predictions for each modality
    for modality in dataset.active_modalities:
        if modality in auxiliary_predictions:
            ax = axes[idx]
            visualize_labels(auxiliary_predictions[modality][0].detach().cpu(), dataset.label_colors, ax, f"{modality} Aux Pred")
            idx += 1

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"sample_{mode}_i{sample_idx}_prediction_e{epoch}_aux.png")
    plt.savefig(out_path)
    plt.close(fig)

def visualize_labels(label_map, label_colors, ax, title):
    """Helper function to visualize label maps"""
    height, width = label_map.shape
     # TODO hotfix, as labels range from 1-9 while argmax is 0-8
    label_map = label_map + 1
    
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    for label, color in label_colors.items():
        mask = label_map == label
        rgb_image[mask] = color
    ax.imshow(rgb_image.astype(np.uint8))
    ax.set_title(title)
    ax.axis('off')


def visualize_labels_w_legend(label_map, label_colors, idx_to_class, ax, title):
    """Helper function to visualize label maps with legend"""
    height, width = label_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    # TODO hotfix, as labels range from 1-9 while argmax is 0-8
    label_map = label_map + 1
    # Create legend patches
    legend_patches = []
    
    for label, color in label_colors.items():
        mask = label_map == label
        rgb_image[mask] = color
        
        # Create a patch for the legend using class name
        class_name = idx_to_class[label]
        patch = mpatches.Patch(color=np.array(color)/255, label=class_name)
        legend_patches.append(patch)
    
    ax.imshow(rgb_image.astype(np.uint8))
    ax.set_title(title)
    ax.axis('off')
    
    # Add legend
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

def visualize_sample(dataset, index, modalities_to_visualize):
    """
    Visualizes specified modalities from a sample from a dataset at a given index.
    
    Args:
        dataset (Dataset): A dataset object that supports indexing to get a sample.
        index (int): The index of the sample in the dataset to visualize.
        modalities_to_visualize (list): List of modalities to visualize e.g., ["s1", "planet"].
    
    Each tensor in the dictionary is expected to be in the format T x C x H x W,
    where T is the time series, C is the number of channels, H is the height, and W is the width.
    """
    # Fetch the sample from the dataset
    sample = dataset[index]
    num_modalities = len(modalities_to_visualize)
    if num_modalities == 0:
        print("No modalities specified for visualization.")
        return

    # Calculate the number of rows needed in the subplot
    n_timesteps = sum([sample[modality].shape[0] for modality in modalities_to_visualize if modality in sample])+1

    fig, axes = plt.subplots(nrows=(n_timesteps//num_modalities)+1, ncols=num_modalities, figsize=(15, 5 * ((n_timesteps//num_modalities)+1)))
    if num_modalities == 1:
        axes = np.array([axes]).transpose()  # Ensure axes is two-dimensional
    axes = axes.flatten()

    idx = 0
    for modality in modalities_to_visualize:
        if modality not in sample:
            print(f"Modality {modality} not found in sample.")
            continue
        for t in range(sample[modality].shape[0]):
            ax = axes[idx]
            img = sample[modality][t].numpy().transpose(1, 2, 0)  # CHW to HWC
            # If it has more than one channel, display the first three as RGB
            img = np.clip(img, a_min=img.min(), a_max=1540)
            if img.shape[2] > 3:
                img = img[:, :, :3]
            else:
                min = np.nanmin(img)
                if np.isnan(min):
                    min = 0
                img = np.nan_to_num(img, nan=min)
            if img.min() != img.max():
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1 for display
            if img.shape[2] == 2:
                ax.imshow(img[:, :, 0], cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(f"{modality} - Time {sample[f"{modality}_dates"][t]}")
            ax.axis('off')
            idx += 1

    label_map = sample["labels"]
    height, width = label_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    ax = axes[idx]
    # Plotting the RGB image
    for label, color in dataset.label_colors.items():
        mask = label_map == label
        rgb_image[mask] = color
    ax.imshow(rgb_image.astype(np.uint8))

    ax.set_title(f"Labels")
    ax.axis('off')
    for ax in axes:
        ax.axis("off")


    plt.tight_layout()
    plt.savefig(f"sample_visualization_test_{index}.png")
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    image_path = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    dataset = MultiModalHarzDataset(image_path, patch_size=128, apply_transforms=True)
    visualize_sample(dataset, 870, ["s1", "s2"])  # Specify which modalities to visualize
