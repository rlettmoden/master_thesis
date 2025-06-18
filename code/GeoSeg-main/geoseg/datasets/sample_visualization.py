from mm_harz import MultiModalHarzDataset
import numpy as np
import matplotlib.pyplot as plt
import tifffile

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

# Example usage
if __name__ == "__main__":
    image_path = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    dataset = MultiModalHarzDataset(image_path, patch_size=128, apply_transforms=True)
    visualize_sample(dataset, 870, ["s1", "s2"])  # Specify which modalities to visualize
