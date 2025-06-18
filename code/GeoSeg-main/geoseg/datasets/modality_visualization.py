from mm_harz import MultiModalHarzDataset
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from os.path import dirname
import tqdm

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
        image_clipped = np.clip(image, a_min=None, a_max=clip_max)
    return image_clipped


def norm_0_255(x):
    x = x.astype(np.float32)
    x_normed = (x - x.min()) / (x.max() - x.min())*255
    return x_normed.astype(np.uint8)


def visualize_images(dataset, modality, year):
    config = dataset.modalities[modality]
    n_images = len(config["image_paths"])
    n_rows = 1
    if not config['has_rgb']:
        n_rows = 2
    
    filtered_paths = [path for path in config["image_paths"] if year in path]
    n_images = len(filtered_paths)
    fig, ax = plt.subplots(n_rows, n_images, figsize=(24*3, 10))
    for i in range(n_images):
        image_path = filtered_paths[i]
        clip_min = -100 if modality == "s1" else 0
        image = np.nan_to_num(image)
        image = load_and_process_image(image_path, config['channel_last_data'], clip_max=1540)# §, 1540 if config['has_rgb'] else 6332)
        if config['has_rgb']:
            rgb_patch = norm_0_255(image[:3]).transpose(1, 2, 0)
            ax[i].imshow(rgb_patch)
            if modality == "planet":
                title = config['str_replace_op'](os.path.basename(dirname(image_path)))
            else:
                title = config['str_replace_op'](os.path.basename(image_path))
            ax[i].set_title(title)
        else:
            grey_normed = norm_0_255(image).transpose(1, 2, 0)  # Assuming the first channel for grayscale
            ax[0, i].imshow(grey_normed[:, :, 0], cmap='gray')
            ax[0, i].set_title(f"VV {config['str_replace_op'](os.path.basename(image_path))}")
            ax[0, i].axis('off')
            ax[1, i].imshow(grey_normed[:, :, 1], cmap='gray')
            ax[1, i].set_title(f"VH {config['str_replace_op'](os.path.basename(image_path))}")
            ax[1, i].axis('off')
    
    plt.savefig(f"debug_out/{modality}_{year}_visualizations.png")

from PIL import Image
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def save_label_image(dataset, year):
    """
    Saves the label image for a specific year.
    
    Args:
        dataset: The dataset object containing the labels.
        year (str): The year for which to save the labels (e.g., '2020').
    """
    label_map = dataset.labels[year]
    height, width = label_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for label, color in dataset.label_colors.items():
        mask = label_map == label
        rgb_image[mask] = color

    img = Image.fromarray(rgb_image)
    # Create the directory if it doesn't exist
    os.makedirs(os.path.join("visuals", "labels"), exist_ok=True)
    
    # Save the image in the visuals/labels directory
    img_path = os.path.join("visuals", "labels", f"labels_{year}.png")
    img.save(img_path)
    print(f"Label image for {year} saved as {img_path}")

def create_class_color_legend(dataset):
    """
    Creates and saves an image showing the class-to-color mapping.
    
    Args:
        dataset: The dataset object containing the class information.
    """
    # Set up the image
    img_width = 300
    img_height = len(dataset.class_names) * 30 + 10
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Draw each class and its color
    for i, class_name in enumerate(dataset.class_names):
        y = i * 30 + 5
        color = dataset.label_colors[dataset.class_name_to_index[class_name]]
        
        # Draw color square
        draw.rectangle([5, y, 25, y+20], fill=color, outline='black')
        
        # Write class name
        draw.text((35, y), class_name, font=font, fill='black')

    # Create the directory if it doesn't exist
    os.makedirs(os.path.join("visuals", "labels"), exist_ok=True)

    # Save the image in the visuals/labels directory
    img_path = os.path.join("visuals", "labels", "class_color_legend.png")
    img.save(img_path)
    print(f"Class color legend saved as {img_path}")

def visualize_images_save(dataset, modality, year):
    config = dataset.modalities[modality]
    filtered_paths = [path for path in config["image_paths"] if year in path]
    
    # Create the output directory
    out_dir = os.path.join("visuals", modality)
    os.makedirs(out_dir, exist_ok=True)
    
    for i, image_path in tqdm.tqdm(enumerate(filtered_paths)):
        clip_min = -100 if modality == "s1" else 0
        image = load_and_process_image(image_path, config['channel_last_data'], clip_max=1540)
        image = np.nan_to_num(image, nan=np.nanmin(image))
        if config['has_rgb']:
            rgb_patch = norm_0_255(image[:3]).transpose(1, 2, 0)
            img = Image.fromarray(rgb_patch)
            if modality == "planet":
                title = config['str_replace_op'](os.path.basename(os.path.dirname(image_path)))
            else:
                title = config['str_replace_op'](os.path.basename(image_path))
            img.save(os.path.join(out_dir, f"{year}_{i}_{title}.png"))
        else:
            grey_normed = norm_0_255(image).transpose(1, 2, 0)
            vv_img = Image.fromarray(grey_normed[:, :, 0], mode='L')
            vh_img = Image.fromarray(grey_normed[:, :, 1], mode='L')
            title = config['str_replace_op'](os.path.basename(image_path))
            vv_img.save(os.path.join(out_dir, f"{year}_{i}_VV_{title}.png"))
            vh_img.save(os.path.join(out_dir, f"{year}_{i}_VH_{title}.png"))


if __name__ == "__main__":
    image_path = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    dataset = MultiModalHarzDataset(image_path, patch_size=128, apply_transforms=False, active_modalities=["s1", "s2", "planet"], temporal_dropout=None)
    # # analyse_mask(dataset)
    # for modality in ("s1", "s2", "planet"):
    #     visualize_images(dataset, modality, "2020")
    # for modality in ("s1", "s2", "planet"):
    #     visualize_images_save(dataset, modality, "2020")
    # Save label images for 2020 and 2021
    save_label_image(dataset, '2020')
    save_label_image(dataset, '2021')

    # Create and save the class color legend
    create_class_color_legend(dataset)