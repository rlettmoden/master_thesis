import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from os.path import dirname


from datetime import datetime
import re


import tifffile


class TifDatasetNew(Dataset):
    def __init__(self, root_dir, is_channel_last=False, image_prefix=None, day_directory=False, mask_prefix=None, patch_size=64):
        """
        Args:
        root_dir (str): Root directory containing year and day subdirectories.
        patch_size (int): Size of the image patches to be extracted.
        """
        self.patch_size = patch_size
        self.to_tensor = transforms.ToTensor()
        self.image_paths = []
        self.mask_paths = []
        self.num_tifs = 0

        self.cached_idx = None
        self.new_image = False
        self.is_channel_last = is_channel_last

        self.include_masks=day_directory
        label_dir = os.path.join(dirname(root_dir), "labels")
        self.label_path = os.listdir(label_dir)
        self.label_path = [os.path.join(label_dir, file_name) for file_name in self.label_path if file_name.endswith(".tif")] 
        labels = [tifffile.imread(img_path) for img_path in self.label_path]
        self.labels = np.stack(labels, axis=0)

        # Walk through the directory structure and gather all .tif and mask paths
        for year_dir in os.listdir(root_dir):
            if year_dir not in ("2021", "2020"):
                continue
            year_path = os.path.join(root_dir, year_dir)
            for day in os.listdir(year_path):
                if day_directory:
                    day_path = os.path.join(year_path, day)
                    image_path = os.path.join(day_path, f"{image_prefix}.tif")
                    if os.path.exists(image_path):
                        self.image_paths.append(image_path)
                        self.num_tifs += 1
                    mask_path = os.path.join(day_path, f"{mask_prefix}.tif")
                    if os.path.exists(mask_path):
                        self.mask_paths.append(mask_path)
                else:
                    image_path = os.path.join(year_path, f"{day}") 
                    if image_path[-4:] == ".tif":
                        self.image_paths.append(image_path)


        # Assuming all images are the same size, calculate the number of patches
        self.num_patches_per_image = None
        if self.image_paths:
            example_image = tifffile.imread(self.image_paths[0])
            if self.is_channel_last:
                h, w = example_image.shape[0], example_image.shape[1]
                self.n_channels = example_image.shape[2]
            else:
                h, w = example_image.shape[1], example_image.shape[2]
                self.n_channels = example_image.shape[0]
            self.h = h
            self.w = w
            self.num_patches_per_image = (h // patch_size) * (w // patch_size)

    def __len__(self):
        return len(self.image_paths) * self.num_patches_per_image

    def __getitem__(self, index):
        # Determine which image and patch corresponds to the flat index
        img_idx = index // self.num_patches_per_image
        patch_idx = index % self.num_patches_per_image

        # Load image and mask
        if self.cached_idx == None or img_idx != self.cached_idx:
            self.image = tifffile.imread(self.image_paths[img_idx])
            if self.is_channel_last:
                self.image = self.image.transpose(2, 0, 1)
            self.cached_idx = img_idx
        
        num_patches_one_row = self.image.shape[2] // self.patch_size

        row = patch_idx // num_patches_one_row
        col = patch_idx % num_patches_one_row

        left = col * self.patch_size
        upper = row * self.patch_size
        right = left + self.patch_size
        lower = upper + self.patch_size

        # Extract the patch and corresponding mask patch
        image_patch = self.image[:, upper:lower, left:right]

        # Convert to tensor and adjust dimensions as necessary
        image_patch = self.to_tensor(image_patch).permute(1, 0, 2)
        if self.include_masks:
            self.mask = tifffile.imread(self.mask_paths[img_idx])
            mask_patch = self.mask[:, upper:lower, left:right]
            mask_patch = self.to_tensor(mask_patch).permute(1, 0, 2)
            return image_patch, mask_patch
        return (image_patch)

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


import numpy as np
import matplotlib.pyplot as plt

def get_rgb_label(label_map):
    # Define the label to color mappings (simple RGB tuples)
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

    # Convert the segmentation map to an RGB image
    height, width = label_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    for label, color in label_colors.items():
        mask = label_map == label
        rgb_image[mask] = color
    return rgb_image.astype(np.uint8)

def visualize_label_map(label_map, axis):
    rgb_image = get_rgb_label(label_map)

    # Plotting the RGB image
    axis.imshow(rgb_image)
    axis.set_title('Segmentation Map')
    plt.axis('off')  # Turn off axis numbers and ticks


def visualize_image_patches(dataset_s1, dataset_s2, dataset_planet, ratio, ratio_idx, modalities=['s1', 's2', 'planet']):
    fig, axes = plt.subplots(1, len(modalities)+2, figsize=(15, 5))
    modality_visualization = {
        "s1":{
            "n_images_2021":29,
            "n_images_2020":25,
            "channel_last_data":True,
            "has_rgb":False,
            "str_replace_op": lambda x: datetime.strptime(re.findall(r'\d{8}T\d{6}', x)[0], '%Y%m%dT%H%M%S').strftime("%Y_%m_%d"),
            "dataset":dataset_s1
        },
        "s2":{
            "n_images_2021":4,
            "n_images_2020":3,
            "channel_last_data":True,
            "has_rgb":True,
            "str_replace_op": lambda x: x.replace("Sentinel2_", "").replace(".tif", ""),
            "dataset":dataset_s2
        },
        "planet":{
            "n_images_2021":4,
            "n_images_2020":3,
            "channel_last_data":True,
            "has_rgb":True,
            "str_replace_op": lambda x: x.replace("_psscene_analytic_sr_udm2", "").replace("normalized_", ""),
            "dataset":dataset_planet
        }
    }

    imgs = {
        "planet":None,
        "s2":None,
        "s1":None,
    }
    
    for i, modality in enumerate(modalities):
        config = modality_visualization[modality]
        dataset = config['dataset']
        image_path = dataset.image_paths[0]
        print(image_path)

        clip_sigma_environment = modality == "s1"
        image = load_and_process_image(image_path, config['channel_last_data'], clip_max=1540, clip_sigma_environment=clip_sigma_environment)# §, 1540 if config['has_rgb'] else 6332)
        h, w = image.shape[1], image.shape[2] # image in C x H x W

        patch_size = (h*ratio[0], w*ratio[1])
        w_idx = (ratio_idx - int(ratio_idx))*w
        h_idx = int(ratio_idx)*patch_size[0]

        w_idx, h_idx = int(w_idx), int(h_idx)
        image_patch = image[:, h_idx:h_idx+int(patch_size[0]), w_idx:w_idx+int(patch_size[1])]
        
        if config['has_rgb']:
            rgb_patch = norm_0_255(image_patch[:3]).transpose(1, 2, 0)
            axes[i].imshow(rgb_patch)
            if modality == "planet":
                title = config['str_replace_op'](os.path.basename(dirname(image_path)))
            else:
                title = config['str_replace_op'](os.path.basename(image_path))
            axes[i].set_title(title)
            # axes[i].axis('off')
        else:
            grey_normed = norm_0_255(image_patch).transpose(1, 2, 0)  # Assuming the first channel for grayscale
            axes[i].imshow(grey_normed[:, :, 0], cmap='gray')
            # axes[i].hist(grey_normed[:, :, 0].flatten())
            axes[i].set_title(f"VV {config['str_replace_op'](os.path.basename(image_path))}")
            # axes[i].axis('off')

            axes[len(modalities)].imshow(grey_normed[:, :, 1], cmap='gray')
            # axes[len(modalities)].hist(grey_normed[:, :, 1].flatten())
            axes[len(modalities)].set_title(f"VH {config['str_replace_op'](os.path.basename(image_path))}")
            # axes[len(modalities)].axis('off')

        axes[i].set_title(f'{modality} Modality')
        if modality in ("s2", "planet"):
            imgs[modality] = rgb_patch
    label_map = dataset.labels[0]
    h, w = label_map.shape[0], label_map.shape[1] # image in C x H x W

    patch_size = (h*ratio[0], w*ratio[1])
    w_idx = (ratio_idx - int(ratio_idx))*w
    h_idx = int(ratio_idx)*patch_size[0]

    w_idx, h_idx = int(w_idx), int(h_idx)
    label_map_patch = label_map[h_idx:h_idx+int(patch_size[0]), w_idx:w_idx+int(patch_size[1])]

    visualize_label_map(label_map_patch, axes[len(modalities)+1])
    
    plt.savefig("test_alignment.png")

    # visualize_gradients(imgs["s2"], imgs["planet"], get_rgb_label(label_map_patch), ratio_idx)
    visualize_gradients_s2_planet(imgs["s2"], imgs["planet"], ratio_idx)


import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_gradients(image):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradients using Sobel operator
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return grad_mag

def visualize_gradients(image1, image2, image3, idx):
    # Calculate gradients for all three images
    grad1 = calculate_gradients(image1)
    grad2 = calculate_gradients(image2)
    grad3 = calculate_gradients(image3)
    
    # Resize grad2 to match grad1's size
    grad2_resized = cv2.resize(grad2, (grad1.shape[1], grad1.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Create a color image to visualize gradients
    color_grad = np.zeros((grad1.shape[0], grad1.shape[1], 3), dtype=np.uint8)
    
    # Assign gradients to different color channels
    color_grad[:,:,0] = grad1  # Red channel for image1
    color_grad[:,:,1] = grad2_resized  # Green channel for image2
    color_grad[:,:,2] = grad3  # Blue channel for image3
    
    # Display the result
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('S2')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title('planet')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    plt.title('Label')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(color_grad)
    plt.title('Gradient Overlap')
    plt.axis('off')
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(f"alignment_gradients_{idx}.png")
    plt.close()  # Close the figure to free up memory

def visualize_gradients_s2_planet(image1, image2, idx):
    # Calculate gradients for both images
    grad1 = calculate_gradients(image1)
    grad2 = calculate_gradients(image2)
    
    # Resize grad2 to match grad1's size
    grad2_resized = cv2.resize(grad2, (grad1.shape[1], grad1.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Create a color image to visualize gradients
    color_grad = np.zeros((grad1.shape[0], grad1.shape[1], 3), dtype=np.uint8)
    
    # Assign gradients to different color channels
    color_grad[:,:,0] = grad1  # Red channel for image1
    color_grad[:,:,1] = grad2_resized  # Green channel for image2
    
    # Display the result
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('s2')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title('planet')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(color_grad)
    plt.title('Gradient Overlap')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"alignment_gradients_{idx}.png")
    plt.close()  # Close the figure to free up memory


if __name__ == "__main__":
    modality = "planet"
    image_path = f'/beegfs/work/y0092788/data/{modality}'  # Update this to your .tif file path
    patch_size = 128
    dataset_planet = TifDatasetNew(image_path, patch_size=patch_size, day_directory=True, mask_prefix="composite_udm2", image_prefix="composite_Clip")
    modality = "s2"
    image_path = f'/beegfs/work/y0092788/data/{modality}'  # Update this to your .tif file path
    dataset_s2 = TifDatasetNew(image_path, patch_size=patch_size, is_channel_last=True)
    modality = "s1"
    image_path = f'/beegfs/work/y0092788/data/{modality}'  # Update this to your .tif file path
    dataset_s1 = TifDatasetNew(image_path, patch_size=patch_size, is_channel_last=True)#, day_directory=True, mask_prefix="composite_udm2", image_prefix="composite")
    
    ratio_s2 =  (patch_size/dataset_s2.h, patch_size/dataset_s2.w)
    ratio_s1 =  (patch_size/dataset_s1.h, patch_size/dataset_s1.w)
    visualize_image_patches(dataset_s1, dataset_s2, dataset_planet, ratio_s2, 8.0)
    visualize_image_patches(dataset_s1, dataset_s2, dataset_planet, ratio_s2, 8.5)
    visualize_image_patches(dataset_s1, dataset_s2, dataset_planet, ratio_s2, 16.0)
    visualize_image_patches(dataset_s1, dataset_s2, dataset_planet, ratio_s2, 16.5)
    visualize_image_patches(dataset_s1, dataset_s2, dataset_planet, ratio_s2, 24.0)
    visualize_image_patches(dataset_s1, dataset_s2, dataset_planet, ratio_s2, 24.5)