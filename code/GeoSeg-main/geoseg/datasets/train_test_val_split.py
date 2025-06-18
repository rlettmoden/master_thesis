import numpy as np
from mm_harz import MultiModalHarzDataset
import cv2
import yaml
import os
from PIL import Image
from matplotlib import pyplot as plt

def mmseg_create_empty_files_by_indices(dataset, years, root_dir="root"):
    """
    Creates empty text files for each index in train, validation, and test sets for each year.

    Args:
        dataset: Instance of MultiModalHarzDataset.
        years (list of str): List of years to process.
        root_dir (str): Root directory where the files will be organized.
    """
    for year in years:
        # Define regions for splitting indices (these could be parameterized or dynamically defined elsewhere)
        file_name = f"indices_config_y{year}_p{dataset.patch_size}.yaml"
        splits = read_yaml(file_name)
        # Define subdirectories for each set
        sets = {
            "train": splits["train_indices"],
            "val": splits["validation_indices"],
            "test": splits["test_indices"]
        }

        # Determine which image and patch corresponds to the flat index
        year_counts = np.fromiter(dataset.year_n_lengths.values(), dtype=float)
        cum_sum_years = np.cumsum(year_counts)

        start_idx_per_year = (cum_sum_years - cum_sum_years[0]).astype(int)


        # Create files for each set
        for set_name, indices in sets.items():
            set_path = os.path.join(root_dir, "img_dir", set_name, f"{year}_p{dataset.patch_size}")
            if not os.path.exists(set_path):
                os.makedirs(set_path)

            label_path = os.path.join(root_dir, "ann_dir", set_name, f"{year}_p{dataset.patch_size}")
            if not os.path.exists(label_path):
                os.makedirs(label_path)
            
            for idx in indices:
                year_idx = np.argwhere(idx < cum_sum_years).flatten()[0]
                patch_idx = idx - start_idx_per_year[year_idx]
                
                # horizontal
                row_idx = patch_idx // dataset.num_patches_horizontal
                # vertical
                col_idx = patch_idx % dataset.num_patches_horizontal

                patch_size = dataset.patch_size
                row = row_idx*patch_size
                # vertical
                col = col_idx*patch_size
                filename = os.path.join(set_path, f"patch_{row_idx}_{col_idx}.txt")
                with open(filename, 'w') as file:
                    pass  # Create an empty file

                label_map = dataset.labels[year]
                label_patch = label_map[row:row+patch_size, col:col+patch_size]

                filename = os.path.join(label_path, f"patch_{row_idx}_{col_idx}.png")
                label_patch = Image.fromarray(label_patch)
                label_patch.save(filename)
                # break

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_and_stitch_images(dataset, indices, splits, modality, year, colors=None):
    num_patches_vertical = dataset.num_patches_vertical
    num_patches_horizontal = dataset.num_patches_horizontal
    patch_size = dataset.patch_size

    image_height = num_patches_vertical * patch_size
    image_width = num_patches_horizontal * patch_size
    stitched_image = np.zeros((image_height, image_width, 3), dtype=np.uint8) # TODO s2 only

    # Determine which image and patch corresponds to the flat index
    year_counts = np.fromiter(dataset.year_n_lengths.values(), dtype=float)
    cum_sum_years = np.cumsum(year_counts)

    start_idx_per_year = (cum_sum_years - cum_sum_years[0]).astype(int)

    for count, idx in enumerate(indices):
        sample = dataset[idx]
        patch = sample[modality][0][:3, :, :].transpose(1, 2, 0)# np.transpose(sample[modality][0][:3], (1, 2, 0)) # TODO only works for s2 yet
        # normalize 
        patch = np.clip(patch, a_min=patch.min(), a_max=1540)
        if patch.shape[2] == 3:
            patch = patch[:, :, :3]
        else:
            min = np.nanmin(patch)
            if np.isnan(min):
                min = 0
            patch = np.nan_to_num(patch, nan=min)
        if patch.min() != patch.max():
            patch = (patch - patch.min()) / (patch.max() - patch.min())  # Normalize to 0-1 for display

        split = [k for k,v in splits.items() if idx in v][0]
        if colors != None:
            overlay = np.full(patch.shape, colors[split], dtype=np.float64)
            patch = cv2.addWeighted((patch), 0.9, overlay, 0.1, 0)
        patch=(patch*255).astype(np.uint8)
        # patch=patch[:, :, ::-1]
        year_idx = np.argwhere(idx < cum_sum_years).flatten()[0]
        patch_idx = idx - start_idx_per_year[year_idx]
        
        # horizontal
        row = patch_idx // dataset.num_patches_horizontal
        row = row*patch_size
        # vertical
        col = patch_idx % dataset.num_patches_horizontal
        col = col*patch_size
        stitched_image[row:row + patch_size, col:col + patch_size] = patch

    return stitched_image


def write_indices_to_yaml(train_indices, val_indices, test_indices, filename="indices_config.yaml"):
    """
    Writes indices for training, validation, and testing to a YAML file.
    """
    indices_dict = {
        "train_indices": train_indices,
        "validation_indices": val_indices,
        "test_indices": test_indices
    }
    with open(filename, 'w') as file:
        yaml.dump(indices_dict, file, default_flow_style=False)

def split_indices_rectangular(dataset, year, val_start, val_end, test_start, test_end):
    """
    Splits the patch indices into train, validation, and test sets based on specified regions.

    Args:
        dataset: Instance of MultiModalHarzDataset.
        year (str): The year as a string to specify the patch indices key.
        val_start (tuple): Starting (row, col) index for the validation rectangle.
        val_end (tuple): Ending (row, col) index for the validation rectangle.
        test_start (tuple): Starting (row, col) index for the test rectangle.
        test_end (tuple): Ending (row, col) index for the test rectangle.

    Returns:
        A tuple of numpy arrays: (train_indices, val_indices, test_indices)
    """
    num_vertical = dataset.num_patches_vertical
    num_horizontal = dataset.num_patches_horizontal

    all_indices = dataset.modalities[f"patch_idx_{year}"]

    # Calculate linear indices from 2D grid indices
    val_indices = [i * num_horizontal + j for i in range(val_start[0], val_end[0] + 1)
                                               for j in range(val_start[1], val_end[1] + 1)]
    test_indices = [i * num_horizontal + j for i in range(test_start[0], test_end[0] + 1)
                                                for j in range(test_start[1], test_end[1] + 1)]

    # Convert to numpy arrays for easier processing
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # Apply offset
    val_indices = all_indices[0] + val_indices
    test_indices = all_indices[0] + test_indices

    # Ensure no overlap between validation and test sets
    assert not set(val_indices) & set(test_indices), "Validation and test regions overlap"

    # All indices that are not in validation or test sets are training indices
    train_indices = np.array([idx for idx in all_indices if idx not in val_indices and idx not in test_indices])

    return train_indices, val_indices, test_indices

def split_indices_random_6_2_2(dataset, year):
    """
    Splits the patch indices into train, validation, and test sets based on specified regions.

    Args:
        dataset: Instance of MultiModalHarzDataset.
        year (str): The year as a string to specify the patch indices key.
        val_start (tuple): Starting (row, col) index for the validation rectangle.
        val_end (tuple): Ending (row, col) index for the validation rectangle.
        test_start (tuple): Starting (row, col) index for the test rectangle.
        test_end (tuple): Ending (row, col) index for the test rectangle.

    Returns:
        A tuple of numpy arrays: (train_indices, val_indices, test_indices)
    """

    all_indices = dataset.modalities[f"patch_idx_{year}"]

    # Calculate linear indices from 2D grid indices
    np.random.seed(0)
    perm = np.random.permutation(len(all_indices))
    val_start_idx, test_start_idx = int(len(all_indices)*0.6), int(len(all_indices)*0.8)
    train_indices = all_indices[perm[:val_start_idx]]
    val_indices = all_indices[perm[val_start_idx:test_start_idx]]
    test_indices = all_indices[perm[test_start_idx:]]

    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    # Example of using the function
    year = "2021"
    # val_start = (15, 15)  # start row and column for validation region
    # val_end = (20, 20)    # end row and column for validation region
    # test_start = (25, 25) # start row and column for test region
    # test_end = (30, 30)   # end row and column for test region

    image_path = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    dataset = MultiModalHarzDataset(image_path, patch_size=128, active_modalities=["s2"], apply_transforms=False)

    years = ("2020", "2021")
    all_train_indices = []
    all_val_indices = []
    all_test_indices = []

    for year in years:
        
        train_indices, val_indices, test_indices = split_indices_random_6_2_2(dataset, year)
        train_indices, val_indices, test_indices = split_indices_random_6_2_2(dataset, year)
        
        all_train_indices.extend(train_indices.tolist())
        all_val_indices.extend(val_indices.tolist())
        all_test_indices.extend(test_indices.tolist())
        # colors = None
        colors = {'train_indices': (0, 1, 0), 'validation_indices': (1, 0, 0), 'test_indices': (0, 0, 1)}
        indices = np.sort(np.concatenate([train_indices, val_indices, test_indices]))
        splits = {
            "train_indices":train_indices,
            "validation_indices":val_indices,
            "test_indices":test_indices
        }
        stitched_image = load_and_stitch_images(dataset, indices, splits, "s2", year, colors)
        plt.imshow(stitched_image)
        plt.savefig(f'train_test_val_visuals_{year}.png')  # Save the result for each split

    file_name = f"indices_config_p{dataset.patch_size}.yaml"
    write_indices_to_yaml(all_train_indices, all_val_indices, all_test_indices, file_name)



    # mmseg_create_empty_files_by_indices(dataset, years, "/beegfs/work/y0092788/data")
