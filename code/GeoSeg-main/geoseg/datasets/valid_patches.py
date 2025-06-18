import torch
from mm_harz import MultiModalHarzDataset, is_invalid_patch
import json
import yaml
import numpy as np
from tqdm import tqdm

def get_invalid_patches(dataset):
    invalid_paths = {}
    # Iterate over all items in the dataset
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]  # Retrieve data using __getitem__
        invalid_paths[idx] = {}
        
        # Process each modality
        for modality in dataset.active_modalities:
            # TODO Debug
            # if modality in ("s1", "s2"):
            #     continue
            patches = data[modality]  # Get patches for the current modality
            invalid_idxs = is_invalid_patch(torch.tensor(patches, dtype=torch.float))

            if invalid_idxs.count_nonzero() > 0:
                invalid_paths[idx][modality] = invalid_idxs.numpy().tolist()
        if len(invalid_paths[idx]) == 0:
            del invalid_paths[idx]
    return invalid_paths

def write_paths_to_yaml(paths, filename='invalid_paths_planet.yaml'):
    with open(filename, 'w') as file:
        yaml.dump(paths, file, default_flow_style=False)

if __name__ == "__main__":
    image_path = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    dataset = MultiModalHarzDataset(image_path, patch_size=128, apply_transforms=False, active_modalities=["s1", "s2", "planet"], apply_dropout=False)
    # dataset = MultiModalHarzDataset(image_path, patch_size=128, apply_transforms=False, active_modalities=["planet"])
    invalid = get_invalid_patches(dataset)
    write_paths_to_yaml(invalid)
    # dataset = MultiModalHarzDataset(image_path, patch_size=128, mode="validation", indices_path="/beegfs/work/y0092788/indices_config_p128.yaml", apply_transforms=False, active_modalities=["s1", "s2"])
    # paths_val = get_invalid_patches(dataset)
    # dataset = MultiModalHarzDataset(image_path, patch_size=128, mode="train", indices_path="/beegfs/work/y0092788/indices_config_p128.yaml", apply_transforms=False, active_modalities=["s1", "s2"])
    # paths_train = get_invalid_patches(dataset)
    # all_paths = {**paths_train, **paths_test, **paths_val} 
    # dataset = MultiModalHarzDataset(image_path, patch_size=128, mode="test", indices_path="/beegfs/work/y0092788/indices_config_p128.yaml", apply_transforms=False, active_modalities=["s1", "s2"])
    # paths_test = get_invalid_patches(dataset)
    # dataset = MultiModalHarzDataset(image_path, patch_size=128, mode="validation", indices_path="/beegfs/work/y0092788/indices_config_p128.yaml", apply_transforms=False, active_modalities=["s1", "s2"])
    # paths_val = get_invalid_patches(dataset)
    # dataset = MultiModalHarzDataset(image_path, patch_size=128, mode="train", indices_path="/beegfs/work/y0092788/indices_config_p128.yaml", apply_transforms=False, active_modalities=["s1", "s2"])
    # paths_train = get_invalid_patches(dataset)
    # all_paths = {**paths_train, **paths_test, **paths_val} 
