import torch
from mm_harz import MultiModalHarzDataset
import json
import yaml
from tqdm import tqdm

def calculate_statistics(dataset):
    # Initialize dictionaries to store sum, sum of squares, and count for each modality
    stats = {modality: {'sum': 0, 'sum_sq': 0, 'count': 0} for modality in dataset.active_modalities}
    
    # Iterate over all items in the dataset
    for idx in tqdm(range(0, len(dataset))):
        data = dataset[idx]  # Retrieve data using __getitem__
        
        # Process each modality
        for modality in dataset.active_modalities:
            patches = torch.tensor(data[modality], dtype=torch.float64)  # Get patches for the current modality
            # Calculate sum, sum of squares, and count
            stats[modality]['sum'] += torch.nansum(patches, dim=[0, 2, 3])  # Sum over all dimensions except channels
            stats[modality]['sum_sq'] += torch.nansum(patches ** 2, dim=[0, 2, 3])
            stats[modality]['count'] += (~torch.isnan(patches)).count_nonzero(dim=[0,2,3])# patches.shape[0] * patches.shape[2] * patches.shape[3]  # Number of elements per channel
    # Calculate mean and standard deviation for each modality
    results = {}
    for modality, values in stats.items():
        mean = values['sum'] / values['count']
        std = torch.sqrt(values['sum_sq'] / values['count'] - mean ** 2)
        results[modality] = {'mean': mean.tolist(), 'std': std.tolist()}
    
    return results


def write_statistics_to_json(dataset, filename='modality_stats.json'):
    stats = calculate_statistics(dataset)
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=4)

def write_statistics_to_yaml(dataset, filename='modality_stats.yaml'):
    stats = calculate_statistics(dataset)
    with open(filename, 'w') as f:
        yaml.dump(stats, f, indent=4)

if __name__ == "__main__":
    image_path = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    # dataset = MultiModalHarzDataset(image_path, patch_size=128, indices_path="/beegfs/work/y0092788/indices_config_p128_norm.yaml", apply_transforms=False, apply_dropout=False, active_modalities=["s1", "s2", "planet"])
    dataset = MultiModalHarzDataset(image_path, patch_size=128, indices_path="/beegfs/work/y0092788/indices_config_p128_norm.yaml", apply_transforms=False, apply_dropout=False, active_modalities=["planet"])
    write_statistics_to_yaml(dataset)
# Example usage:
# Assuming 'dataset' is an instance of MultiModalHarzDataset
# stats = calculate_statistics(dataset)
# print(stats)
