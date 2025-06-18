import torch
from mm_harz import MultiModalHarzDataset, is_invalid_patch
import json
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(dataset):
    all_labels = []
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        labels = data['labels']  # Assuming 'label' is the key for labels in your dataset
        all_labels.extend(labels)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=np.array(all_labels).flatten())
    return dict(enumerate(class_weights))

def write_paths_to_yaml(paths, filename='class_weights.yaml'):
    with open(filename, 'w') as file:
        yaml.dump(paths, file, default_flow_style=False)

if __name__ == "__main__":
    image_path = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    dataset = MultiModalHarzDataset(image_path, patch_size=128, apply_transforms=False, active_modalities=[], apply_dropout=False, indices_path="/beegfs/work/y0092788/indices_config_p128_norm.yaml")
    
    class_weights = compute_class_weights(dataset)
    print("Class Weights:", class_weights)
