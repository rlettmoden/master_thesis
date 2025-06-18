from torch.utils.data import Dataset
from torchvision import transforms
import os
import tifffile
from os.path import dirname

import re
from datetime import datetime

import numpy as np
import yaml

import cv2
import torch
from torchvision.transforms import Compose

from torch.nn.functional import pad

from tqdm import tqdm

from .tm_transform import ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip
import torchvision.transforms.functional as F

def is_invalid_patch(patch):
    """Check if the patch invalid entries."""
    invalid_idxs = (torch.isnan(patch)).count_nonzero(dim=[1,2,3]) > 0# /(patch_size**2) > threshold
    flattened = patch.flatten(start_dim=1)
    invalid_idxs =  torch.logical_or(invalid_idxs, torch.min(flattened, dim=1)[0] == torch.max(flattened, dim=1)[0])
    if (~invalid_idxs).count_nonzero() == 0: # no valid patches
        pass
    return invalid_idxs


def get_training_transform(modality_config):
    train_transform = [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        Normalize(modality_config)
    ]
    return Compose(train_transform)



def get_val_transform(modality_config):
    val_transform = [
        ToTensor(),
        Normalize(modality_config)
    ]
    return Compose(val_transform)

def save_patch(patch, directory, filename):
    """Save a patch to a TIFF file."""
    path = os.path.join(directory, filename)
    tifffile.imwrite(path, patch)

def extract_patch(data, i, j, patch_size):
    """Extract a patch from the data array."""
    return data[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

def pad_patch(patch, target_size):
    """Pad the patch if it is not the target size."""
    if patch.shape[0] < target_size or patch.shape[1] < target_size:
        padded_patch = np.pad(patch, ((0, target_size - patch.shape[0]), 
                                      (0, target_size - patch.shape[1]),
                                      (0, 0)),
                              mode='constant', constant_values=np.nan)
        return padded_patch
    return patch

def process_patches(image, mask, patch_size, patch_dir):
    num_patches_i = image.shape[1] // patch_size
    num_patches_j = image.shape[2] // patch_size

    for i in range(num_patches_i):
        for j in range(num_patches_j):
            # Extract and save image patch
            image_patch = extract_patch(image, i, j, patch_size)
            image_patch = pad_patch(image_patch, patch_size)
            save_patch(image_patch, patch_dir, f'patch_{i}_{j}.tif')
            
            # Extract and save mask patch if mask is provided
            if mask is not None:
                mask_patch = extract_patch(mask, i, j, patch_size)
                mask_patch = pad_patch(mask_patch, patch_size)
                save_patch(mask_patch, patch_dir, f'mask_{i}_{j}.tif')


# PAPER
TEMP_DROPOUT_PAPER = {
    "s1":0.2,
    "s2":0.4,
    "planet":0.4,
}

# hard-coded
class MultiModalHarzDataset(Dataset):
    def __init__(self, root_dir, apply_padding=False, indices_path=None, invalid_indices_path=None, train_transform=get_training_transform, invalid_patch_policy="ignore", temporal_dropout=None, apply_transforms=True, mode='train', image_prefix="composite_Clip", patch_size=128, planet_ratio=3, include_masks=False, active_modalities=["s1", "s2", "planet"], single_temporal=False, data_fraction=1.):
        """
        Args:
        root_dir (str): Root directory containing year and day subdirectories.
        patch_size (int): Size of the image patches to be extracted.
        """
        self.patch_size = patch_size
        self.root_dir = root_dir

        self.cached_idx = None
        self.new_image = False
        self.include_masks = include_masks
        self.planet_ratio = planet_ratio
        self.mode = mode
        self.single_temporal = single_temporal
        self.apply_padding = apply_padding
        self.temporal_dropout = temporal_dropout

        self.indices_path = indices_path
        self.invalid_patch_policy = invalid_patch_policy
        self.data_fraction = data_fraction
        if indices_path:
            self.load_config()  # Load indices based on the mode

        self.invalid_indices_path = invalid_indices_path
        if invalid_indices_path:
            self.load_invalid_indices()
        else:
            self.invalid_indices = {}


        self.modalities = {
            "s1":{
                "channel_last_data":True,
                "has_rgb":False,
                "str_replace_op": lambda x: datetime.strptime(re.findall(r'\d{8}T\d{6}', x)[0], '%Y%m%dT%H%M%S').strftime("%Y_%m_%d"),
                "get_date_time": lambda x: datetime.strptime(re.findall(r'\d{8}T\d{6}', x)[0], '%Y%m%dT%H%M%S'),
                "image_paths":[],
                "image_paths_years":[],
                "hw":(4219, 4669),
                "n_channels":2,
                "n_max_sequence":30,

                # TODO recalculate
                "mean": np.array([-10.692324957630964, -17.286611128527618]),
                "std": np.array([3.689593965138173, 3.913801947647816]),
                # "mean":np.array([-14.1651, -14.1840]),
                # "std":np.array([5.0635, 5.0669]),
                "patch_size":patch_size
            },
            "s2":{
                "channel_last_data":True,
                "has_rgb":True,
                "str_replace_op": lambda x: datetime.strptime(x.replace("Sentinel2_", "").replace(".tif", ""), "%d%b%Y").strftime("%Y_%m_%d"),
                "get_date_time": lambda x: datetime.strptime(x.replace("Sentinel2_", "").replace(".tif", ""), "%d%b%Y"),
                "image_paths":[],
                "image_paths_years":[],
                "hw":(4219, 4669),
                "n_channels":10,
                "n_max_sequence":4,

                # TODO recalculate
                "mean": np.array([604.0264582430149, 846.0304643580894, 804.2992443269711, 1278.750781430169, 2684.828425098293, 3260.1241522582986, 3510.0911956488776, 3383.0482265478995, 2079.1742377051773, 1292.9238936175827]),
                "std": np.array([914.7867083692877, 885.3493940840641, 993.355564013849, 1000.4001726605933, 1018.3944405449548, 1187.6169429620802, 1198.8942685381799, 1240.427399773772, 1059.1396661787817, 959.4992879010775]),
                # "mean":np.array([2069.7214, 1613.4626, 1749.7418, 2247.7941, 2335.6058, 2080.5164, 1641.3789, 1759.7111, 2234.7056, 2304.8712]),
                # "std":np.array([1589.6047, 1312.7740, 1489.6716, 1544.4296, 1478.2880, 1591.8617, 1350.9224, 1504.9025, 1540.7133, 1478.4534]),
                
                "patch_size":patch_size
            },
            "planet":{
                "channel_last_data":True,
                "has_rgb":True,
                "str_replace_op": lambda x: datetime.strptime(x.replace("_bigROI4_psscene_analytic_sr_udm2", "").replace("Planet_", ""), "%d%b%Y").strftime("%Y_%m_%d"),
                "get_date_time": lambda x: datetime.strptime(x.replace("_bigROI4_psscene_analytic_sr_udm2", "").replace("Planet_", ""), "%d%b%Y"),
                "image_paths":[],
                "image_paths_years":[],
                "hw":(14064, 15564),
                "n_channels":4,
                "n_max_sequence":4,
                
                # TODO recalculate
                # "mean":np.array([1270.5644, 1270.1028, 1268.5266, 1268.0938]),
                # "std":np.array([1330.7834, 1325.6566, 1326.9633, 1327.1194]),
                # TODO currently used
                "mean":np.array([360.7112855739826, 574.2727481705358, 572.26729266286, 3105.309585714875]),
                "std":np.array([300.5757232071191, 390.64980461876706, 542.9750857602421, 1223.7283821417468]),
                # "mean":np.array([0, 0, 0, 0]),
                # "std":np.array([1000., 1000., 1000., 1000.]),
                
                "mask_paths":[],
                "patch_size":patch_size*planet_ratio
            }
        }
        if mode=="train":
            self.transform = train_transform(self.modalities)
        else:
            self.transform = get_val_transform(self.modalities)
        self.apply_transforms = apply_transforms 


        ### LABELS ###
        self.label_colors = {
            1: (233,255,190),   # tree
            2: (255,115,223),   # grass
            3: (255,255,0),     # crop
            4: (255,0,0),       # built-up
            5: (0,77,168),      # water
            6: (38,115,0),      # conifer
            7: (56,168,0),      # Decidiuous
            8: (0, 0, 0)        # Dead tree
        }
        # [ 1.80854371,  1.12056942,  0.31253058,  3.03685606, 17.14859262, 0.70857039,  0.67901063, 11.99709964]
        self.class_names = ['Tree', 'Grass', 'Crop', 'Built-up', 'Water', 'Conifer', 'Deciduous', 'Dead tree']

        self.class_name_to_index = {
            'Tree': 1, 
            'Grass': 2, 
            'Crop': 3, 
            'Built-up': 4, 
            'Water': 5, 
            'Conifer': 6, 
            'Deciduous': 7, 
            'Dead tree': 8
        }
        self.index_to_class_name = {
            1: 'Tree', 
            2: 'Grass', 
            3: 'Crop', 
            4: 'Built-up', 
            5: 'Water', 
            6: 'Conifer', 
            7: 'Deciduous', 
            8: 'Dead tree'
        }
        label_dir = os.path.join(dirname(root_dir), "labels")
        self.label_path = os.listdir(label_dir)
        self.label_path = [os.path.join(label_dir, file_name) for file_name in self.label_path if file_name.endswith(".tif")] 
        labels = [tifffile.imread(img_path) for img_path in self.label_path]
        label_keys = [re.findall(r'\d{4}.tif', label)[0][:4] for label in self.label_path]
        self.labels=dict(zip(label_keys, labels))

        self.active_modalities = active_modalities
        # Walk through the directory structure and gather all .tif and mask paths
        for modality in self.modalities.keys():
            modality_dict = self.modalities[modality]
            modality_dir = os.path.join(root_dir, modality)
            for year_dir in os.listdir(modality_dir):
                if year_dir not in ("2021", "2020"):
                    continue
                year_path = os.path.join(modality_dir, year_dir)
                for day in os.listdir(year_path):
                    if modality=="planet":
                        day_path = os.path.join(year_path, day)
                        image_path = os.path.join(day_path, f"{image_prefix}.tif")
                        if os.path.exists(image_path):
                            modality_dict["image_paths"].append(image_path)
                        # mask_path = os.path.join(day_path, f"{mask_prefix}.tif")
                        # if os.path.exists(mask_path):
                        #     modality_dict["mask_paths"].append(mask_path)
                    else:
                        image_path = os.path.join(year_path, f"{day}") 
                        if image_path.endswith(".tif"):
                            modality_dict["image_paths"].append(image_path)

            # save the years of the path
            for path in modality_dict["image_paths"]:
                path_name = os.path.basename(os.path.dirname(path)) if modality == "planet" else os.path.basename(path)
                year = str(modality_dict['get_date_time'](path_name).year)
                modality_dict["image_paths_years"].append(year)

            # make string arrays to numpy string arrays -> better indexing
            modality_dict["image_paths_years"] = np.array(modality_dict["image_paths_years"])
            modality_dict["image_paths"] = np.array(modality_dict["image_paths"])


            # Assuming all images are the same size, calculate the number of patches
            self.num_patches_per_image = None
            if len(modality_dict["image_paths"]) > 0:
                pass
                # uncomment if ratios change
                # example_image = tifffile.imread(modality_dict["image_paths"][0])
                # if modality_dict["channel_last_data"]:
                #     modality_dict["hw"] = (example_image.shape[0], example_image.shape[1])
                #     modality_dict["n_channels"] = example_image.shape[2]
                # else:
                #     modality_dict["hw"] = (example_image.shape[1], example_image.shape[2])
                #     modality_dict["n_channels"] = example_image.shape[0]
                
        # correct planet n_patches
        (h_s2, w_s2) = self.modalities["s2"]["hw"]
        num_patches_vertical = h_s2 // patch_size
        num_patches_horizontal = w_s2 // patch_size
        self.num_patches_per_image = num_patches_vertical * num_patches_horizontal
        self.num_patches_vertical = num_patches_vertical
        self.num_patches_horizontal = num_patches_horizontal

        for i, year in enumerate(np.unique(modality_dict["image_paths_years"])):
            all_idx = np.arange(num_patches_vertical*num_patches_horizontal) + i*self.num_patches_per_image
            self.modalities[f"patch_idx_{year}"] = all_idx
        self.year_n_lengths = [len(self.modalities[f"patch_idx_{year}"]) for year in np.unique(modality_dict["image_paths_years"])]
        self.year_n_lengths = dict(zip(np.unique(modality_dict["image_paths_years"]), self.year_n_lengths))

    def load_config(self):
        with open(self.indices_path, 'r') as file:
            config = yaml.safe_load(file)
        self.indices = config.get(f"{self.mode}_indices", [])
        np.random.seed(0)
        if self.data_fraction != 1.:
            frac_len = int(len(self.indices) * self.data_fraction)
            self.indices = sorted(np.random.permutation(self.indices)[:frac_len])
        if not self.indices:
            raise ValueError(f"No indices found for mode {self.mode} in the config file.")
        
    def load_invalid_indices(self):
        with open(self.invalid_indices_path, 'r') as file:
            config = yaml.safe_load(file)
        for k, v in config.items():
            for k_modality, v_modality in v.items():
                config[k][k_modality] = np.array(v_modality)
        self.invalid_indices = config
        if not self.invalid_indices_path:
            raise ValueError(f"No indices found in the config file.")

    def determine_year_and_patch(self, index):
        # Determine which image and patch corresponds to the flat index
        year_counts = np.fromiter(self.year_n_lengths.values(), dtype=float)
        cum_sum_years = np.cumsum(year_counts)
        year_idx = np.argwhere(index < cum_sum_years).flatten()[0]
        year_str = list(self.year_n_lengths.keys())[year_idx]

        start_idx_per_year = (cum_sum_years - cum_sum_years[0]).astype(int)
        patch_idx = self.modalities[f"patch_idx_{year_str}"][index - start_idx_per_year[year_idx]] - start_idx_per_year[year_idx]
        return year_str, patch_idx, start_idx_per_year[year_idx]
    
    def load_invalid_path(self, path, modality):
        image_patch = tifffile.imread(path)
        image_patch = image_patch#).to(dtype=torch.float64) # H x W x C
        image_patch = image_patch.transpose(2, 0, 1)
        image_patch = torch.tensor(image_patch, dtype=torch.float32)
        return F.normalize(image_patch, self.modalities[modality]["mean"], self.modalities[modality]["std"])
    
    def load_patch(self, year_str, patch_idx, year_offset, dataset_idx):
        #### Initialize Patch Data
        patches_dict = self._initialize_patches_dict()

        #### Calculate Patch Position
        row, column = self._calculate_patch_position(patch_idx)

        #### Load Labels
        patches_dict["labels"] = self._load_labels(year_str, row, column)

        ### Set patch idx
        patches_dict["idx"] = dataset_idx

        #### Process Modalities
        has_invalid_patches = (patch_idx + year_offset) in self.invalid_indices

        for modality in ("s1", "s2", "planet"):
            self._process_modality(patches_dict, modality, year_str, row, column, has_invalid_patches, patch_idx, year_offset)

        #### Post-processing
        patches_dict["non_empty_modalities"] = [mod for mod in ("s1", "s2", "planet") if len(patches_dict[mod]) > 0]

        if self.apply_transforms:
            self.transform(patches_dict)

        self._create_padding_masks(patches_dict)

        return patches_dict

    def _initialize_patches_dict(self):
        """Initialize the patches dictionary with empty lists and default values."""
        modalities = ["s1", "s2", "planet"]
        fields = ["", "_dates", "_doy", "_paths", "_paths_dropped", "_invalid_paths", "_padding_mask"]
        
        patches_dict = {f"{mod}{field}": [] for mod in modalities for field in fields}
        patches_dict.update({
            "device_tensor": torch.zeros(1),
            "non_empty_modalities": None,
            "labels": None
        })
        return patches_dict

    def _calculate_patch_position(self, patch_idx):
        """Calculate the row and column of the patch."""
        row = patch_idx // self.num_patches_horizontal
        column = patch_idx % self.num_patches_horizontal
        return row, column

    def _load_labels(self, year_str, row, column):
        """Load and preprocess labels for the given patch."""
        patch_size = self.modalities["s2"]["patch_size"]
        labels = self.labels[year_str][row*patch_size:(row+1)*patch_size,
                                    column*patch_size:(column+1)*patch_size]
        return labels - 1

    def _process_modality(self, patches_dict, modality, year_str, row, column, has_invalid_patches, patch_idx, year_offset):
        """Process a single modality (s1, s2, or planet)."""
        modality_dict = self.modalities[modality]
        modality_path = os.path.join(self.root_dir, modality, f"patches_{modality_dict['patch_size']}")
        
        paths = self._get_modality_paths(modality_dict, year_str, modality)
        valid_paths = self._handle_invalid_patches(patches_dict, modality, paths, has_invalid_patches, patch_idx, year_offset)
        valid_paths = self._apply_temporal_dropout(patches_dict, modality, valid_paths)
        
        self._process_valid_paths(patches_dict, modality, valid_paths, modality_dict, year_str, row, column, modality_path)
        
        if modality in self.active_modalities:
            self._load_image_patches(patches_dict, modality, has_invalid_patches, patch_idx, year_offset)

    def _get_modality_paths(self, modality_dict, year_str, modality):
        """Get the paths for the given modality."""
        paths = modality_dict["image_paths"][modality_dict["image_paths_years"] == year_str]
        if modality == "planet":
            return [os.path.basename(os.path.dirname(img_path)) for img_path in paths]
        return [os.path.basename(img_path).replace(".tif", "") for img_path in paths]

    def _handle_invalid_patches(self, patches_dict, modality, paths, has_invalid_patches, patch_idx, year_offset):
        """Handle invalid patches based on the invalid_patch_policy."""
        has_invalid_patches_modality = has_invalid_patches and modality in self.invalid_indices[patch_idx+year_offset]
        
        if has_invalid_patches_modality and self.invalid_patch_policy == "drop":
            invalid_indices = self.invalid_indices[patch_idx+year_offset][modality]
            valid_paths = np.array(paths)[~invalid_indices]
            patches_dict[f"{modality}_invalid_paths"] = np.array(paths)[invalid_indices].tolist()
        else:
            valid_paths = np.array(paths)
        
        return valid_paths

    def _apply_temporal_dropout(self, patches_dict, modality, valid_paths):
        """Apply temporal dropout if configured."""
        if self.temporal_dropout is not None and self.temporal_dropout[modality] > 0 and self.mode == "train":
            p_dropout = self.temporal_dropout[modality]
            dropout_mask = np.random.uniform(0, 1, len(valid_paths)) > p_dropout
            if not np.any(dropout_mask):
                dropout_mask[0] = True
            patches_dict[f"{modality}_paths_dropped"] = valid_paths[~dropout_mask].tolist()
            return valid_paths[dropout_mask]
        return valid_paths

    def _process_valid_paths(self, patches_dict, modality, valid_paths, modality_dict, year_str, row, column, modality_path):
        """Process valid paths for the given modality."""
        for day in valid_paths:
            path = os.path.join(modality_path, year_str, day, f"patch_{row}_{column}.tif")
            if not os.path.exists(path):
                print(f"WARN: path does not exist | {path}")
                continue
            
            datetime = modality_dict["get_date_time"](day)
            patches_dict[f"{modality}_dates"].append(datetime.strftime("%Y_%m_%d"))
            patches_dict[f"{modality}_doy"].append(datetime.timetuple().tm_yday)
            patches_dict[f"{modality}_paths"].append(path)

    def _load_image_patches(self, patches_dict, modality, has_invalid_patches, patch_idx, year_offset):
        """Load image patches for the given modality."""
        for path in patches_dict[f"{modality}_paths"]:
            if not os.path.exists(path):
                print(f"WARN: path does not exist | {path}")
                continue
            
            image_patch = tifffile.imread(path).transpose(2, 0, 1)
            patches_dict[modality].append(image_patch)
            
            # if self.single_temporal:
            #     break
        
        if patches_dict[modality]:
            patches_dict[modality] = np.stack(patches_dict[modality])
            if has_invalid_patches and self.invalid_patch_policy == "fill":
                patches_dict[modality] = np.nan_to_num(patches_dict[modality], nan=0.)

    def _create_padding_masks(self, patches_dict):
        """Create padding masks for each modality."""
        for modality in ("s1", "s2", "planet"):
            patches_dict[f"{modality}_padding_mask"] = torch.zeros(len(patches_dict[modality]), dtype=torch.bool)
            patches_dict[f"{modality}_doy"] = torch.tensor(patches_dict[f"{modality}_doy"])


    def save_patches_to_files(self):
        output_root = self.root_dir
        for modality in self.active_modalities:#("s1", "s2", "planet"):
            modality_dict = self.modalities[modality]
            # if modality == "planet":
            #     continue
            for k, img_path in enumerate(tqdm(modality_dict['image_paths'])):
                # Extract year and filename for directory structure
                path_name = os.path.basename(os.path.dirname(img_path)) if modality == "planet" else os.path.basename(img_path)
                year = str(modality_dict['get_date_time'](path_name).year)
                file_name = os.path.basename(path_name).split('.')[0]  # Assuming filename without extension

                # Calculate number of patches per image
                image = tifffile.imread(img_path)
                if not modality_dict['channel_last_data']:
                    image = image.transpose(1, 2, 0)  # Convert to channel last if needed

                mask = None
                if modality == "planet":
                    # resize planet
                    (h_s2, w_s2) = self.modalities["s2"]["hw"]
                    image = cv2.resize(image, dsize=(h_s2*self.planet_ratio, w_s2*self.planet_ratio)[::-1], interpolation=cv2.INTER_CUBIC)
                    # mask = tifffile.imread(self.modalities["planet"]["mask_paths"][k])
                    # if not modality_dict['channel_last_data']:
                    #     mask = mask.transpose(1, 2, 0)  # Convert to channel last if needed

                patch_size = modality_dict['patch_size']
                num_patches_vertical = self.num_patches_vertical
                num_patches_horizontal = self.num_patches_horizontal

                # Create directory path for the patches
                patch_dir = os.path.join(output_root, modality, f'patches_{patch_size}', year, file_name)
                os.makedirs(patch_dir, exist_ok=True)

                # Generate patches and save them
                for i in range(num_patches_vertical):
                    for j in range(num_patches_horizontal):
                        # Extract and save image patch
                        image_patch = extract_patch(image, i, j, patch_size)
                        image_patch = pad_patch(image_patch, patch_size)
                        save_patch(image_patch, patch_dir, f'patch_{i}_{j}.tif')
                        
                        # Extract and save mask patch if mask is provided
                        if mask is not None:
                            mask_patch = extract_patch(mask, i, j, patch_size)
                            mask_patch = pad_patch(mask_patch, patch_size)
                            save_patch(mask_patch, patch_dir, f'mask_{i}_{j}.tif')

    def __len__(self):
        if self.indices_path:
            # TODO fix
            # if self.mode == "train":
            #     return 1
            # elif self.mode == "validation":
            #     return 5
            return len(self.indices)
        else:
            return np.cumsum(np.fromiter(self.year_n_lengths.values(), dtype=int))[-1]

    def __getitem__(self, index):
        if self.indices_path:
            flatten_idx = self.indices[index]
        else:
            flatten_idx = index
        year_str, patch_idx, year_offset = self.determine_year_and_patch(flatten_idx)
        return self.load_patch(patch_idx=patch_idx, year_str=year_str, year_offset=year_offset, dataset_idx=index)
            
if __name__ == "__main__":
    # image_path = '/beegfs/work/y0092788/data/planet/Dataset for Dead Trees/4_band_normalized_dataset/2020/normalized_2020_august_6_psscene_analytic_sr_udm2/composite.tif'  # Update this to your .tif file path
    # dataset = TifDataset(image_path, patch_size=196)
    image_path = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    dataset = MultiModalHarzDataset(image_path, patch_size=128, active_modalities=["planet"], apply_transforms=False, planet_ratio=1)#, invalid_indices_path="/beegfs/work/y0092788/invalid_paths.yaml")
    # test = dataset[0]
    # dataset.save_patches_to_files()
    # analyse_mask(dataset)
    # data_root = '/beegfs/work/y0092788/data/'  # Update this to your .tif file path
    # invalid_indices_path = "/beegfs/work/y0092788/invalid_paths.yaml"
    # splits_path = "/beegfs/work/y0092788/indices_config_p128.yaml"
    # train_dataset = MultiModalHarzDataset(data_root, patch_size=128, mode="train", invalid_patch_policy="drop", 
    #                                     invalid_indices_path=invalid_indices_path, indices_path=splits_path,
    #                                     active_modalities=["s1", "s2"], single_temporal=True)
    # for i in range(len(train_dataset)):
    #     test = train_dataset[i]