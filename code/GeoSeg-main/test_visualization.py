import torch
import random
import argparse
from pytorch_lightning import Trainer
from utils.sample_visualization import visualize_sample_with_predictions
from train_supervision import Supervision_Train
from tools.cfg import py2cfg
import numpy as np
from torch.utils.data import Subset

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def test_and_visualize(config, checkpoint_path, out_dir, num_samples=5):
    # Load the model from checkpoint
    model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config, out_dir=out_dir)
    model.eval()

    # Get the test dataset
    test_dataset = config.test_dataset

    # Generate random indices
    total_samples = len(test_dataset)
    np.random.seed(0)
    indices = np.random.permutation(total_samples)[:num_samples]

    # # Create a subset of the test dataset with the selected indices
    # selected_samples = Subset(test_dataset, indices)

    # Visualize the selected samples
    for i, idx in enumerate(indices):
        sample = test_dataset[idx]
        
        # Move sample to the same device as the model
        # sample = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

        for modality in config.active_modalities:
            sample[modality] = sample[modality].unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            prediction = model(sample)
            logits = prediction["logits"]
            pre_mask = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1)

        # Denormalize the input images if necessary
        if test_dataset.apply_transforms:
            for modality in config.active_modalities:
                mean, std = test_dataset.modalities[modality]["mean"], test_dataset.modalities[modality]["std"]
                sample[modality] = denormalize(sample[modality].cpu(), mean=mean, std=std)

        # Visualize the sample
        visualize_sample_with_predictions(
            sample, 
            idx, 
            test_dataset, 
            pre_mask.cpu(), 
            epoch=0,  # We're not in training, so epoch doesn't matter
            out_dir=out_dir, 
            mode="test_visualization"
        )

    print(f"Visualized {num_samples} random samples from the test set.")

def main():
    parser = argparse.ArgumentParser(description="Test and visualize model predictions")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("out_dir", type=str, help="Path to theoutput directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()

    config = py2cfg(args.config_path)
    test_and_visualize(config, args.checkpoint_path, args.out_dir, args.num_samples)

if __name__ == "__main__":
    main()
