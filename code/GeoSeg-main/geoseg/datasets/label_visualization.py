import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import TifDatasetNew

sns.set_theme()

def plot_label_distribution_ratio(label_map, class_names):
    """
    Plots the ratio of pixel classes in a label map using Seaborn.

    Parameters:
    label_map (numpy.ndarray): A 2D NumPy array where each value represents a class label.
    class_names (list): A list of class names corresponding to the class labels.
    """
    # Flatten the label map to a 1D array
    labels = label_map.flatten()
    
    # Calculate the ratio of each class
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = labels.size
    ratios = counts / total_pixels
    
    # Create a bar plot for the ratios
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=unique, y=ratios)
    
    # Set plot labels and title
    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Pixel Ratio')
    ax.set_title('Pixel Distribution Ratio for Classes')
    
    # Set custom class names on the x-axis
    ax.set_xticklabels(class_names)
    
    # Show the plot
    plt.savefig("class_dist_ratio.png")


def plot_label_distribution(label_map, class_names):
    """
    Plots the distribution of pixel classes in a label map using Seaborn.

    Parameters:
    label_map (numpy.ndarray): A 2D NumPy array where each value represents a class label.
    class_names (list): A list of class names corresponding to the class labels.
    """
    # Flatten the label map to a 1D array
    labels = label_map.flatten()
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create a count plot for the labels
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=unique, y=counts)
    ax.set_yscale("log")
    
    # Set plot labels and title
    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Pixel Distribution for Classes')
    
    # Set custom class names on the x-axis
    ax.set_xticklabels(class_names)
    
    # Show the plot
    plt.savefig("class_dist.png")


if __name__ == "__main__":
    modality = "s1" # placeholder
    image_path = f'/beegfs/work/y0092788/data/{modality}'  # Update this to your .tif file path
    dataset = TifDatasetNew(image_path, patch_size=196, is_channel_last=True)#, day_directory=True, mask_prefix="composite_udm2", image_prefix="composite")

    plot_label_distribution(dataset.labels, dataset.class_names)
    plot_label_distribution_ratio(dataset.labels, dataset.class_names)
