import matplotlib.pyplot as plt
import pandas as pd

def plot_miou_from_csv(csv_path, output_path):
    # Load the csv file
    data = pd.read_csv(csv_path)
    
    # Group by epoch and take the mean to merge rows
    data_grouped = data.groupby('epoch', dropna=True).first()
    
    # Plot train and validation mIoU over epochs
    plt.figure(figsize=(10,6))
    plt.plot(data_grouped.index, data_grouped['train_mIoU'], label='Train mIoU')
    plt.plot(data_grouped.index, data_grouped['val_mIoU'], label='Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Train and Validation mIoU over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory

def plot_loss_from_csv(csv_path, output_path):
    # Load the csv file
    data = pd.read_csv(csv_path)
    
    # Group by epoch and take the mean to merge rows
    data_grouped = data.groupby('epoch', dropna=True).first()
    
    # Plot train and validation mIoU over epochs
    plt.figure(figsize=(10,6))
    plt.plot(data_grouped.index, data_grouped['train_loss_all_epoch'], label='Train loss')
    plt.plot(data_grouped.index, data_grouped['val_loss_all_epoch'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Train and Validation loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory

def plot_aux_loss_from_csv(csv_path, output_path, modalities):
    # Load the csv file
    data = pd.read_csv(csv_path)
    
    # Group by epoch and take the mean to merge rows
    data_grouped = data.groupby('epoch', dropna=True).first()
    
    # Plot train and validation mIoU over epochs
    plt.figure(figsize=(10,6))
    plt.plot(data_grouped.index, data_grouped['train_loss_main_epoch'], label='Train loss')
    plt.plot(data_grouped.index, data_grouped['val_loss_main_epoch'], label='Validation loss')

    for modality in modalities:
        plt.plot(data_grouped.index, data_grouped[f'train_loss_{modality}_epoch'], label=f'{modality} Train loss')
        plt.plot(data_grouped.index, data_grouped[f'val_loss_{modality}_epoch'], label=f'{modality} Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Train and Validation loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory

# Example usage:
# plot_miou_from_csv('metrics.csv', 'output_plot.png')
