import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
from tabulate import tabulate

import seaborn as sns



sns.set_theme(font_scale=1.5)

def create_miou_bar_plots_temporal_subsets(experiment_folders):
    modalities = []
    temporal_types = []
    miou_values = []

    for folder in experiment_folders:
        parts = folder.split('/')[-1].split('_')
        temporal_type = {
            'mt': 'Max Pooling',
            'st': 'Single Temporal',
            'ltae': 'LTAE'
        }.get(parts[0], 'Unknown')
        
        modality = parts[1]

        # Extract the base name from the path
        base_name = os.path.basename(folder)

        # Split the base name by underscores
        parts = base_name.split('_')

        # Join the relevant parts to get the experiment name
        experiment_name = '_'.join(parts[:-2])

        version_1_path = os.path.join(folder, 'testing', experiment_name, 'version_1', 'metrics.csv')
        
        df = pd.read_csv(version_1_path)
        miou = df['test_mIoU'].iloc[-1]
        
        modalities.append(modality)
        temporal_types.append(temporal_type)
        miou_values.append(miou)

    plot_data = pd.DataFrame({
        'Modality': modalities,
        'Temporal Type': temporal_types,
        'Test mIoU': miou_values
    })

    modality_order = ['s1', 's2', 'planet', 'all']

    def create_bar_plot(data, title, filename, legend=False):
        plt.figure(figsize=(15, 10))
        ax = sns.barplot(x='Modality', y='Test mIoU', data=data, order=modality_order)
        plt.title(title)
        plt.xlabel('Modality')
        plt.ylabel('Test mIoU')
        plt.xticks(rotation=45)
        if legend:
            ax.legend(title='Temporal Type', loc='upper right')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x='Modality', y='Test mIoU', hue='Temporal Type', data=plot_data, order=modality_order)
    plt.title('Test mIoU Comparison Across Modalities and Temporal Types')
    plt.xlabel('Modality')
    plt.ylabel('Test mIoU')
    plt.xticks(rotation=45)
    ax.legend(title='Temporal Type', loc='upper left')
    plt.tight_layout()
    out_file = os.path.join("visuals", f"st_mt_modalities.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

    st_data = plot_data[plot_data['Temporal Type'] == 'Single Temporal']
    out_file = os.path.join("visuals", f"st_modalities.png")
    create_bar_plot(st_data, 'Test mIoU Comparison for Single-temporal Modalities', out_file)

    mt_data = plot_data[plot_data['Temporal Type'] == 'Max Pooling']
    out_file = os.path.join("visuals", f"mt_modalities.png")
    create_bar_plot(mt_data, 'Test mIoU Comparison for Multi-temporal Modalities', out_file)

def plot_miou_over_epochs_w_test(experiment_paths, experiment_names, filename):
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", n_colors=len(experiment_paths))
    
    if experiment_names is None:
        experiment_names = [f"Experiment {i+1}" for i in range(len(experiment_paths))]
    
    for i, (experiment_path, exp_name) in enumerate(zip(experiment_paths, experiment_names)):
        # Extract the base name from the path
        base_name = os.path.basename(experiment_path)

        # Split the base name by underscores
        parts = base_name.split('_')

        # Join the relevant parts to get the experiment name
        experiment_name = '_'.join(parts[:-2])

        version_0_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_0', 'metrics.csv')
        version_1_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_1', 'metrics.csv')
        
        train_val_df = pd.read_csv(version_0_path)
        test_df = pd.read_csv(version_1_path)
        
        sns.lineplot(data=train_val_df, x='epoch', y='train_mIoU', 
                     label=f'{exp_name} (Train)', color=colors[i])
        
        sns.lineplot(data=train_val_df, x='epoch', y='val_mIoU', 
                     linestyle=':', label=f'{exp_name} (Val)', color=colors[i])
        
        test_miou = test_df['test_mIoU'].iloc[-1]
        plt.axhline(y=test_miou, color=colors[i], linestyle='--', 
                    label=f'{exp_name} (Test)')
    
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Train, Validation, and Test mIoU over Epochs')
    plt.legend(title="Experiments", loc='lower right')#, bbox_to_anchor=(1.05, 1), loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    out_file = os.path.join("visuals", f"{filename}.png")
    plt.savefig(out_file, bbox_inches='tight')


def collect_best_miou_scores_and_losses(experiment_paths, experiment_names):
    if experiment_names is None:
        experiment_names = [f"Experiment {i+1}" for i in range(len(experiment_paths))]
    
    results = []
    
    for i, (experiment_path, exp_name) in enumerate(zip(experiment_paths, experiment_names)):
        base_name = os.path.basename(experiment_path)
        parts = base_name.split('_')
        experiment_name = '_'.join(parts[:-2])

        version_0_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_0', 'metrics.csv')
        version_1_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_1', 'metrics.csv')
        
        train_val_df = pd.read_csv(version_0_path)
        test_df = pd.read_csv(version_1_path)
        
        best_train_miou = train_val_df['train_mIoU'].max()
        best_val_miou = train_val_df['val_mIoU'].max()
        best_test_miou = test_df['test_mIoU'].max()
        
        # Check for main loss columns first, if not present use regular loss columns
        train_loss_col = 'train_loss_main_epoch' if 'train_loss_main_epoch' in train_val_df.columns else 'train_loss_all_epoch'
        val_loss_col = 'val_loss_main_epoch' if 'val_loss_main_epoch' in train_val_df.columns else 'val_loss_all_epoch'
        test_loss_col = 'test_loss_main_epoch' if 'test_loss_main_epoch' in test_df.columns else 'test_loss_all_epoch'
        
        best_train_loss = train_val_df[train_loss_col].min()
        best_val_loss = train_val_df[val_loss_col].min()
        best_test_loss = test_df[test_loss_col].min()
        
        results.append([exp_name, best_train_miou, best_val_miou, best_test_miou, 
                        best_train_loss, best_val_loss, best_test_loss])
    
    headers = ["Experiment", "mIoU Train", "mIoU Val", "mIoU Test", 
               "Loss Train", "Loss Val", "Loss Test"]
    table = tabulate(results, headers=headers, tablefmt="grid", floatfmt=".4f")
    
    print(table)


def plot_miou_over_epochs(csv_files, experiment_names, filename):
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", n_colors=len(csv_files))
    
    if experiment_names is None:
        experiment_names = [f"Experiment {i+1}" for i in range(len(csv_files))]
    
    for i, (file, exp_name) in enumerate(zip(csv_files, experiment_names)):
        train_val_file = file.replace('version_1', 'version_0')
        df = pd.read_csv(train_val_file)
        
        sns.lineplot(data=df, x='epoch', y='train_mIoU', 
                     label=f'{exp_name} (Train)', color=colors[i])
        
        sns.lineplot(data=df, x='epoch', y='val_mIoU', 
                     linestyle=':', label=f'{exp_name} (Val)', color=colors[i])
    
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Train and Validation mIoU over Epochs')
    plt.legend(title="Experiment", loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    out_file = os.path.join("visuals", f"{filename}.png")
    plt.savefig(out_file, bbox_inches='tight')

def visualize_class_miou_seaborn(experiment_paths, experiment_names, filename, flip=True):
    experiments = experiment_names
    class_mious = []
    test_mious = []
    
    class_names = ['Built-up', 'Conifer', 'Crop', 'Dead tree', 'Deciduous', 'Grass', 'Water']
    
    for experiment_path in experiment_paths:
        # Extract the base name from the path
        base_name = os.path.basename(experiment_path)

        # Split the base name by underscores
        parts = base_name.split('_')

        # Join the relevant parts to get the experiment name
        experiment_name = '_'.join(parts[:-2])

        version_1_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_1', 'metrics.csv')
        
        df = pd.read_csv(version_1_path)
        
        test_iou_row = df.iloc[-1]
        
        class_iou = [test_iou_row[f'IoU_test_{class_name}'] for class_name in class_names if class_name != "Tree"]
        class_mious.append(class_iou)
        
        test_mious.append(test_iou_row['test_mIoU'])
    
    class_mious = np.array(class_mious)

    data = []
    for exp, miou, test_miou in zip(experiments, class_mious, test_mious):
        for class_name, value in zip(class_names, miou):
            data.append({'Experiment': exp, 'Class': class_name, 'IoU': value})
        data.append({'Experiment': exp, 'Class': 'mIoU', 'IoU': test_miou})
    
    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 12))
    
    sns.barplot(x='Class', y='IoU', hue='Experiment', data=df)
    
    plt.title('Class-wise IoU and Test mIoU for Different Experiments')
    if flip:
        plt.xticks(rotation=45, ha='right')
    plt.legend(title='Modality', loc='lower right')
    
    plt.tight_layout()
    out_file = os.path.join("visuals", f"{filename}.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def visualize_class_miou_seaborn_pres(experiment_paths, experiment_names, filename, flip=True):
    experiments = experiment_names
    class_mious = []
    test_mious = []
    
    class_names = ['Built-up', 'Conifer', 'Crop', 'Dead tree', 'Deciduous', 'Grass', 'Water']
    
    for experiment_path in experiment_paths:
        # Extract the base name from the path
        base_name = os.path.basename(experiment_path)

        # Split the base name by underscores
        parts = base_name.split('_')

        # Join the relevant parts to get the experiment name
        experiment_name = '_'.join(parts[:-2])

        version_1_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_1', 'metrics.csv')
        
        df = pd.read_csv(version_1_path)
        
        test_iou_row = df.iloc[-1]
        
        class_iou = [test_iou_row[f'IoU_test_{class_name}'] for class_name in class_names if class_name != "Tree"]
        class_mious.append(class_iou)
        
        test_mious.append(test_iou_row['test_mIoU'])
    
    class_mious = np.array(class_mious)

    data = []
    for exp, miou, test_miou in zip(experiments, class_mious, test_mious):
        for class_name, value in zip(class_names, miou):
            data.append({'Experiment': exp, 'Class': class_name, 'IoU': value})
        data.append({'Experiment': exp, 'Class': 'mIoU', 'IoU': test_miou})
    
    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 12))
    
    sns.barplot(x='Class', y='IoU', data=df)
    
    plt.title(f'Class-wise IoU and Test mIoU for {experiment_names[0]}')
    if flip:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    out_file = os.path.join("visuals", f"{filename}.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def visualize_class_miou_auxiliary_seaborn(experiment_paths, filename, flip=True):
    classification_heads = ["S1", "S2", "Planet", "All", "All - No Auxiliary"]
    class_names = ['Built-up', 'Conifer', 'Crop', 'Dead tree', 'Deciduous', 'Grass', 'Water']
    
    # Read the CSV data
    df = pd.read_csv(experiment_paths["main"])
    df_no_aux = pd.read_csv(experiment_paths["All - No Auxiliary"])
    
    data = []
    for head in classification_heads:
        if head == "All":
            class_iou = [df[f'IoU_test_{class_name}'].iloc[-1] for class_name in class_names]
            mIoU = df['test_mIoU'].iloc[-1]
        elif head == "All - No Auxiliary":
            class_iou = [df_no_aux[f'IoU_test_{class_name}'].iloc[-1] for class_name in class_names]
            mIoU = df_no_aux['test_mIoU'].iloc[-1]
        else:
            lowercase_head = head.lower()
            class_iou = [df[f'IoU_{lowercase_head}_test_{class_name}'].iloc[-1] for class_name in class_names]
            mIoU = df[f'test_{lowercase_head}_mIoU'].iloc[-1]
        
        for class_name, value in zip(class_names, class_iou):
            data.append({'Classification Head': head, 'Class': class_name, 'IoU': value})
        
        # Add mIoU for each classification head
        data.append({'Classification Head': head, 'Class': 'mIoU', 'IoU': mIoU})
    
    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(15, 10))
    
    sns.barplot(x='Class', y='IoU', hue='Classification Head', data=df_plot)
    
    plt.title('Class-wise IoU and Test mIoU for Different Classification Heads')
    if flip:
        plt.xticks(rotation=45, ha='right')
    plt.legend(title='Classification Head', loc='lower right')
    
    plt.tight_layout()
    out_file = os.path.join("visuals", f"{filename}.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def visualize_class_miou(log_files):
    experiments = []
    class_mious = []
    
    class_names = ['Built-up', 'Conifer', 'Crop', 'Dead tree', 'Deciduous', 'Grass', 'Tree', 'Water']
    
    for log_file in log_files:
        exp_name = os.path.basename(os.path.dirname(log_file))
        experiments.append(exp_name)
        
        test_file = log_file.replace('version_0', 'version_1')
        df = pd.read_csv(test_file)
        
        test_iou_row = df.iloc[-1]
        
        class_iou = [test_iou_row[f'IoU_test_{class_name}'] for class_name in class_names]
        class_mious.append(class_iou)
    
    class_mious = np.array(class_mious)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.8 / len(experiments)
    
    for i, (exp, miou) in enumerate(zip(experiments, class_mious)):
        ax.bar(x + i*width, miou, width, label=exp)
    
    ax.set_ylabel('Mean IoU')
    ax.set_title('Class-wise IoU for Different Experiments')
    ax.set_xticks(x + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig("test.png")
    plt.close(fig)

def plot_aux_loss_from_csv(experiment_path, output_path, modalities=["s1", "s2", "planet"]):

    # Extract the base name from the path
    base_name = os.path.basename(experiment_path)

    # Split the base name by underscores
    parts = base_name.split('_')

    # Join the relevant parts to get the experiment name
    experiment_name = '_'.join(parts[:-2])

    version_0_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_0', 'metrics.csv')
    
    data = pd.read_csv(version_0_path)

    # Load the csv file
    # data = pd.read_csv(csv_path)
    
    # Group by epoch and take the mean to merge rows
    data_grouped = data.groupby('epoch', dropna=True).first()
    
    # Plot train and validation mIoU over epochs
    plt.figure(figsize=(10,8))
    plt.plot(data_grouped.index, data_grouped['train_loss_main_epoch'], label='Main Head Train loss')
    plt.plot(data_grouped.index, data_grouped['val_loss_main_epoch'], label='Main Head Validation loss')

    for modality in modalities:
        plt.plot(data_grouped.index, data_grouped[f'train_loss_{modality}_epoch'], label=f'{modality[:1].upper()}{modality[1:]} Head Train loss')
        plt.plot(data_grouped.index, data_grouped[f'val_loss_{modality}_epoch'], label=f'{modality[:1].upper()}{modality[1:]} Head Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Train and Validation loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    out_file = os.path.join("visuals", f"{output_path}.png")
    plt.savefig(out_file)
    plt.close()  # Close the figure to free up memory

# def create_miou_bar_plots_temporal_subsets(experiment_folders):
#     # Initialize lists to store data
#     modalities = []
#     temporal_types = []
#     miou_values = []

#     # Process each experiment folder
#     for folder in experiment_folders:
#         # Extract modality and temporal type from folder name
#         parts = folder.split('/')[-1]
#         parts = parts.split('_')
#         match parts[0]:
#             case 'mt':
#                 temporal_type = 'Max Pooling'
#             case 'st':
#                 temporal_type = 'Single Temporal'
#             case 'ltae':
#                 temporal_type = 'LTAE'
                
#         modality = parts[1]

#         # Construct the full path to the CSV file
#         csv_path = os.path.join(folder, 'testing', f"{parts[0]}_{parts[1]}", 'version_0', 'metrics.csv')

#         df = pd.read_csv(csv_path)
        
#         # Get the test mIoU value
#         miou = df['test_mIoU'].iloc[-1]  # Assuming the last row has the final mIoU
        
#         # Append data to lists
#         modalities.append(modality)
#         temporal_types.append(temporal_type)
#         miou_values.append(miou)

#     # Create a DataFrame from the collected data
#     plot_data = pd.DataFrame({
#         'Modality': modalities,
#         'Temporal Type': temporal_types,
#         'Test mIoU': miou_values
#     })

#     # Define the desired order of modalities
#     modality_order = ['s1', 's2', 'planet', 'all']

#     # Function to create and save a bar plot
#     def create_bar_plot(data, title, filename, legend=False):
#         plt.figure(figsize=(16, 9))
#         ax = sns.barplot(x='Modality', y='Test mIoU', data=data, order=modality_order)
#         plt.title(title)
#         plt.xlabel('Modality')
#         plt.ylabel('Test mIoU')
#         plt.xticks(rotation=45)
#         if legend:
#             ax.legend(title='Temporal Type', loc='upper right')
#         plt.tight_layout()
#         plt.savefig(filename, bbox_inches='tight')
#         plt.close()

#     # Create and save the combined plot
#     plt.figure(figsize=(16, 9))
#     ax = sns.barplot(x='Modality', y='Test mIoU', hue='Temporal Type', data=plot_data, order=modality_order)
#     plt.title('Test mIoU Comparison Across Modalities and Temporal Types')
#     plt.xlabel('Modality')
#     plt.ylabel('Test mIoU')
#     plt.xticks(rotation=45)
#     ax.legend(title='Temporal Type', loc='upper left')
#     plt.tight_layout()
#     out_file = os.path.join("visuals", f"st_mt_modalities.png")
#     plt.savefig(out_file, bbox_inches='tight')
#     plt.close()

#     # Create and save single-temporal plot
#     st_data = plot_data[plot_data['Temporal Type'] == 'Single-temporal']
#     out_file = os.path.join("visuals", f"st_modalities.png")
#     create_bar_plot(st_data, 'Test mIoU Comparison for Single-temporal Modalities', out_file)

#     # Create and save multi-temporal plot
#     mt_data = plot_data[plot_data['Temporal Type'] == 'Multi-temporal']
#     out_file = os.path.join("visuals", f"mt_modalities.png")
#     create_bar_plot(mt_data, 'Test mIoU Comparison for Multi-temporal Modalities', out_file)

# # Example usage:
# experiment_folders = [
#     'mt_all_2024-08-01_07-19-13',
#     'mt_s1_2024-07-30_11-57-29',
#     'st_all_2024-08-02_09-15-22',
#     'st_s1_2024-07-31_14-30-45',
#     'mt_s2_2024-08-03_16-42-37',
#     'st_s2_2024-08-04_08-11-59',
#     'mt_planet_2024-08-05_13-27-18',
#     'st_planet_2024-08-06_10-05-33'
# ]


# def plot_miou_over_epochs_w_test(csv_files, experiment_names, filename):
#     plt.figure(figsize=(12, 8))
    
#     # Define a color palette
#     colors = sns.color_palette("husl", n_colors=len(csv_files))
    
#     if experiment_names is None:
#         experiment_names = [f"Experiment {i+1}" for i in range(len(csv_files))]
    
#     for i, (file, exp_name) in enumerate(zip(csv_files, experiment_names)):
#         df = pd.read_csv(file)
        
#         # Plot train mIoU with solid line
#         sns.lineplot(data=df, x='epoch', y='train_mIoU', 
#                      label=f'{exp_name} (Train)', color=colors[i])
        
#         # Plot validation mIoU with dotted line
#         sns.lineplot(data=df, x='epoch', y='val_mIoU', 
#                      linestyle=':', label=f'{exp_name} (Val)', color=colors[i])
        
#         # Plot test mIoU as a horizontal line with crosses as markers
#         if 'test_mIoU' in df.columns:
#             test_miou = df['test_mIoU'].iloc[-1]  # Assuming the last value is the final test mIoU
#             plt.axhline(y=test_miou, color=colors[i], linestyle='--', 
#                         label=f'{exp_name} (Test)')
#             # plt.plot(df['epoch'], [test_miou] * len(df), 'x', color=colors[i], 
#             #          markersize=8, markeredgewidth=2)
    
#     plt.xlabel('Epoch')
#     plt.ylabel('mIoU')
#     plt.title('Train, Validation, and Test mIoU over Epochs')
#     plt.legend(title="Experiments", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()
#     out_file = os.path.join("visuals", f"{filename}.png")
#     plt.savefig(out_file, bbox_inches='tight')

# def plot_miou_over_epochs(csv_files, experiment_names, filename):
#     plt.figure(figsize=(12, 8))
    
#     # Define a color palette
#     colors = sns.color_palette("husl", n_colors=len(csv_files))
    
#     if experiment_names is None:
#         experiment_names = [f"Experiment {i+1}" for i in range(len(csv_files))]
    
#     for i, (file, exp_name) in enumerate(zip(csv_files, experiment_names)):
#         df = pd.read_csv(file)
        
#         # Plot train mIoU with solid line
#         sns.lineplot(data=df, x='epoch', y='train_mIoU', 
#                      label=f'{exp_name} (Train)', color=colors[i])
        
#         # Plot validation mIoU with dotted line
#         sns.lineplot(data=df, x='epoch', y='val_mIoU', 
#                      linestyle=':', label=f'{exp_name} (Val)', color=colors[i])
    
#     plt.xlabel('Epoch')
#     plt.ylabel('mIoU')
#     plt.title('Train and Validation mIoU over Epochs')
#     # plt.legend(title="Experiments", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.legend(title="Experiment", loc='lower right')
#     plt.grid(True)
#     plt.tight_layout()
#     out_file = os.path.join("visuals", f"{filename}.png")
#     plt.savefig(out_file, bbox_inches='tight')


# def visualize_class_miou_seaborn(log_files, experiment_names, filename, flip=True):
#     experiments = experiment_names
#     class_mious = []
#     test_mious = []
    
#     class_names = ['Built-up', 'Conifer', 'Crop', 'Dead tree', 'Deciduous', 'Grass', 'Tree', 'Water']
    
#     for log_file in log_files:
#         # Read the metrics.csv file
#         df = pd.read_csv(log_file)
        
#         # Find the last row with test IoU values
#         test_iou_row = df.iloc[-1]
        
#         # Extract class IoU values
#         class_iou = [test_iou_row[f'IoU_test_{class_name}'] for class_name in class_names]
#         class_mious.append(class_iou)
        
#         # Extract test mIoU
#         test_mious.append(test_iou_row['test_mIoU'])
    
#     # Convert to numpy array for easier manipulation
#     class_mious = np.array(class_mious)

#     # Convert data to long format for seaborn
#     data = []
#     for exp, miou, test_miou in zip(experiments, class_mious, test_mious):
#         for class_name, value in zip(class_names, miou):
#             data.append({'Experiment': exp, 'Class': class_name, 'Mean IoU': value})
#         # Add test mIoU as a separate "class"
#         data.append({'Experiment': exp, 'Class': 'mIoU', 'Mean IoU': test_miou})
    
#     df = pd.DataFrame(data)

#     # Set up the plot
#     plt.figure(figsize=(14, 8))
    
#     # Create vertical bar plot using seaborn
#     sns.barplot(x='Class', y='Mean IoU', hue='Experiment', data=df)
    
#     # Customize the plot
#     plt.title('Class-wise Mean IoU and Test mIoU for Different Experiments')
#     if flip:
#         plt.xticks(rotation=45, ha='right')
#     # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.legend(loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig(f"{filename}.png", bbox_inches='tight')
    


# def visualize_class_miou(log_files):
#     experiments = []
#     class_mious = []
    
#     class_names = ['Built-up', 'Conifer', 'Crop', 'Dead tree', 'Deciduous', 'Grass', 'Tree', 'Water']
    
#     for log_file in log_files:
#         # Extract experiment name from log file path
#         exp_name = os.path.basename(os.path.dirname(log_file))
#         experiments.append(exp_name)
        
#         # Read the metrics.csv file
#         df = pd.read_csv(log_file)
        
#         # Find the last row with test IoU values
#         test_iou_row = df.iloc[-1]
        
#         # Extract class IoU values
#         class_iou = [test_iou_row[f'IoU_test_{class_name}'] for class_name in class_names]
#         class_mious.append(class_iou)
    
#     # Convert to numpy array for easier manipulation
#     class_mious = np.array(class_mious)
    
#     # Set up the plot
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Create grouped bar plot
#     x = np.arange(len(class_names))
#     width = 0.8 / len(experiments)
    
#     for i, (exp, miou) in enumerate(zip(experiments, class_mious)):
#         ax.bar(x + i*width, miou, width, label=exp)
    
#     # Customize the plot
#     ax.set_ylabel('Mean IoU')
#     ax.set_title('Class-wise Mean IoU for Different Experiments')
#     ax.set_xticks(x + width * (len(experiments) - 1) / 2)
#     ax.set_xticklabels(class_names, rotation=45, ha='right')
#     ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
#     plt.tight_layout()
#     plt.savefig("test.png")
#     plt.close(fig)


# SINGLE_MODAL = [
#     "/beegfs/work/y0092788/logs/ablations_subsets/mt_s1_e40_lr_075_crop_dropout_04_nasa_2024-07-22_13-07-31/testing/mt_s1_e40_lr_075_crop_dropout_04_nasa/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/ablations_subsets/mt_s2_e40_lr_075_crop_dropout_04_nasa_2024-07-22_08-09-26/testing/mt_s2_e40_lr_075_crop_dropout_04_nasa/version_0/metrics.csv",
#     # "/beegfs/work/y0092788/logs/ablations_subsets/mt_planet_128_e40_lr_075_crop_075_dropout_04_nasa_2024-07-22_08-09-09/testing/mt_planet_128_e40_lr_075_crop_075_dropout_04_nasa/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/ablations_subsets/mt_planet_e40_lr_075_crop_075_dropout_aux_sum_nasa_2024-07-22_15-33-19/testing/mt_planet_e40_lr_075_crop_075_dropout_aux_sum_nasa/version_0/metrics.csv",
# ]
# SINGLE_MODAL_NAMES = [
#     "S1",
#     "S2",
#     # "Planet (128)",
#     "Planet (384)",
# ]

SUBSETS_MODAL = [
    "/beegfs/work/y0092788/logs/modalities/mt_s2_planet_2024-09-15_11-04-12",
    "/beegfs/work/y0092788/logs/modalities/mt_s1_s2_2024-09-15_11-04-02",
    "/beegfs/work/y0092788/logs/modalities/mt_s1_s2_2024-09-15_11-04-02",
    "/beegfs/work/y0092788/logs/st_architecture/mt_all_ubarn_2024-09-14_08-33-45"
]
SUBSETS_MODAL_NAMES = [
    "Planet (384) + S2",
    "Planet (384) + S1",
    "S1 + S2",
    "ALL",
]
# TRAINING_PARADIGMS = [
#     "/beegfs/work/y0092788/logs/mt_planet_s1_s2_e40_lr_075_crop_075_dropout_sum_nasa_n_s_scratch_2024-07-24_15-40-15/testing/mt_planet_s1_s2_e40_lr_075_crop_075_dropout_sum_nasa_n_s_scratch/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/mt_planet_s1_s2_e40_lr_075_crop_075_dropout_sum_nasa_n_s_imagenet_2024-07-25_10-58-12/testing/mt_planet_s1_s2_e40_lr_075_crop_075_dropout_sum_nasa_n_s_imagenet/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/mt_planet_s1_s2_e40_lr_075_crop_075_dropout_sum_nasa_n_s_2024-07-23_18-45-06/testing/mt_planet_s1_s2_e40_lr_075_crop_075_dropout_sum_nasa_n_s/version_0/metrics.csv",
# ]
WEIGHTS = [
    "/beegfs/work/y0092788/logs/weights/ltae_s1_mt_s2_planet_scratch_2024-09-16_21-44-35",
    "/beegfs/work/y0092788/logs/weights/ltae_s1_mt_s2_planet_imagenet_2024-09-16_21-17-13",
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55",
]
WEIGHTS_NAMES = [
    "Scratch",
    "ImageNet",
    "Self-Supervised (Foundation Model)",
]
WEIGHTS_PRE = [
    "/beegfs/work/y0092788/logs/weights/ltae_s1_mt_s2_planet_imagenet_2024-09-16_21-17-13",
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55",
]
WEIGHTS_PRE_NAMES = [
    "ImageNet",
    "Self-Supervised (Foundation Model)",
]

TEMPORAL = [
    "/beegfs/work/y0092788/logs/st_architecture/mt_all_ubarn_2024-09-14_08-33-45",
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55",
    "/beegfs/work/y0092788/logs/temporal/ltae_all_2024-09-15_12-02-51",
]

TEMPORAL_NAMES = [
    "Temporal Max Pooling",
    "L-TAE S1 & Temporal Max Pooling S2/Planet",
    "L-TAE",
]

TEMPORAL_S1 = [
    "/beegfs/work/y0092788/logs/temporal/st_s1_2024-11-22_11-34-32",
    "/beegfs/work/y0092788/logs/modalities/mt_s1_2024-09-15_11-04-30",
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_2024-09-15_11-06-31",
]

TEMPORAL_NAMES_S1 = [
    "Single Temporal",
    "Temporal Max Pooling",
    "L-TAE",
]
TEMPORAL_S2 = [
    "/beegfs/work/y0092788/logs/temporal/st_s2_2024-11-22_11-22-30",
    "/beegfs/work/y0092788/logs/modalities/mt_s2_2024-09-15_11-04-21",
    "/beegfs/work/y0092788/logs/temporal/ltae_s2_2024-09-15_11-06-22",
]
TEMPORAL_PLANET = [
    "/beegfs/work/y0092788/logs/temporal/st_planet_2024-11-22_11-33-48",
    "/beegfs/work/y0092788/logs/modalities/mt_planet_2024-09-14_10-11-42",
    "/beegfs/work/y0092788/logs/temporal/ltae_planet_2024-09-15_12-06-39",
]
TEMPORAL_ALL = [
    "/beegfs/work/y0092788/logs/temporal/st_all_2024-11-22_11-33-59",
    "/beegfs/work/y0092788/logs/st_architecture/mt_all_ubarn_2024-09-14_08-33-45",
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55",
]

TEMPORAL_NAMES_ALL = [
    "Single Temporal",
    "Temporal Max Pooling",
    "L-TAE S1 & Temporal Max Pooling S2/Planet",
]

DROPOUT = [
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_drop_2024-09-12_16-46-22",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aug_2024-09-12_16-45-40",
]

DROPOUT_NAMES = [
    "w/ Temporal Dropout",
    "w/o Temporal Dropout",
]

AUG = [
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aug_2024-09-12_16-45-40",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_no_aug_2024-09-12_16-46-29",
]

AUG_NAMES = [
    "w/ Augmentations",
    "w/o Augmentations",
]

AUX = [
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_lower_sen1_2024-09-14_20-01-23",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_2024-09-12_16-45-47",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_drop_2024-09-12_16-46-22",
]

AUX_NAMES = [
    "w/ Auxiliary Head: S1 Weight: 0.25",
    "w/ Auxiliary Head",
    "w/o Auxiliary Heads",
]

# LTAE = [
#     "/beegfs/work/y0092788/logs/mt_all_2024-08-01_07-19-13/testing/mt_all/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/ltae_all_sum_2024-08-03_14-53-41/testing/ltae_all_sum/version_0/metrics.csv",
# ]

# LTAE_NAMES = [
#     "Max Pooling",
#     "LTAE",
# ]

LEARNING_RATE = [
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_low_lr_2024-09-17_12-41-36",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_no_aug_2024-09-12_16-46-29",
]

LEARNING_RATE_NAMES = [
    "Default LR",
    "Higher spatial encoder LR",
]

DECODER = [
    "/beegfs/work/y0092788/logs/spatial_decoder/mt_all_segformer_2024-09-18_08-51-41",
    "/beegfs/work/y0092788/logs/spatial_decoder/mt_all_prithvi_single_new_weights_2024-09-11_16-49-01",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_drop_2024-09-12_16-46-22"
    # "/beegfs/work/y0092788/logs/spatial_decoder/mt_all_non_shared_2024-08-29_11-47-41",
    # "/beegfs/work/y0092788/logs/spatial_decoder/mt_all_prithvi_double_2024-08-29_11-47-12"
]

DECODER_NAMES = [
    "SegFormer",
    "Pritvi Single Block",
    "UperNet",
    # "Pritvi Double Block",
]

ARCHITECTURE = [
    "/beegfs/work/y0092788/logs/fusion/mt_all_concat_2024-09-13_08-52-45",
    "/beegfs/work/y0092788/logs/st_architecture/mt_all_ubarn_2024-09-14_08-33-45",
]

ARCHITECTURE_NAMES = [
    "Spatial E -> Temporal E -> Spatial D (UTAE)",
    "Spatial E -> Spatial D -> Temporal E (UBARN)",
]

FUSION = [
    "/beegfs/work/y0092788/logs/spatial_decoder/mt_all_prithvi_single_new_weights_2024-09-11_16-49-01",
    "/beegfs/work/y0092788/logs/fusion/mt_all_concat_2024-09-13_08-52-45",
    "/beegfs/work/y0092788/logs/fusion/mt_all_mid_2024-09-13_10-10-39",
    # "/beegfs/work/y0092788/logs/ablations_fusion/mt_all_mid_2024-08-01_21-48-46/testing/mt_all_mid/version_0/metrics.csv",
    # "/beegfs/work/y0092788/logs/ablations_fusion/mt_all_mid_utae_2024-08-05_07-27-09/testing/mt_all_mid_utae/version_0/metrics.csv",
]

FUSION_NAMES = [
    "Late: Sum",
    "Late: Concat",
    "Mid",
    # "Mid UTAE",
]

ST_MT = [
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_2024-09-15_11-06-31",
    "/beegfs/work/y0092788/logs/temporal/ltae_s2_2024-09-15_11-06-22",
    "/beegfs/work/y0092788/logs/temporal/ltae_planet_2024-09-15_12-06-39",
    "/beegfs/work/y0092788/logs/temporal/ltae_all_2024-09-15_12-02-51",

    "/beegfs/work/y0092788/logs/modalities/mt_planet_2024-09-14_10-11-42",
    "/beegfs/work/y0092788/logs/modalities/mt_s1_2024-09-15_11-04-30",
    "/beegfs/work/y0092788/logs/modalities/mt_s2_2024-09-15_11-04-21",
    "/beegfs/work/y0092788/logs/st_architecture/mt_all_ubarn_2024-09-14_08-33-45",

    # old
    # "/beegfs/work/y0092788/logs/ablations_temporal/st_all_2024-08-02_08-37-35",
    # "/beegfs/work/y0092788/logs/ablations_temporal/st_planet_2024-07-30_19-41-15",
    # "/beegfs/work/y0092788/logs/ablations_temporal/st_s1_2024-07-30_19-41-50",
    # "/beegfs/work/y0092788/logs/ablations_temporal/st_s2_2024-07-30_19-41-54",
    # new
    "/beegfs/work/y0092788/logs/temporal/st_all_2024-11-22_11-33-59",
    "/beegfs/work/y0092788/logs/temporal/st_planet_2024-11-22_11-33-48",
    "/beegfs/work/y0092788/logs/temporal/st_s1_2024-11-22_11-34-32",
    "/beegfs/work/y0092788/logs/temporal/st_s2_2024-11-22_11-22-30",
]


CLASSES_VISUAL_NAMES = [
    "S1",
    "S2",
    "Planet",
    # "Planet & S1",
    "All"
]
CLASSES_VISUAL = [
    "/beegfs/work/y0092788/logs/modalities/mt_s1_2024-09-15_11-04-30",
    "/beegfs/work/y0092788/logs/modalities/mt_s2_2024-09-15_11-04-21",
    "/beegfs/work/y0092788/logs/modalities/mt_planet_2024-09-14_10-11-42",
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55"
]

TRAINING_REGULARIZATION_NAMES = [
    "Default Learning Rate",
    "Higher Spatial Encoder Learning Rate",
    "Augmentations",
    "Temporal Dropout",
    "Auxiliary Head",
]
TRAINING_REGULARIZATION = [
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_low_lr_2024-09-17_12-41-36",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_no_aug_2024-09-12_16-46-29",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aug_2024-09-12_16-45-40",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_drop_2024-09-12_16-46-22",
    "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_lower_sen1_2024-09-14_20-01-23"
]

EARLY_FUSION_NAMES = [
    "Early Fusion",
    "Late Fusion: Concatenate",
    "Late Fusion: Concatenate with high S1 dropout",
]
EARLY_FUSION = [
    "/beegfs/work/y0092788/logs/fusion/mt_all_early_2024-10-02_09-16-22",
    "/beegfs/work/y0092788/logs/fusion/mt_all_concat_2024-09-13_08-52-45",
    "/beegfs/work/y0092788/logs/fusion/mt_all_concat_high_s1_drop_2024-10-02_14-58-31",
]
SPATIAL_ENCODER_PARAMETER_NAMES = [
    "ViT-B (S2 Planet)",
    "ViT-L (S2 Planet)",
    "ViT-B (Planet)",
    "ViT-L (Planet)",
]
SPATIAL_ENCODER_PARAMETER = [
    "/beegfs/work/y0092788/logs/modalities/mt_s2_planet_2024-09-15_11-04-12",
    "/beegfs/work/y0092788/logs/mt_s2_planet_vit_l_2024-09-16_08-56-04",
    "/beegfs/work/y0092788/logs/modalities/mt_planet_2024-09-14_10-11-42",
    "/beegfs/work/y0092788/logs/mt_planet_vit_l_2024-09-16_08-56-14",
]
# DATA_FRAC = [
#     "/beegfs/work/y0092788/logs/mt_all_2024-08-01_07-19-13/testing/mt_all/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/mt_all_frac_0_5_2024-08-15_18-10-04/testing/mt_all_frac_0_5/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/mt_all_frac_0_25_2024-08-15_18-17-44/testing/mt_all_frac_0_25/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/mt_all_frac_0_1_2024-08-15_18-09-57/testing/mt_all_frac_0_1/version_0/metrics.csv",
#     "/beegfs/work/y0092788/logs/mt_all_frac_0_1_scratch_2024-08-16_07-55-15/testing/mt_all_frac_0_1_scratch/version_0/metrics.csv"
# ]

# DATA_FRAC_NAMES = [
#     "All",
#     "0.5",
#     "0.25",
#     "0.1",
#     "0.1 scratch",
# ]

# Example usage:
AUX_CLASS_CSV = {
    "main": "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_2024-09-12_16-45-47/testing/mt_all_dofa_aux/version_3/metrics.csv",
    "All - No Auxiliary": "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_drop_2024-09-12_16-46-22/testing/mt_all_dofa_drop/version_1/metrics.csv"  # Assuming you have this file for the "All - No Auxiliary" configuration
}
AUX_CLASS_LOWER_S1_CSV = {
    "main": "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_lower_sen1_2024-09-14_20-01-23/testing/mt_all_dofa_aux_lower_sen1/version_2/metrics.csv",
    "All - No Auxiliary": "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_drop_2024-09-12_16-46-22/testing/mt_all_dofa_drop/version_1/metrics.csv"  # Assuming you have this file for the "All - No Auxiliary" configuration
}

DOWNSCALE_PARAMETER_NAMES = [
    "ViT-B",
    "ViT-S",
    "ViT-Ti",
]
DOWNSCALE_PARAMETER = [
    "/beegfs/work/y0092788/logs/weights/ltae_s1_mt_s2_planet_scratch_2024-09-16_21-44-35",
    "/beegfs/work/y0092788/logs/ltae_s1_mt_s2_planet_concat_vit_s_2024-10-24_10-51-42",
    "/beegfs/work/y0092788/logs/ltae_s1_mt_s2_planet_concat_vit_ti_2024-10-24_13-52-45",
]

CLASSES_VISUAL_PRES_NAMES = [
    "L-TAE S1 & Temporal Max Pooling S2/Planet"
]
CLASSES_VISUAL_PRES = [
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55"
]

# create_miou_bar_plots_temporal_subsets(ST_MT)
# plot_miou_over_epochs_w_test(EARLY_FUSION, EARLY_FUSION_NAMES, "early_fusion")
# plot_miou_over_epochs_w_test(DROPOUT, DROPOUT_NAMES, "dropout")
# plot_miou_over_epochs_w_test(AUG, AUG_NAMES, "aug")
# plot_miou_over_epochs_w_test(AUX, AUX_NAMES, "Aux")
# # # plot_miou_over_epochs(DATA_FRAC, DATA_FRAC_NAMES, "data_frac")
# plot_miou_over_epochs_w_test(LEARNING_RATE, LEARNING_RATE_NAMES, "lr_visual")
# plot_miou_over_epochs_w_test(DECODER, DECODER_NAMES, "decoder")
# # plot_miou_over_epochs_w_test(ARCHITECTURE, ARCHITECTURE_NAMES, "architecture")
# plot_miou_over_epochs_w_test(FUSION, FUSION_NAMES, "fusion")
# plot_miou_over_epochs_w_test(TEMPORAL, TEMPORAL_NAMES, "temporal")
# # plot_miou_over_epochs_w_test(TEMPORAL_S1, TEMPORAL_NAMES_S1, "temporal_s1")
# plot_miou_over_epochs_w_test(WEIGHTS, WEIGHTS_NAMES, "training_paradigms_w_test")
# plot_miou_over_epochs_w_test(WEIGHTS_PRE, WEIGHTS_PRE_NAMES, "training_paradigms_w_test_presi")
# visualize_class_miou_seaborn(CLASSES_VISUAL, CLASSES_VISUAL_NAMES, "class_distribution")
# visualize_class_miou_seaborn(TEMPORAL_S1, TEMPORAL_NAMES_S1, "class_distribution_S1")
# visualize_class_miou_seaborn(TEMPORAL_S2, TEMPORAL_NAMES_S1, "class_distribution_S2")
# visualize_class_miou_seaborn(TEMPORAL_PLANET, TEMPORAL_NAMES_S1, "class_distribution_Planet")
# visualize_class_miou_seaborn(TEMPORAL_ALL, TEMPORAL_NAMES_ALL, "class_distribution_All")
visualize_class_miou_seaborn_pres(CLASSES_VISUAL_PRES, CLASSES_VISUAL_PRES_NAMES, "class_distribution_All_pres")

# collect_best_miou_scores_and_losses(TRAINING_REGULARIZATION, TRAINING_REGULARIZATION_NAMES)
# visualize_class_miou_auxiliary_seaborn(AUX_CLASS_CSV, "aux_classes")
# visualize_class_miou_auxiliary_seaborn(AUX_CLASS_LOWER_S1_CSV, "aux_classes_lower_s1")
# # plot_miou_over_epochs(CLASSES_VISUAL, CLASSES_VISUAL_NAMES, "subset")
# # # visualize_class_miou_seaborn(SINGLE_MODAL, SINGLE_MODAL_NAMES, "single_modal")
# # # visualize_class_miou_seaborn(SUBSETS_MODAL, SUBSETS_MODAL_NAMES, "subsets")
# # plot_miou_over_epochs(WEIGHTS, WEIGHTS_NAMES, "training_paradigms")
# collect_best_miou_scores_and_losses(SPATIAL_ENCODER_PARAMETER, SPATIAL_ENCODER_PARAMETER_NAMES)

# plot_miou_over_epochs_w_test(DOWNSCALE_PARAMETER, DOWNSCALE_PARAMETER_NAMES, "down_scale")
# collect_best_miou_scores_and_losses(DOWNSCALE_PARAMETER, DOWNSCALE_PARAMETER_NAMES)

# AUX_NAME_1 = "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_2024-09-12_16-45-47"
# AUX_NAME_2 = "/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_lower_sen1_2024-09-14_20-01-23"

# plot_aux_loss_from_csv(AUX_NAME_1, "aux_loss_05")
# plot_aux_loss_from_csv(AUX_NAME_2, "aux_loss_025")