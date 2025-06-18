import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

def extract_fraction(experiment_name):
    match = re.search(r'frac_(\d+_\d+)', experiment_name)
    if match:
        return float(match.group(1).replace('_', '.'))
    return 1.0  # For the full dataset experiment

def read_metrics(experiment_path):
    # Extract the base name from the path
    base_name = os.path.basename(experiment_path)

    # Split the base name by underscores
    parts = base_name.split('_')

    # Join the relevant parts to get the experiment name
    experiment_name = '_'.join(parts[:-2])

    version_0_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_0', 'metrics.csv')
    version_1_path = os.path.join(experiment_path, 'testing', experiment_name, 'version_1', 'metrics.csv')
    
    if os.path.exists(version_0_path):
        df_train_val = pd.read_csv(version_0_path)
        # Find the epoch with the highest val_mIoU
        best_epoch_df_idx = df_train_val['val_mIoU'].idxmax()
        best_epoch = df_train_val.loc[best_epoch_df_idx, 'epoch']
        df_part = df_train_val[df_train_val["epoch"] == best_epoch]
        df_part = df_part.agg("mean")
        train_miou = df_part['train_mIoU']
        val_miou = df_part['val_mIoU']
        # df_train_val = pd.read_csv(version_0_path)
        # train_miou = df_train_val['train_mIoU'].iloc[-1]
        # val_miou = df_train_val['val_mIoU'].iloc[-1]
    else:
        train_miou = val_miou = None
    
    if os.path.exists(version_1_path):
        df_test = pd.read_csv(version_1_path)
        test_miou = df_test['test_mIoU'].iloc[-1]
    else:
        test_miou = None
    
    return train_miou, val_miou, test_miou

def process_experiments(experiment_list):
    data = []
    for exp in experiment_list:
        fraction = extract_fraction(exp)
        is_scratch = 'scratch' in exp
        train_miou, val_miou, test_miou = read_metrics(exp)
        data.append({
            'fraction': fraction,
            'is_scratch': is_scratch,
            'train_miou': train_miou,
            'val_miou': val_miou,
            'test_miou': test_miou
        })
    return pd.DataFrame(data)

def plot_miou_scores(df):
    # Set the style and figure size
    # sns.set_style("whitegrid")
    sns.set_theme(font_scale=1.5)
    # sns.set_context(font_scale=1.75)
    plt.figure(figsize=(12, 8))

    # Create the plot
    for is_scratch, group in df.groupby('is_scratch'):
        linestyle = '--' if is_scratch else '-'
        label_suffix = 'Scratch' if is_scratch else 'Foundation'
        
        sns.lineplot(data=group, x='fraction', y='train_miou', linestyle=linestyle, 
                     marker='o', label=f'Train ({label_suffix})', color='blue')
        sns.lineplot(data=group, x='fraction', y='val_miou', linestyle=linestyle, 
                     marker='s', label=f'Validation ({label_suffix})', color='green')
        sns.lineplot(data=group, x='fraction', y='test_miou', linestyle=linestyle, 
                     marker='^', label=f'Test ({label_suffix})', color='red')

    # Customize the plot
    # plt.xscale('log')
    plt.xlabel('Amount of Data')
    plt.ylabel('mIoU Score')
    plt.title('mIoU Scores vs Amount of Data')
    plt.legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    out_file = os.path.join("visuals", f"data_fraction.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

# List of experiments
experiments = [
    '/beegfs/work/y0092788/logs/data/mt_all_frac_0_01_2024-09-16_09-35-06',
    '/beegfs/work/y0092788/logs/data/mt_all_frac_0_1_2024-09-16_09-34-50',
    '/beegfs/work/y0092788/logs/data/mt_all_frac_0_1_scratch_2024-09-16_09-00-56',
    '/beegfs/work/y0092788/logs/data/mt_all_frac_0_01_scratch_2024-09-16_09-02-00',
    "/beegfs/work/y0092788/logs/data/mt_all_frac_0_05_2024-09-16_09-40-43",
    "/beegfs/work/y0092788/logs/data/mt_all_frac_0_5_2024-09-16_09-41-00",
    "/beegfs/work/y0092788/logs/data/mt_all_frac_0_05_scratch_2024-09-16_09-01-00",
    "/beegfs/work/y0092788/logs/data/mt_all_frac_0_5_scratch_2024-09-16_09-01-26",
    "/beegfs/work/y0092788/logs/data/mt_all_frac_0_025_2024-09-16_09-40-23",
    "/beegfs/work/y0092788/logs/data/mt_all_frac_0_25_2024-09-16_09-40-52",
    "/beegfs/work/y0092788/logs/data/mt_all_frac_0_025_scratch_2024-09-16_09-01-02",
    "/beegfs/work/y0092788/logs/data/mt_all_frac_0_25_scratch_2024-09-16_09-32-35",
    "/beegfs/work/y0092788/logs/temporal/ltae_s1_mt_s2_planet_concat_2024-09-14_09-48-55",
    "/beegfs/work/y0092788/logs/weights/mt_all_scratch_2024-09-12_08-46-31"
]

# Process the experiments and create the plot
df = process_experiments(experiments)
plot_miou_scores(df)
