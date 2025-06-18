import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 

# Assuming the data is in a DataFrame called 'df'
df = pd.read_csv('results_simple.csv')

# Set the Seaborn style
# sns.set(font_scale=5)
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.75)

# Create the figure with a single axis
fig, ax = plt.subplots(figsize=(16, 10))

# Define color mapping
color_map = {
    'S1': '#FF6B6B',  # Red
    'S2': '#4ECDC4',  # Teal
    'Planet': '#45B7D1',  # Blue
    'S2 + Planet': '#FFA07A',  # Light Salmon
    'S2 + S1': '#98D8C8',  # Light Sea Green
    'Planet + S1': '#A78BFA',  # Light Purple
    'All': '#FFD700',  # Gold
    'Other': '#808080'  # Gray
}

# Define color mapping
color_map = {
    'S1': '#FF6B6B',  # Red
    'S2': '#4ECDC4',  # Teal
    'Planet': '#45B7D1',  # Blue
    'S2 + Planet': '#FFA07A',  # Light Salmon
    'S2 + S1': '#98D8C8',  # Light Sea Green
    'Planet + S1': '#A78BFA',  # Light Purple
    'All': '#FFD700',  # Gold
    'Other': '#808080'  # Gray
}

# Function to determine category based on model name
def get_category(model_name):
    if 'S1' in model_name and 'S2' in model_name and 'Planet' in model_name:
        return 'All'
    elif 'S1' in model_name and 'S2' in model_name:
        return 'S2 + S1'
    elif 'S1' in model_name and 'Planet' in model_name:
        return 'Planet + S1'
    elif 'S2' in model_name and 'Planet' in model_name:
        return 'S2 + Planet'
    elif 'S1' in model_name:
        return 'S1'
    elif 'S2' in model_name:
        return 'S2'
    elif 'Planet' in model_name:
        return 'Planet'
    else:
        return 'All'

# Assign categories to each data point
df['Modality'] = df['Model'].apply(get_category)

# Create scatter plot
scatter = sns.scatterplot(
    data=df,
    x='Total FLOPs',
    y='Test mIoU',
    hue='Modality',
    palette=color_map,
    s=300,  # Fixed size for all points
    alpha=0.7,
    ax=ax
)

# Set labels and title
ax.set_xlabel('Total FLOPs')
ax.set_ylabel('Test mIoU')
ax.set_title('FLOPs vs mIoU for Different Models and Modalities')

# Set x-axis to log scale
ax.set_xscale('log')

# Define a subset of important models to label
# important_models = ['L-TAE/Max-Pool All', 'DOFA', 'ST S2', 'ST S1', 'MT Planet ViT-L', 'ST Planet', 'MT S2 Planet', "MT S2", "MT S1", "MT Planet", "ST All", "MT S1 S2", "L-TAE S1", "L-TAE S2", "Early Fusion"]
important_models = ['L-TAE/Max-Pool All', "L-TAE Planet", "Early Fusion", "ST Planet", "MT Planet", 'DOFA', 'ST S2', 'ST S1', 'MT S2 Planet', "MT S2", "MT S1", "ST All", "L-TAE S1", "L-TAE S2", "MT All", "ViT-Ti (Scratch)", "ViT-S (Scratch)"]

# Adding text labels for the subset of important models
for i, model in enumerate(df['Model']):
    if model in important_models:
        if model in "ST Planet" or model == "MT Planet ViT-L" or model in "MT All" or model in "ViT-S (Scratch)":
            ax.annotate(model, (df['Total FLOPs'][i], df['Test mIoU'][i]), 
                        xytext=(7, -15), textcoords='offset points', fontsize=16)
        elif model in "Early Fusion":
            ax.annotate(model, (df['Total FLOPs'][i], df['Test mIoU'][i]), 
                        xytext=(-100, 10), textcoords='offset points', fontsize=16)
        elif model in "MT Planet":
            ax.annotate(model, (df['Total FLOPs'][i], df['Test mIoU'][i]), 
                        xytext=(-90, -15), textcoords='offset points', fontsize=16)
        else:
            ax.annotate(model, (df['Total FLOPs'][i], df['Test mIoU'][i]), 
                        xytext=(7, 10), textcoords='offset points', fontsize=16)

# Adjust legend
plt.legend(title='Modalities', loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout to make room for the legend
plt.tight_layout()
out_file = os.path.join("visuals", f"flops_pres.png")
plt.savefig(out_file, bbox_inches='tight', dpi=300)
plt.close()