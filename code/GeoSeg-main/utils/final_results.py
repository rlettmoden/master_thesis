import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    'Model': [
        'DOFA', 'Prithvi Single', 'Prithvi Partial Double', 'Prithvi Double', 
        'SegFormer', 'Mid Fusion', 'Concatenate Fusion', 'UBARN', 'MT S1 Planet', 
        'MT S2 Planet', 'MT S1 S2', 'MT Planet', 'MT S1', 'MT S2', 'ST S1', 
        'ST S2', 'ST Planet', 'ST All', 'L-TAE S1', 'L-TAE S2', 'L-TAE Planet', 
        'L-TAE All', 'L-TAE/Max-Pool All', 'MT Planet ViT-L', 'MT S2 Planet ViT-L'
    ],
    'Test mIoU': [
        0.8159, 0.8202, 0.8202, 0.8200, 0.8075, 0.8189, 0.8213, 0.8227, 0.8166, 
        0.8169, 0.7897, 0.8112, 0.6113, 0.7861, 0.4996, 0.7382, 0.7822, 0.5526, 
        0.6451, 0.7788, 0.8109, 0.8233, 0.8237, 0.8124, 0.8179
    ],
    'Total FLOPs': [
        7.50e11, 3.57e11, 3.94e11, 5.03e11, 3.77e11, 3.56e11, 3.90e11, 4.27e11, 
        3.91e11, 2.21e11, 2.13e11, 1.85e11, 1.91e11, 2.14e10, 8.25e9, 8.56e9, 
        7.28e10, 7.28e10, 2.12e11, 2.35e10, 2.05e11, 4.69e11, 4.48e11, 5.83e11, 
        6.69e11
    ]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 8))
sns.barplot(x='Test mIoU', y='Model', data=df, palette='coolwarm', edgecolor=".6")

# Adding FLOPs as text
for index, row in df.iterrows():
    plt.text(row['Test mIoU'] - 0.02, index, f"{row['Total FLOPs']:.1e}", color='black', va="center", ha="right")

plt.title('Model Comparison')
plt.xlabel('Test mIoU')
plt.ylabel('Model')
plt.xlim(0.48, 0.83)

out_file = os.path.join("visuals", f"results.png")
plt.savefig(out_file, bbox_inches='tight', dpi=300)
plt.close()