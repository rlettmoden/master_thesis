import csv
import pandas as pd
from tabulate import tabulate

def read_metrics(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)[-1]  # Get the last row

def create_table(metrics, modalities = ['planet', 's1', 's2', '']):
    classes = ['Built-up', 'Conifer', 'Crop', 'Dead tree', 'Deciduous', 'Grass', 'Water']
    
    data = []
    
    for modality in modalities:
        row = [modality if modality else 'All']
        
        for class_name in classes:
            key = f'IoU_{modality}_test_{class_name}' if modality else f'IoU_test_{class_name}'
            value = metrics.get(key, 'N/A')
            row.append(value)
        
        miou_key = f'test_{modality}_mIoU' if modality else 'test_mIoU'
        oa_key = f'test_{modality}_OA' if modality else 'test_OA'
        f1_key = f'test_{modality}_F1' if modality else 'test_F1'
        
        row.extend([
            metrics.get(miou_key, 'N/A'),
            metrics.get(oa_key, 'N/A'),
            metrics.get(f1_key, 'N/A')
        ])
        
        data.append(row)
    
    headers = ['Modality'] + classes + ['mIoU', 'OA', 'F1']
    return tabulate(data, headers=headers, tablefmt='pipe', floatfmt='.4f')

# Read the CSV file
metrics = read_metrics('/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_2024-09-12_16-45-47/testing/mt_all_dofa_aux/version_3/metrics.csv')

# Create and print the table
table = create_table(metrics)
print(table)

# Read the CSV file
metrics = read_metrics('/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_aux_lower_sen1_2024-09-14_20-01-23/testing/mt_all_dofa_aux_lower_sen1/version_2/metrics.csv')

# Create and print the table
table = create_table(metrics)
print(table)


# Create and print the table
metrics = read_metrics("/beegfs/work/y0092788/logs/training_regularization/mt_all_dofa_drop_2024-09-12_16-46-22/testing/mt_all_dofa_drop/version_1/metrics.csv")
table = create_table(metrics, modalities=[""])
print(table)

# Read the CSV file
metrics = read_metrics('/beegfs/work/y0092788/logs/mt_all_dofa_aux_0125_sen1_2024-10-02_07-08-18/testing/mt_all_dofa_aux_0125_sen1/version_1/metrics.csv')

# Create and print the table
table = create_table(metrics)
print(table)

# Read the CSV file
metrics = read_metrics('/beegfs/work/y0092788/logs/mt_all_dofa_aux_0375_sen1_2024-09-22_10-00-52/testing/mt_all_dofa_aux_0375_sen1/version_1/metrics.csv')

# Create and print the table
table = create_table(metrics)
print(table)