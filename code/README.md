# Code

EXECUTE ONCE BEFORE TRAINING: Pre-processing Scripts
- (mean_std.py: calc mean + std for all modalities)
- valid_patches.py: filter valid patches and write to a file
- train_test_val_split.py: split dataset into 0.6 - 0.2 - 0.2
- mm_harz.py -> save_patches_to_files: for each modality, split the whole .tif tile into smaller patches for training 

Model Assessment
- SLURM scripts: write functions in bash_rc for faster submissions
    - train.sh: run train_supervision.py to train a model with a configuration. Usage: <config_name> <log_dir>
    - test_qualitative.sh: run test_qualitative.py to create confusion matrices + visualizations for each class. Usage: <exp_dir>
    - test_metrics.sh: run test_metrics.py to run the best checkpoint on the test set. Usage: <exp_dir>
- Scripts
    - train_supervision.py: Train a configuration
    - inference_all_splits.py: Evaluate and visualize model predictions on the whole dataset (train, test, val)


Utils:
- flops_calculator.py: Calculate flops for a configuration