#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=58GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive

echo "TIMESTAMP: $(date +"%m-%d-%Y_%H-%M")"
export CUDA_HOME=/cluster/cuda/12.2.1
export LD_LIBRARY_PATH=/cluster/cuda/12.2.1/lib64
export PATH=/usr/local/cuda/12.2.1/bin${PATH:+:${PATH}}
module load cuda/12.2.1


source ~/anaconda3/bin/activate
conda activate dofa

# Check if a configuration name and output directory were passed as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <config_name> <log_dir>"
    exit 1
else
    CONFIG="$1"
    OUT_DIR="$2"
fi

echo "Creating directory ${OUT_DIR}"
mkdir -p "${OUT_DIR}"


TRAIN_SCRIPT="/beegfs/work/y0092788/thesis/GeoSeg-main/train_supervision.py"

config_file=/beegfs/work/y0092788/thesis/GeoSeg-main/config/${CONFIG}.py
cp "$config_file" "$OUT_DIR/"

python3 ${TRAIN_SCRIPT} -c ${config_file} -o ${OUT_DIR}
echo "TIMESTAMP: $(date +"%m-%d-%Y_%H-%M")"