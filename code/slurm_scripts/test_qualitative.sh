#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=55GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive

echo "TIMESTAMP: $(date +"%m-%d-%Y_%H-%M")"
export CUDA_HOME=/cluster/cuda/12.2.1
export LD_LIBRARY_PATH=/cluster/cuda/12.2.1/lib64
export PATH=/usr/local/cuda/12.2.1/bin${PATH:+:${PATH}}
module load cuda/12.2.1

source ~/anaconda3/bin/activate
conda activate dofa

# # Check if all required arguments are provided
# if [ $# -ne 1 ]; then
#     echo "Error: Missing arguments."
#     echo "Usage: \$0 <out_dir>"
#     exit 1
# fi

OUT_DIR="$1"

echo "Creating directory ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

TEST_SCRIPT="/beegfs/work/y0092788/thesis/GeoSeg-main/test_qualitative.py"

# Copy the config file to the output directory
# cp "${CONFIG_PATH}" "${OUT_DIR}/"

python3 ${TEST_SCRIPT} ${OUT_DIR}
echo "TIMESTAMP: $(date +"%m-%d-%Y_%H-%M")"
