#!/bin/bash
#SBATCH --job-name=ryley_training
#SBATCH --partition=pi_abodner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=35:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

if [ $# -lt 3 ]; then
    echo "Error: Missing arguments"
    echo "Usage: sbatch --array=0-N run_job.sh <config_file> <subdirectory> <samples_csv>"
    exit 1
fi

CONFIG_FILE=$1
SUBDIR=$2
IFS=',' read -ra TRAIN_SAMPLES <<< "$3"

TRAIN_SAMPLE=${TRAIN_SAMPLES[$SLURM_ARRAY_TASK_ID]}

# Create temporary config file
TEMP_CONFIG="configs/temp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"
sed -i "s/train_samples: .*/train_samples: $TRAIN_SAMPLE/" "$TEMP_CONFIG"

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Training samples: $TRAIN_SAMPLE"
echo "Config file: $TEMP_CONFIG"
echo "Subdirectory: $SUBDIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

uv run src/train.py "$TEMP_CONFIG" --device cuda:0 --subdirectory "$SUBDIR"

rm "$TEMP_CONFIG"

echo "Job finished on $(date)"