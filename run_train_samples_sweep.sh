#!/bin/bash

# Check if config file argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <base_config_file> [cuda_device]"
    echo "Example: $0 config.yaml cuda:0"
    exit 1
fi

BASE_CONFIG=$1
CUDA_DEVICE=${2:-cuda:1}  # Default to cuda:1 if not provided

# Check if the config file exists
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Config file '$BASE_CONFIG' not found"
    exit 1
fi

# Extract subdirectory name from config file (remove path and .yaml extension)
SUBDIRECTORY=$(basename "$BASE_CONFIG" .yaml)

# Check if subdirectory exists and is not empty
FULL_PATH="outputs/$SUBDIRECTORY"
if [ -d "$FULL_PATH" ] && [ "$(ls -A $FULL_PATH)" ]; then
    echo "Warning: Directory '$FULL_PATH' is not empty."
    read -p "Do you want to remove all files and reset the sweep? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing contents of $FULL_PATH..."
        rm -rf "$FULL_PATH"/*
    else
        echo "Continuing without removing files..."
    fi
fi

samples=(1 10 20 50 100 500 1000 2000)

# Array to store PIDs
pids=()

# Trap SIGINT (Ctrl+C) and kill all child processes
trap 'echo "Caught Ctrl+C, killing all jobs..."; kill ${pids[@]} 2>/dev/null; rm -f configs/temp_config_*.yaml; exit 1' INT

echo "Using CUDA device: $CUDA_DEVICE"

i=0
for sample in "${samples[@]}"
do
    sed "s/train_samples: [0-9]\+/train_samples: ${sample}/" "$BASE_CONFIG" > configs/temp_config_${i}.yaml
    
    python3 src/train.py configs/temp_config_${i}.yaml --device "$CUDA_DEVICE" --subdirectory "$SUBDIRECTORY" &
    pids+=($!)  # Store the PID of the last background process
    ((i++))
    
    sleep 0.5
done

# Wait for each specific PID
for pid in ${pids[@]}; do
    wait $pid
done

rm -f configs/temp_config_*.yaml