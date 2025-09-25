#!/bin/bash
set -e  # stop on error

# Install dependencies
pip install -r requirements.txt

# Download dataset
echo "Installing dataset"
curl -L -o numpy_middle.zip "https://zenodo.org/record/16997999/files/numpy_middle.zip?download=1"
unzip -o numpy_middle.zip -d dataset 
curl -L -o numpy_nearwall.zip "https://zenodo.org/record/16997999/files/numpy_nearwall.zip?download=1"
unzip -o numpy_nearwall.zip -d dataset


# Run training
echo "Running training script on midplane single box"
python3 src/train.py configs/midplane_single_timestep_1_box.yaml

echo "Running training script on boundary single box"
python3 src/train.py configs/boundary_single_timestep_1_box.yaml

