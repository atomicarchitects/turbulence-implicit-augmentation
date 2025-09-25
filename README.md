# Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution

## Requirements
To install requirements:
```
pip install -r requirements.txt
```
## To get the dataset:
```
curl -L -o numpy_middle.zip "https://zenodo.org/record/16997999/files/numpy_middle.zip?download=1"
unzip -o numpy_middle.zip -d dataset 
curl -L -o numpy_nearwall.zip "https://zenodo.org/record/16997999/files/numpy_nearwall.zip?download=1"
unzip -o numpy_near_wall.zip -d dataset
```
## To train the model..

For $T=1$ training samples with 1 box at the boundary:
```
python3 src/train.py configs/boundary_single_timestep_1_box.yaml
```
For $T=1$ training samples with 1 box at the boundary with explicit rotational augmentation:
```
python3 src/train.py configs/boundary_single_timestep_1_box_augmented.yaml
```
For $T=1$ training samples with 3 boxes at the boundary:
```
python3 src/train.py configs/boundary_single_timestep_3_boxes.yaml
```
For $T=1$ training samples with 1 box at the midplane:
```
python3 src/train.py configs/midplane_single_timestep_1_box.yaml
```

## End-to-End examle pipeline:
```
./main.sh
```

