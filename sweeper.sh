#!/bin/bash

echo "
Here's a sweep for you:
      .-.
      | |
      |=|
      |=|
      | |
      | |
      | |
      | |
      | |
      | |
      | |
      | |
      | |
      | |
      | |
      | |
      | |
      |=|
      |_|
    .=/I\=.
   ////V\\\\
   |#######|
   |||||||||
   |||||||||
   |||||||||
   |||||||||
   |||||||||
"

# ===== EDIT THIS LIST =====
TRAIN_SAMPLES=(1 3 8 16 33 166 500) #(1 10 25 50 100 500 1500)
# ==========================

ARRAY_MAX=$((${#TRAIN_SAMPLES[@]} - 1))
SAMPLES_STR=$(IFS=,; echo "${TRAIN_SAMPLES[*]}")

echo "=== Sweep settings ==="
echo "TRAIN_SAMPLES: ${SAMPLES_STR}"
echo "Array range: 0-${ARRAY_MAX}"
echo ""

echo "=== Config differences ==="
echo ""
diff --side-by-side --suppress-common-lines \
    configs/SR_middle_cnn_3box_aug.yaml \
    configs/SR_middle_cnn_3box_noaug.yaml && echo "(middle: aug vs noaug)"
echo ""
diff --side-by-side --suppress-common-lines \
    configs/SR_nearwall_cnn_3box_aug.yaml \
    configs/SR_nearwall_cnn_3box_noaug.yaml && echo "(nearwall: aug vs noaug)"
echo ""
diff --side-by-side --suppress-common-lines \
    configs/SR_middle_cnn_3box_aug.yaml \
    configs/SR_nearwall_cnn_3box_aug.yaml && echo "(aug: middle vs nearwall)"
echo ""

read -p "Enter sweep subdirectory name: " SUBDIR

if [ -z "$SUBDIR" ]; then
    echo "Error: Subdirectory name cannot be empty"
    exit 1
fi

sbatch --array=0-${ARRAY_MAX} sbatch_train_samples_sweep.sh configs/SR_middle_cnn_3box_noaug.yaml "$SUBDIR" "$SAMPLES_STR"
sbatch --array=0-${ARRAY_MAX} sbatch_train_samples_sweep.sh configs/SR_middle_cnn_3box_aug.yaml "$SUBDIR" "$SAMPLES_STR"
sbatch --array=0-${ARRAY_MAX} sbatch_train_samples_sweep.sh configs/SR_nearwall_cnn_3box_noaug.yaml "$SUBDIR" "$SAMPLES_STR"
sbatch --array=0-${ARRAY_MAX} sbatch_train_samples_sweep.sh configs/SR_nearwall_cnn_3box_aug.yaml "$SUBDIR" "$SAMPLES_STR"
