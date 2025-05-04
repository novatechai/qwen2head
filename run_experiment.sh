#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "-----------------------------------"
echo "Starting Dual-Head Qwen Experiment"
echo "-----------------------------------"

# --- Step 1: Data Preparation ---
echo "$(date): Starting Data Preparation..."
python prepare_data.py
echo "$(date): Data Preparation Finished."
echo "-----------------------------------"

# --- Step 2: Model Training ---
echo "$(date): Starting Model Training..."
python train_dual_head.py
echo "$(date): Model Training Finished."
echo "-----------------------------------"

# --- Optional Step 3: Inference (Example) ---
# Uncomment the following lines to run inference after training
# Make sure the --model_path points to the correct output directory from training_config
# trained_model_dir=$(python -c "import config; print(config.training_config['output_dir'])")
# if [ -d "$trained_model_dir" ]; then
#     echo "$(date): Starting Inference Example..."
#     python inference.py --model_path "$trained_model_dir" --prompt "This is a test prompt."
#     echo "$(date): Inference Example Finished."
#     echo "-----------------------------------"
# else
#      echo "$(date): Skipping Inference - Trained model directory not found: $trained_model_dir"
#      echo "-----------------------------------"
# fi

echo "$(date): Experiment Script Completed." 