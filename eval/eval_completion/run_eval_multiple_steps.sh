#!/bin/bash

# Specify the global_step numbers you want to evaluate
GLOBAL_STEPS=(156 312 468 624 780 936 1092 1248 1404 1560)

# Base path for the model checkpoints
BASE_PATH="/home/ubuntu/working_dir/logs/Open_DLLM_SFT_code/checkpoints"

# Path to the original run_eval.sh script
EVAL_SCRIPT="$(dirname "$0")/run_eval.sh"

# Loop through each global_step
for step in "${GLOBAL_STEPS[@]}"; do
    echo "========================================="
    echo "Running evaluation for global_step: $step"
    echo "========================================="
    
    MODEL_PATH="${BASE_PATH}/global_step_${step}/hf_ckpt"
    
    # Create a temporary script with the modified MODEL_PATH
    TEMP_SCRIPT=$(mktemp)
    sed "s|MODEL_PATH=.*|MODEL_PATH=\"$MODEL_PATH\"|" "$EVAL_SCRIPT" > "$TEMP_SCRIPT"
    chmod +x "$TEMP_SCRIPT"
    
    # Run the modified script
    bash "$TEMP_SCRIPT"
    
    # Clean up
    rm "$TEMP_SCRIPT"
    
    echo "Completed evaluation for global_step: $step"
    echo ""
done

echo "All evaluations completed!"

