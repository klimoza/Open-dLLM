#!/bin/bash

MODEL_PATH="fredzzp/open-dcoder-0.5B"
MAX_NEW_TOKENS=128
STEPS=128
TEMPERATURE=0.8
ALG="entropy"
NUM_PROCESSES=8


export CUDA_VISIBLE_DEVICES="0"
export HF_ALLOW_CODE_EVAL=1
# REMASKER_PATH="/home/ubuntu/Open-dLLM/checkpoints/remasker-training-open-dcoder-0.5B-layers12-lr1e-5-bs8-ga32-rand0.05-rep0.05-ls0.00-init_from_backbone-denoising-t0.2-t0.1-temp0.0-no_hidden_states/step_5000"
# REMASKER_PATH="/home/ubuntu/Open-dLLM/checkpoints/remasker-training-open-dcoder-0.5B-layers12-lr1e-5-bs8-ga32-rand0.05-rep0.05-ls0.00-init_from_backbone-denoising-t0.2-t0.1-temp0.0/step_45000"
# REMASKER_PATH="/home/ubuntu/Open-dLLM/checkpoints/remasker-training-open-dcoder-0.5B-layers12-lr1e-5-bs8-ga32-rand0.05-rep0.05-ls0.00-init_from_backbone-denoising-t0.2-t0.1-temp0.0-no_hidden_states/step_20000"
# REMASKER_PATH="/home/ubuntu/Open-dLLM/checkpoints/remasker-training-open-dcoder-0.5B-layers12-lr1e-5-bs8-ga32-rand0.00-rep0.00-ls0.00-init_from_backbone-denoising-t0.1-t0.05-temp0.0-no_hidden_states-several_steps4_temp0.0/step_5000"
REMASKER_PATH="/home/ubuntu/Open-dLLM/checkpoints/remasker-training-open-dcoder-0.5B-layers12-lr1e-5-bs8-ga32-rand0.00-rep0.00-ls0.00-init_from_backbone-denoising-t0.95-t0.05-temp0.0-no_hidden_states-several_steps1_temp0.0/step_5000"
# REMASKER_PATH="/home/ubuntu/Open-dLLM/checkpoints/remasker-training-open-dcoder-0.5B-layers12-lr1e-5-bs8-ga32-rand0.00-rep0.00-ls0.00-init_from_backbone-denoising-t0.95-t0.05-temp0.0-no_hidden_states-several_steps1_temp0.0-time_conditioning-confidence_conditioning/step_5000"
ALG_REMASKING="remasking"
REMASKING_LOGITS_SOURCE="model"

# === Hyperparameter lists (will run all combinations) ===
STEPS_LIST=(128)
TEMPERATURE_LIST=(0.0)
REMASKING_SCHEDULE_LIST=("linear")
REMASKING_T_ON_LIST=(0.5)
REMASKING_T_OFF_LIST=(0.05)
REMASKING_ALPHA_ON_LIST=(0.9)
REMASKING_TEMPERATURE_LIST=(0.0)
NON_REMASKING_SAMPLING_ALG_LIST=("entropy")
SEED_LIST=(404 405 406 407 408 409 410)

# Loop over all hyperparameter combinations
for STEPS in "${STEPS_LIST[@]}"; do
for NON_REMASKING_SAMPLING_ALG in "${NON_REMASKING_SAMPLING_ALG_LIST[@]}"; do
for TEMPERATURE in "${TEMPERATURE_LIST[@]}"; do
for REMASKING_SCHEDULE in "${REMASKING_SCHEDULE_LIST[@]}"; do
for REMASKING_T_ON in "${REMASKING_T_ON_LIST[@]}"; do
for REMASKING_T_OFF in "${REMASKING_T_OFF_LIST[@]}"; do
for REMASKING_TEMPERATURE in "${REMASKING_TEMPERATURE_LIST[@]}"; do
for SEED in "${SEED_LIST[@]}"; do

    # For linear schedule, use single alpha; for loop, iterate through all
    if [ "$REMASKING_SCHEDULE" == "linear" ]; then
        ALPHA_LIST=(0.9)
    else
        ALPHA_LIST=("${REMASKING_ALPHA_ON_LIST[@]}")
    fi

    for REMASKING_ALPHA_ON in "${ALPHA_LIST[@]}"; do

    echo "=============================================="
    echo "Running with:"
    echo "  STEPS=$STEPS"
    echo "  TEMPERATURE=$TEMPERATURE"
    echo "  REMASKING_SCHEDULE=$REMASKING_SCHEDULE"
    echo "  REMASKING_T_ON=$REMASKING_T_ON"
    echo "  REMASKING_T_OFF=$REMASKING_T_OFF"
    echo "  REMASKING_ALPHA_ON=$REMASKING_ALPHA_ON"
    echo "  REMASKING_TEMPERATURE=$REMASKING_TEMPERATURE"
    echo "  NON_REMASKING_SAMPLING_ALG=$NON_REMASKING_SAMPLING_ALG"
    echo "  SEED=$SEED"
    echo "=============================================="

    accelerate launch --num_processes $NUM_PROCESSES eval.py \
        --model custom_coder \
        --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG_REMASKING,remasking_schedule=$REMASKING_SCHEDULE,remasking_t_on=$REMASKING_T_ON,remasking_t_off=$REMASKING_T_OFF,remasking_alpha_on=$REMASKING_ALPHA_ON,remasking_logits_source=$REMASKING_LOGITS_SOURCE,remasker_checkpoint_path=$REMASKER_PATH,non_remasking_sampling_algorithm=$NON_REMASKING_SAMPLING_ALG,remasking_temperature=$REMASKING_TEMPERATURE" \
        --tasks humaneval \
        --num_fewshot 0 \
        --batch_size 32 \
        --output_path evals_results/remasking_full_traj \
        --log_samples \
        --seed $SEED \
        --confirm_run_unsafe_code

    done

done
done
done
done
done
done
done
done
