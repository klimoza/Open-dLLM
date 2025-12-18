#!/bin/bash

MODEL_PATH="fredzzp/open-dcoder-0.5B"
# MODEL_PATH="/home/ubuntu/working_dir/logs/Open_DLLM_SFT_code/checkpoints/global_step_26/hf_ckpt"
MAX_NEW_TOKENS=128
STEPS=128
TEMPERATURE=0.8
ALG="entropy"
NUM_PROCESSES=8


export CUDA_VISIBLE_DEVICES="7"
export HF_ALLOW_CODE_EVAL=1

# accelerate launch --num_processes $NUM_PROCESSES eval.py \
#     --model custom_coder \
#     --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
#     --tasks humaneval \
#     --num_fewshot 0 \
#     --batch_size 64 \
#     --output_path evals_results/humaneval-ns0 \
#     --log_samples \
#     --seed 428 \
#     --confirm_run_unsafe_code

# === Remasking evaluation example ===
# Uncomment and configure the following to evaluate with remasking:

REMASKER_PATH="/home/shibaev/Open-dLLM/checkpoints/remasker-training-open-dcoder-0.5B-lr-1e-5-bs8-grad-acc32-random-0.05-repeat-0.05-label-smoothing-0.05/step_12000"
# remasker-training-open-dcoder-0.5B-lr-1e-5-bs8-grad-acc32-random-0.05-repeat-0.05-label-smoothing-0.05
ALG_REMASKING="remasking"
REMASKING_SCHEDULE="linear"
REMASKING_T_ON=0.55
REMASKING_T_OFF=0.2
REMASKING_ALPHA_ON=0.9
REMASKING_LOGITS_SOURCE="model"
REMASKING_TEMPERATURE=0.0
NON_REMASKING_SAMPLING_ALG="entropy"

accelerate launch --num_processes $NUM_PROCESSES eval.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG_REMASKING,remasking_schedule=$REMASKING_SCHEDULE,remasking_t_on=$REMASKING_T_ON,remasking_t_off=$REMASKING_T_OFF,remasking_alpha_on=$REMASKING_ALPHA_ON,remasking_logits_source=$REMASKING_LOGITS_SOURCE,remasker_checkpoint_path=$REMASKER_PATH,non_remasking_sampling_algorithm=$NON_REMASKING_SAMPLING_ALG,remasking_temperature=$REMASKING_TEMPERATURE" \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 64 \
    --output_path evals_results/humaneval-remasking \
    --log_samples \
    --seed 428 \
    --confirm_run_unsafe_code

# accelerate launch --num_processes $NUM_PROCESSES eval.py \
#     --model custom_coder \
#     --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
#     --tasks humaneval_plus \
#     --num_fewshot 0 \
#     --batch_size 10 \
#     --output_path evals_results/humaneval_plus-ns0 \
#     --log_samples \
#     --confirm_run_unsafe_code


# accelerate launch --num_processes $NUM_PROCESSES eval.py \
#     --model custom_coder \
#     --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
#     --tasks mbpp \
#     --num_fewshot 0 \
#     --batch_size 10 \
#     --output_path evals_results/mbpp-ns0 \
#     --log_samples \
#     --confirm_run_unsafe_code

# accelerate launch --num_processes $NUM_PROCESSES eval.py \
#     --model custom_coder \
#     --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
#     --tasks mbpp_plus \
#     --num_fewshot 0 \
#     --batch_size 10 \
#     --output_path evals_results/mbpp_plus-ns0 \
#     --log_samples \
#     --confirm_run_unsafe_code
