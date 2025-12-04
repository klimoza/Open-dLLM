#!/bin/bash

MODEL_PATH="fredzzp/open-dcoder-0.5B"
# MODEL_PATH="/home/ubuntu/working_dir/logs/Open_DLLM_SFT_code/checkpoints/global_step_26/hf_ckpt"
MAX_NEW_TOKENS=128
STEPS=128
TEMPERATURE=0.8
ALG="p2"
NUM_PROCESSES=8


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HF_ALLOW_CODE_EVAL=1

accelerate launch --num_processes $NUM_PROCESSES eval.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 10 \
    --output_path evals_results/humaneval-ns0 \
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
