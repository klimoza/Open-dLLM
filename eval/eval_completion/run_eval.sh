#!/bin/bash

MODEL_PATHS=(
    # "fredzzp/open-dcoder-0.5B"
    "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_alpha0.1_temp1.0/checkpoints/global_step_1000/hf_ckpt"
    "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_alpha0.1_temp1.0/checkpoints/global_step_2000/hf_ckpt"
    "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_alpha0.1_temp1.0/checkpoints/global_step_3000/hf_ckpt"
    "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_alpha0.1_temp1.0/checkpoints/global_step_4000/hf_ckpt"
    "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_alpha0.1_temp1.0/checkpoints/global_step_5000/hf_ckpt"
    "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_alpha0.1_temp1.0/checkpoints/global_step_6000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_alpha0.1_temp1.0/checkpoints/global_step_3000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_alpha0.1_temp1.0/checkpoints/global_step_6000/hf_ckpt"

    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_mask_to_random_ratio0.1/checkpoints/global_step_1000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_mask_to_random_ratio0.1/checkpoints/global_step_2000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_mask_to_random_ratio0.1/checkpoints/global_step_3000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_mask_to_random_ratio0.1/checkpoints/global_step_4000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_mask_to_random_ratio0.1/checkpoints/global_step_5000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_denoising_mask_to_random_ratio0.1/checkpoints/global_step_8000/hf_ckpt"

    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_mask_to_random_0.1_lr5e-5/checkpoints/global_step_1000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_mask_to_random_0.1_lr5e-5/checkpoints/global_step_2000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_mask_to_random_0.1_lr5e-5/checkpoints/global_step_3000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_mask_to_random_0.1_lr5e-5/checkpoints/global_step_4000/hf_ckpt"
    # "/home/ubuntu/Open-dLLM/logs/fredzzp/open_dcoder_0.5B_ft_mask_to_random_0.1_lr5e-5/checkpoints/global_step_5000/hf_ckpt"
)
MAX_NEW_TOKENS=128
STEPS=128
TEMPERATURE=0.8
ALG="p2"
NUM_PROCESSES=8


export CUDA_VISIBLE_DEVICES="0"
export HF_ALLOW_CODE_EVAL=1

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    # Extract model name from path for output directory
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    for SEED in 400 401 402 403 404 405 406 407 408 409; do
        echo "Running evaluation for model $MODEL_NAME with seed $SEED"
        accelerate launch --num_processes $NUM_PROCESSES eval.py \
            --model custom_coder \
            --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
            --tasks humaneval \
            --num_fewshot 0 \
            --batch_size 32 \
            --output_path evals_results/humaneval-p2-improved \
            --log_samples \
            --seed $SEED \
            --confirm_run_unsafe_code
    done
done

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
