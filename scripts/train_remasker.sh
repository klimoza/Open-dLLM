LR=1e-5
BS=8
GRAD_ACC=32
RANDOM_CORRUPTION_RATIO=0.00
REPEAT_CORRUPTION_RATIO=0.00
LABEL_SMOOTHING_ALPHA=0.00
DENOISING_T_ON=0.95
DENOISING_T_OFF=0.05
DENOISING_TEMP=0.0
DENOISING_TEMP_STEPS=0.0
DENOISING_NUM_STEPS=1
LAYERS=12
EFFECTIVE_BS=$((BS * GRAD_ACC))
run_name="remasker-training-open-dcoder-0.5B-layers${LAYERS}-lr${LR}-eff_bs${EFFECTIVE_BS}-init_from_backbone-denoising-t${DENOISING_T_ON}-t${DENOISING_T_OFF}-no_hidden-time_cond-confidence_cond"

CUDA_VISIBLE_DEVICES=0 python scripts/train_remasker.py \
    --backbone_path fredzzp/open-dcoder-0.5B \
    --dataset_path nvidia/OpenCodeInstruct \
    --checkpoint_name $run_name \
    --num_layers $LAYERS \
    --epochs 3 \
    --lr $LR \
    --batch_size $BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --wandb_project remasker-training \
    --wandb_run_name $run_name \
    --use_wandb \
    --no_hidden_states \
    --random_corruption_ratio $RANDOM_CORRUPTION_RATIO \
    --repeat_corruption_ratio $REPEAT_CORRUPTION_RATIO \
    --label_smoothing_alpha $LABEL_SMOOTHING_ALPHA \
    --no_fp16 \
    --max_grad_norm 1.0 \
    --use_denoising_training \
    --denoising_t_on $DENOISING_T_ON \
    --denoising_t_off $DENOISING_T_OFF \
    --denoising_temperature $DENOISING_TEMP \
    --init_from_backbone \
    --init_layer_offset 0 \
    --denoising_num_steps $DENOISING_NUM_STEPS \
    --denoising_temperature $DENOISING_TEMP_STEPS \
    --use_time_conditioning \
    --use_confidence_conditioning