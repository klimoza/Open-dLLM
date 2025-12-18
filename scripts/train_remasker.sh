LR=1e-5
BS=8
GRAD_ACC=32
RANDOM_CORRUPTION_RATIO=0.05
REPEAT_CORRUPTION_RATIO=0.05
LABEL_SMOOTHING_ALPHA=0.05
run_name="remasker-training-open-dcoder-0.5B-lr${LR}-bs${BS}-ga${GRAD_ACC}-rand${RANDOM_CORRUPTION_RATIO}-rep${REPEAT_CORRUPTION_RATIO}-ls${LABEL_SMOOTHING_ALPHA}-init_random"

 CUDA_VISIBLE_DEVICES=2 python scripts/train_remasker.py \
    --backbone_path fredzzp/open-dcoder-0.5B \
    --dataset_path nvidia/OpenCodeInstruct \
    --checkpoint_name $run_name \
    --num_layers 3 \
    --epochs 3 \
    --lr $LR \
    --batch_size $BS \
    --gradient_accumulation_steps $GRAD_ACC \
    --wandb_project remasker-training \
    --wandb_run_name $run_name \
    --use_wandb \
    --random_corruption_ratio $RANDOM_CORRUPTION_RATIO \
    --repeat_corruption_ratio $REPEAT_CORRUPTION_RATIO \
    --label_smoothing_alpha $LABEL_SMOOTHING_ALPHA \
    --no_fp16 \
    --max_grad_norm 1.0 \
   #  --init_from_backbone \
   #  --init_layer_offset 0