#!/usr/bin/env python3
# scripts/train_remasker.py

"""
Training script for the Remasker model.

The remasker learns to identify corrupted tokens in a sequence.
It takes x_0 (predicted tokens) and hidden_states from the backbone,
and outputs binary logits indicating token correctness.

Example usage:
    python scripts/train_remasker.py \
        --backbone_path fredzzp/open-dcoder-0.5B \
        --dataset_path nvidia/OpenCodeInstruct \
        --checkpoint_name remasker_v1 \
        --num_layers 4 \
        --epochs 3 \
        --lr 1e-4 \
        --wandb_project remasker-training \
        --wandb_run_name remasker-training-open-dcoder-0.5B \
        --use_wandb

Dataset format expected:
    - HuggingFace dataset with 'instruction' and 'response' columns (OpenCodeInstruct)
    - Or 'prompt' and 'completion' columns
    - Or 'messages' column with list of dicts
"""

import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from veomni.models.transformers.qwen2.remasking import (
    RemaskerConfig,
    RemaskerTrainingConfig,
    Remasker,
    RemaskerDataset,
    collate_fn,
    load_data,
    train_epoch,
    evaluate,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config: RemaskerTrainingConfig):
    """Main training function."""
    set_seed(config.seed)
    
    # Create checkpoint directory (fail if already exists to prevent overwriting)
    save_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)
    if os.path.exists(save_path):
        raise FileExistsError(
            f"Checkpoint directory already exists: {save_path}\n"
            f"Please use a different --checkpoint_name or remove the existing directory."
        )
    os.makedirs(save_path)
    
    # Save config
    config_path = os.path.join(save_path, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)
    
    print(f"Loading backbone from {config.backbone_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_path)
    backbone = AutoModelForCausalLM.from_pretrained(
        config.backbone_path,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
    ).to(config.device)
    backbone.eval()
    
    # Get mask token id for denoising training
    mask_token_id = getattr(backbone.config, 'mask_token_id', None)
    if config.use_denoising_training and mask_token_id is None:
        # Try to get from tokenizer or use a default
        mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        if mask_token_id is None:
            # Use a common convention: vocab_size (out of vocabulary token)
            mask_token_id = backbone.config.vocab_size
            print(f"Warning: No mask_token_id found, using {mask_token_id}")
        else:
            print(f"Using tokenizer mask_token_id: {mask_token_id}")
    elif config.use_denoising_training:
        print(f"Using backbone mask_token_id: {mask_token_id}")
    
    # Get backbone config for remasker
    backbone_config = backbone.config
    
    # Create remasker config
    remasker_config = RemaskerConfig(
        num_layers=config.remasker_num_layers,
        hidden_size=config.remasker_hidden_size or backbone_config.hidden_size,
        intermediate_size=config.remasker_intermediate_size or backbone_config.intermediate_size,
        num_attention_heads=config.remasker_num_attention_heads or backbone_config.num_attention_heads,
        num_key_value_heads=config.remasker_num_key_value_heads or backbone_config.num_key_value_heads,
        vocab_size=backbone_config.vocab_size,
        backbone_hidden_size=backbone_config.hidden_size,
        use_hidden_states=config.use_hidden_states,
        use_time_conditioning=config.use_time_conditioning,
        use_confidence_conditioning=config.use_confidence_conditioning,
    )
    
    print(f"Creating remasker with {config.remasker_num_layers} layers (use_hidden_states={config.use_hidden_states}, use_time_conditioning={config.use_time_conditioning}, use_confidence_conditioning={config.use_confidence_conditioning})...")
    
    if config.init_from_backbone:
        # Calculate layer offset (default: use last N layers from backbone)
        if config.init_layer_offset < 0:
            backbone_num_layers = backbone_config.num_hidden_layers
            layer_offset = max(0, backbone_num_layers - config.remasker_num_layers)
        else:
            layer_offset = config.init_layer_offset
        
        print(f"Initializing remasker from backbone (layer_offset={layer_offset})...")
        model = Remasker.from_backbone(
            backbone_model=backbone,
            config=remasker_config,
            init_embedding=True,
            init_layers=True,
            layer_offset=layer_offset,
        ).to(config.device)
    else:
        model = Remasker(remasker_config).to(config.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Remasker parameters: {num_params:,} total, {num_trainable:,} trainable")
    
    # Load data
    train_data, eval_data = load_data(config)
    
    # Create datasets
    train_dataset = RemaskerDataset(train_data, tokenizer, backbone, config, is_eval=False, mask_token_id=mask_token_id)
    eval_dataset = RemaskerDataset(eval_data, tokenizer, backbone, config, is_eval=True, mask_token_id=mask_token_id)
    
    # Create dataloaders
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    
    if config.scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )
    else:
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps - warmup_steps,
        )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )
    
    # Initialize wandb
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or config.checkpoint_name,
            config=vars(config),
        )
    
    # Training loop
    global_step = 0
    best_eval_loss = float("inf")
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Save every: {config.save_every_n_steps} steps")
    print(f"Class reweighting: {'enabled' if config.use_class_reweighting else 'disabled'}")
    if config.label_smoothing_alpha > 0:
        print(f"Label smoothing: alpha={config.label_smoothing_alpha} (0->{config.label_smoothing_alpha:.3f}, 1->{1-config.label_smoothing_alpha:.3f})")
    if config.use_denoising_training:
        print(f"Denoising training: t_on={config.denoising_t_on}, t_off={config.denoising_t_off}, "
              f"temperature={config.denoising_temperature}, num_steps={config.denoising_num_steps}")
    if config.use_time_conditioning:
        print("Time conditioning: enabled (remasker will receive noise level as input)")
    if config.use_confidence_conditioning:
        print("Confidence conditioning: enabled (remasker will receive backbone confidence as input)")
    
    for epoch in range(config.epochs):
        # Train
        train_loss, global_step = train_epoch(
            model, backbone, train_loader, optimizer, scheduler, config, epoch, global_step, save_path,
            mask_token_id=mask_token_id, tokenizer=tokenizer
        )
        print(f"\nEpoch {epoch + 1} - Train loss: {train_loss:.4f}")
        
        # Evaluate
        eval_metrics = evaluate(model, backbone, eval_loader, config)
        print(f"Epoch {epoch + 1} - Eval loss: {eval_metrics['eval_loss']:.4f}, "
              f"Eval accuracy: {eval_metrics['eval_accuracy']:.4f}")
        
        # Log to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                **{f"eval/{k}": v for k, v in eval_metrics.items()},
            })
        
        # Save best model
        if eval_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval_loss"]
            best_save_path = os.path.join(save_path, "best")
            model.save_pretrained(best_save_path)
            print(f"Saved best model to {best_save_path}")
    
    # Save final model
    final_save_path = os.path.join(save_path, "final")
    model.save_pretrained(final_save_path)
    print(f"\nTraining complete! Final model saved to {final_save_path}")
    
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Remasker model")
    
    # Model paths
    parser.add_argument("--backbone_path", type=str, default="./models/qwen2-0.5b")
    parser.add_argument("--checkpoint_name", type=str, default="remasker_v1")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    # Remasker architecture
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--init_from_backbone", action="store_true", help="Initialize remasker layers from backbone model")
    parser.add_argument("--init_layer_offset", type=int, default=-1, help="Which backbone layer to start copying from (-1 = auto: use last N layers)")
    parser.add_argument("--no_hidden_states", action="store_true", help="Don't condition remasker on backbone hidden states (use only token embeddings)")
    
    # Corruption settings
    parser.add_argument("--random_corruption_ratio", type=float, default=0.1)
    parser.add_argument("--repeat_corruption_ratio", type=float, default=0.1)
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, default="nvidia/OpenCodeInstruct")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "linear"])
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="remasker-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Class reweighting and label smoothing
    parser.add_argument("--no_class_reweighting", action="store_true", help="Disable class reweighting for imbalanced classes")
    parser.add_argument("--label_smoothing_alpha", type=float, default=0.0, help="Label smoothing: 0->alpha, 1->1-alpha (default: 0.0, no smoothing)")
    
    # Denoising training mode
    parser.add_argument("--use_denoising_training", action="store_true", help="Use denoising-based training that matches inference")
    parser.add_argument("--denoising_t_on", type=float, default=0.1, help="Upper bound for timestep sampling in denoising mode")
    parser.add_argument("--denoising_t_off", type=float, default=0.1, help="Lower bound for timestep sampling in denoising mode")
    parser.add_argument("--denoising_temperature", type=float, default=0.0, help="Temperature for sampling x_0 from logits (0 = greedy)")
    parser.add_argument("--denoising_num_steps", type=int, default=4, help="Number of denoising steps (1 = single-step, >1 = multi-step entropy-based)")
    
    # Time conditioning
    parser.add_argument("--use_time_conditioning", action="store_true", help="Enable time conditioning in the remasker model (condition on noise level)")
    
    # Confidence conditioning
    parser.add_argument("--use_confidence_conditioning", action="store_true", help="Enable confidence conditioning (condition on backbone prediction confidence)")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every_n_steps", type=int, default=1000)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--no_fp16", action="store_true")
    
    args = parser.parse_args()
    
    config = RemaskerTrainingConfig(
        backbone_path=args.backbone_path,
        checkpoint_name=args.checkpoint_name,
        checkpoint_dir=args.checkpoint_dir,
        remasker_num_layers=args.num_layers,
        remasker_hidden_size=args.hidden_size,
        remasker_intermediate_size=args.intermediate_size,
        remasker_num_attention_heads=args.num_attention_heads,
        remasker_num_key_value_heads=args.num_key_value_heads,
        init_from_backbone=args.init_from_backbone,
        init_layer_offset=args.init_layer_offset,
        use_hidden_states=not args.no_hidden_states,
        random_corruption_ratio=args.random_corruption_ratio,
        repeat_corruption_ratio=args.repeat_corruption_ratio,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        scheduler_type=args.scheduler_type,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        num_workers=args.num_workers,
        use_class_reweighting=not args.no_class_reweighting,
        label_smoothing_alpha=args.label_smoothing_alpha,
        save_every_n_steps=args.save_every_n_steps,
        eval_ratio=args.eval_ratio,
        fp16=not args.no_fp16,
        use_denoising_training=args.use_denoising_training,
        denoising_t_on=args.denoising_t_on,
        denoising_t_off=args.denoising_t_off,
        denoising_temperature=args.denoising_temperature,
        denoising_num_steps=args.denoising_num_steps,
        use_time_conditioning=args.use_time_conditioning,
        use_confidence_conditioning=args.use_confidence_conditioning,
    )
    
    main(config)
