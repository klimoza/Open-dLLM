# veomni/models/transformers/qwen2/remasking/training/epoch.py

"""Main training epoch loop for the remasker."""

import os
import random
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from ..config import RemaskerTrainingConfig
from ..logging import log_train_step, log_timestep_eval, evaluate_at_timesteps
from ..metrics import compute_classification_metrics
from ..scheduling import compute_alpha
from .corruption_mode import forward_corruption_mode
from .denoising import (
    create_x_t_and_mask,
    get_x0_predictions,
    apply_x_t_conditioning,
    apply_augmentations,
)
from .loss import compute_remasker_loss


def train_epoch(
    model,
    backbone,
    dataloader,
    optimizer,
    scheduler,
    config: RemaskerTrainingConfig,
    epoch: int,
    global_step: int,
    save_path: str,
    mask_token_id: Optional[int] = None,
    tokenizer = None,
    eval_dataloader = None,
) -> Tuple[float, int]:
    """
    Train for one epoch.
    
    Args:
        model: The remasker model
        backbone: The backbone model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        epoch: Current epoch number
        global_step: Current global step
        save_path: Path to save checkpoints
        mask_token_id: Token id for masking (required for denoising mode)
        tokenizer: Tokenizer (optional, for special tokens)
        eval_dataloader: Evaluation dataloader for timestep evaluation
    
    Returns:
        Tuple of (average_loss, updated_global_step)
    """
    model.train()
    backbone.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Accumulators for metrics over gradient accumulation steps
    accum_metrics = {
        "positive_ratio": 0.0,
        "pred_positive_ratio": 0.0,
        "pred_avg_prob": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "pos_weight": 0.0,
    }
    accum_grad_norm = 0.0
    accum_count = 0
    
    # Get special token ids for corruption (denoising mode)
    special_token_ids = []
    if tokenizer is not None:
        if tokenizer.pad_token_id is not None:
            special_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            special_token_ids.append(tokenizer.eos_token_id)
        if tokenizer.bos_token_id is not None:
            special_token_ids.append(tokenizer.bos_token_id)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        loss_mask = batch["loss_mask"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        prompt_lens = batch["prompt_lens"].to(config.device)
        
        if config.use_denoising_training:
            loss, logits, labels, pos_weight = _forward_denoising_mode(
                model=model,
                backbone=backbone,
                batch=batch,
                config=config,
                mask_token_id=mask_token_id,
                tokenizer=tokenizer,
                special_token_ids=special_token_ids,
            )
        else:
            loss, logits, pos_weight = forward_corruption_mode(
                model=model,
                backbone=backbone,
                input_ids=input_ids,
                labels=labels,
                loss_mask=loss_mask,
                attention_mask=attention_mask,
                config=config,
            )
        
        # Compute classification metrics (no grad needed)
        with torch.no_grad():
            batch_metrics = compute_classification_metrics(logits, labels, loss_mask)
            for k in batch_metrics:
                accum_metrics[k] += batch_metrics[k]
            accum_metrics["pos_weight"] += pos_weight
            accum_count += 1
        
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Clip gradients and get the total norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            accum_grad_norm += grad_norm.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Save checkpoint every N steps
            if config.save_every_n_steps > 0 and global_step % config.save_every_n_steps == 0:
                step_save_path = os.path.join(save_path, f"step_{global_step}")
                model.save_pretrained(step_save_path)
                print(f"\nSaved checkpoint to {step_save_path}")
            
            # Log to wandb with accumulated metrics
            avg_metrics = {k: v / accum_count for k, v in accum_metrics.items()}
            log_train_step(
                loss=loss.item() * config.gradient_accumulation_steps,
                lr=scheduler.get_last_lr()[0],
                grad_norm=accum_grad_norm,
                metrics=avg_metrics,
                global_step=global_step,
                use_wandb=config.use_wandb,
            )
            
            # Timestep evaluation
            if (config.eval_timesteps_every_n_steps > 0 and 
                global_step % config.eval_timesteps_every_n_steps == 0 and
                eval_dataloader is not None):
                model.eval()
                timestep_metrics = evaluate_at_timesteps(
                    model, backbone, eval_dataloader, config, mask_token_id, tokenizer
                )
                model.train()
                log_timestep_eval(timestep_metrics, global_step, config.use_wandb)
            
            # Reset accumulators
            accum_metrics = {
                "positive_ratio": 0.0,
                "pred_positive_ratio": 0.0,
                "pred_avg_prob": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "pos_weight": 0.0,
            }
            accum_grad_norm = 0.0
            accum_count = 0
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
    
    return total_loss / num_batches, global_step


def _forward_denoising_mode(
    model,
    backbone,
    batch,
    config: RemaskerTrainingConfig,
    mask_token_id: int,
    tokenizer,
    special_token_ids,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Forward pass for denoising training mode.
    
    Returns:
        Tuple of (loss, logits, labels, pos_weight)
    """
    input_ids = batch["input_ids"].to(config.device)
    loss_mask = batch["loss_mask"].to(config.device)
    attention_mask = batch["attention_mask"].to(config.device)
    prompt_lens = batch["prompt_lens"].to(config.device)
    ground_truth_ids = batch["ground_truth_ids"].to(config.device)
    batch_size, seq_len = input_ids.shape
    
    # Sample timestep t uniformly from [t_off, t_on]
    t = random.uniform(config.denoising_t_off, config.denoising_t_on)
    
    # Compute alpha (fraction of tokens to keep unmasked)
    alpha = compute_alpha(
        t=t,
        schedule="linear",
        t_on=config.denoising_t_on,
        t_off=config.denoising_t_off,
        alpha_on=0.9,  # Not used for linear schedule
        eps=1e-3
    )
    
    # Create x_t by masking completion tokens
    x_t, mask_positions, fix_mask = create_x_t_and_mask(
        ground_truth_ids=ground_truth_ids,
        attention_mask=attention_mask,
        prompt_lens=prompt_lens,
        alpha=alpha,
        mask_token_id=mask_token_id,
        device=config.device,
    )
    
    # Get hidden states and x_0 predictions from backbone
    with torch.no_grad():
        x_0_full, hidden_states, confidence_full = get_x0_predictions(
            x_t=x_t,
            ground_truth_ids=ground_truth_ids,
            mask_positions=mask_positions,
            fix_mask=fix_mask,
            backbone=backbone,
            attention_mask=attention_mask,
            prompt_lens=prompt_lens,
            mask_token_id=mask_token_id,
            config=config,
        )
        
        # Apply conditioning scheme
        if config.use_x_t_conditioning:
            # Double denoising scheme for x_t conditioning
            x_0_for_remasker, x_t_for_remasker, labels, hidden_states, confidence_full = apply_x_t_conditioning(
                x_0_full=x_0_full,
                ground_truth_ids=ground_truth_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                alpha=alpha,
                mask_token_id=mask_token_id,
                backbone=backbone,
                config=config,
            )
        else:
            # Standard single denoising scheme (no x_t conditioning)
            x_t_for_remasker = None
            
            # Apply augmentations (random/repeat corruption) to completion tokens
            vocab_size = tokenizer.vocab_size if tokenizer else backbone.config.vocab_size
            x_0_for_remasker, labels = apply_augmentations(
                x_0_full=x_0_full,
                ground_truth_ids=ground_truth_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                vocab_size=vocab_size,
                random_corruption_ratio=config.random_corruption_ratio,
                repeat_corruption_ratio=config.repeat_corruption_ratio,
                special_token_ids=special_token_ids,
            )
    
    # Forward pass through remasker with x_0_for_remasker and optional x_t conditioning
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
        # Create timestep tensor for time conditioning (if enabled)
        timestep_tensor = torch.full((batch_size,), t, device=config.device) if config.use_time_conditioning else None
        
        logits = model(
            x_0=x_0_for_remasker,
            hidden_states=hidden_states,
            attention_mask=attention_mask.float(),
            timestep=timestep_tensor,
            confidence=confidence_full,
            x_t=x_t_for_remasker,
        )
        
        # Compute loss
        loss, pos_weight = compute_remasker_loss(
            logits=logits,
            labels=labels,
            loss_mask=loss_mask,
            use_class_reweighting=config.use_class_reweighting,
            label_smoothing_alpha=config.label_smoothing_alpha,
            device=config.device,
        )
        loss = loss / config.gradient_accumulation_steps
    
    return loss, logits, labels, pos_weight
