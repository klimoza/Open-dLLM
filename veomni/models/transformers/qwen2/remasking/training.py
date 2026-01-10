# veomni/models/transformers/qwen2/remasking/training.py

"""Training and evaluation loops for the remasker."""

import os
import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import RemaskerTrainingConfig
from .corruption import corrupt_completion, sample_tokens_from_logits, multi_step_denoise
from .metrics import compute_classification_metrics
from .scheduling import compute_alpha


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
) -> Tuple[float, int]:
    """Train for one epoch."""
    model.train()
    backbone.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Accumulators for metrics over gradient accumulation steps
    accum_metrics = {"positive_ratio": 0.0, "pred_positive_ratio": 0.0, "pred_avg_prob": 0.0, "precision": 0.0, "recall": 0.0, "pos_weight": 0.0}
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
            # Denoising training mode: simulate inference process
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
            
            # Create x_t by masking completion tokens for each sample in batch
            x_t = ground_truth_ids.clone()
            mask_positions = torch.zeros_like(x_t, dtype=torch.bool)
            
            for b in range(batch_size):
                prompt_len = prompt_lens[b].item()
                completion_len = (attention_mask[b].sum().item()) - prompt_len
                if completion_len <= 0:
                    continue
                
                # Number of completion tokens to mask
                num_to_mask = int(completion_len * (1 - alpha))
                if num_to_mask > 0:
                    # Randomly select which positions to mask in completion
                    perm = torch.randperm(completion_len, device=config.device)
                    mask_indices = perm[:num_to_mask]
                    
                    # Apply masking
                    x_t[b, prompt_len + mask_indices] = mask_token_id
                    mask_positions[b, prompt_len + mask_indices] = True
            
            # Create fix_mask: True for prompt positions (never modify these)
            fix_mask = torch.zeros_like(x_t, dtype=torch.bool)
            for b in range(batch_size):
                prompt_len = prompt_lens[b].item()
                fix_mask[b, :prompt_len] = True
            
            # Get hidden states from backbone on x_t (for remasker conditioning)
            # Also get x_0 predictions via single-step or multi-step denoising
            with torch.no_grad():
                # First, get hidden states from x_t if needed for remasker
                if config.use_hidden_states:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                        backbone_outputs = backbone(
                            input_ids=x_t,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True,
                            is_causal=False,  # Bidirectional attention for MDM
                        )
                        hidden_states = backbone_outputs.hidden_states[-1]
                else:
                    hidden_states = None
                
                # Get x_0 predictions
                if config.denoising_num_steps > 1:
                    # Multi-step entropy-based denoising
                    x_0_pred = multi_step_denoise(
                        x_t=x_t,
                        backbone=backbone,
                        attention_mask=attention_mask,
                        mask_token_id=mask_token_id,
                        num_steps=config.denoising_num_steps,
                        temperature=config.denoising_temperature,
                        fix_mask=fix_mask,
                        fp16=config.fp16,
                    )
                    # Build x_0_full: for multi-step, x_0_pred already has all predictions
                    # But we want to ensure prompt positions have ground truth
                    x_0_full = x_0_pred.clone()
                    x_0_full[fix_mask] = ground_truth_ids[fix_mask]
                else:
                    # Single-step sampling (original behavior)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                        backbone_outputs_for_logits = backbone(
                            input_ids=x_t,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            return_dict=True,
                            is_causal=False,
                        )
                        backbone_logits = backbone_outputs_for_logits.logits
                        
                        # CRITICAL: Shift logits to predict the next token (matching inference)
                        backbone_logits = torch.cat([backbone_logits[:, :1], backbone_logits[:, :-1]], dim=1)
                        
                        # Sample x_0 predictions from logits
                        x_0_pred = sample_tokens_from_logits(
                            backbone_logits, 
                            temperature=config.denoising_temperature
                        )
                        
                        # Build x_0_full: use predictions for masked positions, ground truth for unmasked
                        x_0_full = ground_truth_ids.clone()
                        x_0_full[mask_positions] = x_0_pred[mask_positions]
                
                # Apply augmentations (random/repeat corruption) to completion tokens
                # We need to do this per-sample since completion lengths vary
                augmentation_mask = torch.zeros_like(x_0_full, dtype=torch.bool)
                
                for b in range(batch_size):
                    prompt_len = prompt_lens[b].item()
                    actual_len = attention_mask[b].sum().item()
                    completion_len = actual_len - prompt_len
                    if completion_len <= 0:
                        continue
                    
                    completion_slice = x_0_full[b, prompt_len:actual_len]
                    corrupted_completion, corruption_mask = corrupt_completion(
                        completion_slice,
                        vocab_size=tokenizer.vocab_size if tokenizer else backbone.config.vocab_size,
                        random_ratio=config.random_corruption_ratio,
                        repeat_ratio=config.repeat_corruption_ratio,
                        special_token_ids=special_token_ids,
                    )
                    x_0_full[b, prompt_len:actual_len] = corrupted_completion
                    augmentation_mask[b, prompt_len:actual_len] = corruption_mask
                
                # Compute labels: 1 if matches ground truth AND not corrupted by augmentation
                # For completion positions: correct if x_0_full == ground_truth AND not augmented
                prediction_correct = (x_0_full == ground_truth_ids)
                not_augmented = ~augmentation_mask
                labels = (prediction_correct & not_augmented).float()
                
                # Prompt tokens are always labeled as correct (not used in loss anyway)
                for b in range(batch_size):
                    prompt_len = prompt_lens[b].item()
                    labels[b, :prompt_len] = 1.0
            
            # Forward pass through remasker with x_0_full and hidden_states from x_t
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                logits = model(
                    x_0=x_0_full,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask.float(),
                )
                
                # Get masked logits and labels
                masked_logits = logits[loss_mask]
                masked_labels = labels[loss_mask]
                
                # Apply label smoothing if enabled
                if config.label_smoothing_alpha > 0:
                    masked_labels = masked_labels * (1 - 2 * config.label_smoothing_alpha) + config.label_smoothing_alpha
                
                # Compute class weights if enabled
                if config.use_class_reweighting and masked_labels.numel() > 0:
                    num_positive = masked_labels.sum()
                    num_negative = masked_labels.numel() - num_positive
                    
                    if num_positive > 0 and num_negative > 0:
                        pos_weight = num_negative / num_positive
                    else:
                        pos_weight = torch.tensor(1.0, device=config.device)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        pos_weight=pos_weight,
                        reduction="mean",
                    )
                    batch_pos_weight = pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        reduction="mean",
                    )
                    batch_pos_weight = 1.0
                loss = loss / config.gradient_accumulation_steps
        
        else:
            # Original corruption-based training mode
            # Get hidden states from backbone (no gradient) if needed
            if config.use_hidden_states:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                        backbone_outputs = backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        hidden_states = backbone_outputs.hidden_states[-1]  # Final layer
            else:
                hidden_states = None
            
            # Forward pass through remasker
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                logits = model(
                    x_0=input_ids,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask.float(),
                )
                
                # Get masked logits and labels
                masked_logits = logits[loss_mask]
                masked_labels = labels[loss_mask]
                
                # Apply label smoothing if enabled: 0 -> alpha, 1 -> 1-alpha
                if config.label_smoothing_alpha > 0:
                    masked_labels = masked_labels * (1 - 2 * config.label_smoothing_alpha) + config.label_smoothing_alpha
                
                # Compute class weights if enabled
                if config.use_class_reweighting and masked_labels.numel() > 0:
                    # Count positive (correct) and negative (corrupted) samples
                    num_positive = masked_labels.sum()
                    num_negative = masked_labels.numel() - num_positive
                    
                    # pos_weight: weight for positive class to balance with negative class
                    # If positive is majority, pos_weight < 1 to down-weight positives
                    # This is equivalent to up-weighting negatives
                    if num_positive > 0 and num_negative > 0:
                        pos_weight = num_negative / num_positive
                    else:
                        pos_weight = torch.tensor(1.0, device=config.device)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        pos_weight=pos_weight,
                        reduction="mean",
                    )
                    batch_pos_weight = pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight
                else:
                    # No reweighting
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        reduction="mean",
                    )
                    batch_pos_weight = 1.0
                loss = loss / config.gradient_accumulation_steps
        
        # Compute classification metrics (no grad needed)
        with torch.no_grad():
            batch_metrics = compute_classification_metrics(logits, labels, loss_mask)
            for k in batch_metrics:
                accum_metrics[k] += batch_metrics[k]
            accum_metrics["pos_weight"] += batch_pos_weight
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
            if config.use_wandb and WANDB_AVAILABLE:
                avg_metrics = {k: v / accum_count for k, v in accum_metrics.items()}
                wandb.log({
                    "train/loss": loss.item() * config.gradient_accumulation_steps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": accum_grad_norm,
                    "train/positive_ratio": avg_metrics["positive_ratio"],
                    "train/pred_positive_ratio": avg_metrics["pred_positive_ratio"],
                    "train/pred_avg_prob": avg_metrics["pred_avg_prob"],
                    "train/precision": avg_metrics["precision"],
                    "train/recall": avg_metrics["recall"],
                    "train/pos_weight": avg_metrics["pos_weight"],
                    "global_step": global_step,
                })
            
            # Reset accumulators
            accum_metrics = {"positive_ratio": 0.0, "pred_positive_ratio": 0.0, "pred_avg_prob": 0.0, "precision": 0.0, "recall": 0.0, "pos_weight": 0.0}
            accum_grad_norm = 0.0
            accum_count = 0
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
    
    return total_loss / num_batches, global_step


@torch.no_grad()
def evaluate(
    model,
    backbone,
    dataloader,
    config: RemaskerTrainingConfig,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    backbone.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        loss_mask = batch["loss_mask"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        
        # Get hidden states from backbone if needed
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
            if config.use_hidden_states:
                backbone_outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = backbone_outputs.hidden_states[-1]
            else:
                hidden_states = None
            
            # Forward pass through remasker
            logits = model(
                x_0=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask.float(),
            )
            
            # BCE loss
            loss = F.binary_cross_entropy_with_logits(
                logits[loss_mask],
                labels[loss_mask],
                reduction="mean",
            )
        
        total_loss += loss.item()
        
        # Accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds[loss_mask] == labels[loss_mask]).sum().item()
        total_correct += correct
        total_samples += loss_mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return {"eval_loss": avg_loss, "eval_accuracy": accuracy}

