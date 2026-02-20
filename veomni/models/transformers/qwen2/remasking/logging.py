# veomni/models/transformers/qwen2/remasking/logging.py

"""Logging utilities for remasker training."""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import RemaskerTrainingConfig
from .corruption import sample_tokens_from_logits
from .scheduling import compute_alpha


def log_train_step(
    loss: float,
    lr: float,
    grad_norm: float,
    metrics: Dict[str, float],
    global_step: int,
    use_wandb: bool = True,
) -> None:
    """Log training step metrics to wandb."""
    if not use_wandb or not WANDB_AVAILABLE:
        return
    wandb.log({
        "train/loss": loss,
        "train/lr": lr,
        "train/grad_norm": grad_norm,
        "train/positive_ratio": metrics["positive_ratio"],
        "train/pred_positive_ratio": metrics["pred_positive_ratio"],
        "train/pred_avg_prob": metrics["pred_avg_prob"],
        "train/precision": metrics["precision"],
        "train/recall": metrics["recall"],
        "train/pos_weight": metrics["pos_weight"],
        "global_step": global_step,
    })


def log_timestep_eval(
    timestep_metrics: Dict[float, Dict[str, float]],
    global_step: int,
    use_wandb: bool = True,
) -> None:
    """Log timestep evaluation metrics to wandb."""
    if not use_wandb or not WANDB_AVAILABLE:
        return
    for t, metrics in timestep_metrics.items():
        wandb.log({
            f"metrics@t={t:.1f}/loss": metrics["loss"],
            f"metrics@t={t:.1f}/precision": metrics["precision"],
            f"metrics@t={t:.1f}/recall": metrics["recall"],
            f"metrics@t={t:.1f}/positive_ratio": metrics["positive_ratio"],
            "global_step": global_step,
        })


@torch.no_grad()
def evaluate_at_timesteps(
    model,
    backbone,
    eval_dataloader,
    config: RemaskerTrainingConfig,
    mask_token_id: int,
    tokenizer=None,
    timesteps: Optional[List[float]] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate the remasker model at fixed timesteps.
    
    For each timestep t, computes precision, recall, positive_ratio, and loss
    by accumulating metrics across multiple batches from the eval dataloader.
    
    Args:
        model: The remasker model
        backbone: The backbone language model
        eval_dataloader: DataLoader for evaluation data
        config: Training configuration
        mask_token_id: Token ID used for masking
        tokenizer: Tokenizer (optional, for vocab_size)
        timesteps: List of timesteps to evaluate at (default: [0.1, 0.2, ..., 0.9])
    
    Returns:
        Dictionary mapping timestep -> {"loss", "precision", "recall", "positive_ratio"}
    """
    if timesteps is None:
        timesteps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    num_batches = config.eval_timesteps_num_samples // config.batch_size
    
    # Collect batches first (so we can reuse for each timestep)
    batches = []
    for i, batch in enumerate(eval_dataloader):
        if i >= num_batches:
            break
        batches.append({k: v.to(config.device) for k, v in batch.items()})
    
    if len(batches) == 0:
        return {t: {"loss": 0.0, "precision": 0.0, "recall": 0.0, "positive_ratio": 0.0} for t in timesteps}
    
    results = {}
    
    for t in timesteps:
        # Initialize accumulators
        total_loss = 0.0
        total_tp, total_fp, total_fn = 0, 0, 0
        total_positive, total_tokens = 0, 0
        
        # Compute alpha for this timestep
        alpha = compute_alpha(
            t=t,
            schedule="linear",
            t_on=config.denoising_t_on,
            t_off=config.denoising_t_off,
            alpha_on=0.9,
            eps=1e-3
        )
        
        for batch in batches:
            ground_truth_ids = batch["ground_truth_ids"]
            attention_mask = batch["attention_mask"]
            prompt_lens = batch["prompt_lens"]
            loss_mask = batch["loss_mask"]
            
            batch_size, seq_len = ground_truth_ids.shape
            
            # Create x_t by masking completion tokens
            x_t = ground_truth_ids.clone()
            mask_positions = torch.zeros_like(x_t, dtype=torch.bool)
            
            for b in range(batch_size):
                prompt_len = prompt_lens[b].item()
                completion_len = int(attention_mask[b].sum().item()) - prompt_len
                if completion_len <= 0:
                    continue
                
                # Number of completion tokens to mask
                num_to_mask = int(completion_len * (1 - alpha))
                if num_to_mask > 0:
                    perm = torch.randperm(completion_len, device=config.device)
                    mask_indices = perm[:num_to_mask]
                    x_t[b, prompt_len + mask_indices] = mask_token_id
                    mask_positions[b, prompt_len + mask_indices] = True
            
            # Create fix_mask for prompt positions
            fix_mask = torch.zeros_like(x_t, dtype=torch.bool)
            for b in range(batch_size):
                prompt_len = prompt_lens[b].item()
                fix_mask[b, :prompt_len] = True
            
            # Get hidden states from backbone if needed
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                if config.use_hidden_states:
                    backbone_outputs = backbone(
                        input_ids=x_t,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        is_causal=False,
                    )
                    hidden_states = backbone_outputs.hidden_states[-1]
                else:
                    hidden_states = None
                
                # Get x_0 predictions via single-step denoising
                backbone_outputs_for_logits = backbone(
                    input_ids=x_t,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                    is_causal=False,
                )
                backbone_logits = backbone_outputs_for_logits.logits
                
                # Shift logits to predict the next token
                backbone_logits = torch.cat([backbone_logits[:, :1], backbone_logits[:, :-1]], dim=1)
                
                # Sample x_0 predictions from logits
                x_0_pred = sample_tokens_from_logits(
                    backbone_logits,
                    temperature=config.denoising_temperature
                )
                
                # Build x_0_full: predictions for masked positions, ground truth for unmasked
                x_0_full = ground_truth_ids.clone()
                x_0_full[mask_positions] = x_0_pred[mask_positions]
                
                # Compute confidence if needed
                if config.use_confidence_conditioning:
                    probs = torch.softmax(backbone_logits.float(), dim=-1)
                    confidence_full = torch.gather(probs, -1, x_0_full.unsqueeze(-1)).squeeze(-1)
                    for b in range(batch_size):
                        prompt_len = prompt_lens[b].item()
                        confidence_full[b, :prompt_len] = 1.0
                else:
                    confidence_full = None
                
                # Compute labels: 1 if matches ground truth
                labels = (x_0_full == ground_truth_ids).float()
                for b in range(batch_size):
                    prompt_len = prompt_lens[b].item()
                    labels[b, :prompt_len] = 1.0
                
                # Create timestep tensor if needed
                timestep_tensor = torch.full((batch_size,), t, device=config.device) if config.use_time_conditioning else None
                
                # Forward pass through remasker
                logits = model(
                    x_0=x_0_full,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask.float(),
                    timestep=timestep_tensor,
                    confidence=confidence_full,
                    x_t=x_t if config.use_x_t_conditioning else None,
                )
                
                # Compute loss
                masked_logits = logits[loss_mask]
                masked_labels = labels[loss_mask]
                
                if masked_logits.numel() > 0:
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        reduction="mean",
                    )
                    total_loss += loss.item()
                
                # Compute TP, FP, FN for precision/recall
                preds = (torch.sigmoid(logits) > 0.5).float()
                masked_preds = preds[loss_mask]
                
                tp = ((masked_preds == 1) & (masked_labels == 1)).sum().item()
                fp = ((masked_preds == 1) & (masked_labels == 0)).sum().item()
                fn = ((masked_preds == 0) & (masked_labels == 1)).sum().item()
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_positive += masked_labels.sum().item()
                total_tokens += masked_labels.numel()
        
        # Compute final metrics from accumulated counts
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        positive_ratio = total_positive / total_tokens if total_tokens > 0 else 0.0
        avg_loss = total_loss / len(batches) if len(batches) > 0 else 0.0
        
        results[t] = {
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "positive_ratio": positive_ratio,
        }
    
    return results
