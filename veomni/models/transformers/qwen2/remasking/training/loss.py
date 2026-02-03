# veomni/models/transformers/qwen2/remasking/training/loss.py

"""Loss computation helpers for remasker training."""

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_remasker_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    use_class_reweighting: bool,
    label_smoothing_alpha: float,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Compute BCE loss with optional class reweighting and label smoothing.
    
    Args:
        logits: Model output logits [B, L]
        labels: Ground truth labels [B, L]
        loss_mask: Mask indicating which positions to include in loss [B, L]
        use_class_reweighting: Whether to apply class reweighting
        label_smoothing_alpha: Label smoothing factor (0 = no smoothing)
        device: Device for tensor operations
    
    Returns:
        loss: Computed loss tensor
        pos_weight: The positive class weight used (for logging)
    """
    # Get masked logits and labels
    masked_logits = logits[loss_mask]
    masked_labels = labels[loss_mask]
    
    # Apply label smoothing if enabled: 0 -> alpha, 1 -> 1-alpha
    if label_smoothing_alpha > 0:
        masked_labels = masked_labels * (1 - 2 * label_smoothing_alpha) + label_smoothing_alpha
    
    # Compute class weights if enabled
    if use_class_reweighting and masked_labels.numel() > 0:
        # Count positive (correct) and negative (corrupted) samples
        num_positive = masked_labels.sum()
        num_negative = masked_labels.numel() - num_positive
        
        # pos_weight: weight for positive class to balance with negative class
        # If positive is majority, pos_weight < 1 to down-weight positives
        # This is equivalent to up-weighting negatives
        if num_positive > 0 and num_negative > 0:
            pos_weight = num_negative / num_positive
        else:
            pos_weight = torch.tensor(1.0, device=device)
        
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
    
    return loss, batch_pos_weight
