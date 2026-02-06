# veomni/models/transformers/qwen2/remasking/training/loss.py

"""Loss computation helpers for remasker training."""

from typing import Tuple

import torch
import torch.nn.functional as F


def compute_ranknet_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute RankNet pairwise ranking loss within each sequence.
    
    The loss encourages positive samples (correct tokens) to have higher scores
    than negative samples (corrupted tokens). Computed independently for each
    sequence and then averaged.
    
    L = (1/BS) * Σ_b [ (1 / |P_b| * |H_b|) * Σ_{i∈P_b} Σ_{j∈H_b} log(1 + exp(s_j - s_i)) ]
    
    Where for each sequence b:
        P_b = set of positive samples (correct tokens, label=1)
        H_b = set of negative samples (corrupted tokens, label=0)
        s_i = logit score for positive sample i
        s_j = logit score for negative sample j
    
    Args:
        logits: Model output logits [BS, N]
        labels: Binary labels [BS, N] (1=correct, 0=corrupted)
        loss_mask: Mask indicating which positions to include [BS, N]
        device: Device for tensor operations
    
    Returns:
        RankNet pairwise loss averaged across sequences
    """
    batch_size = logits.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    valid_seqs = 0
    
    for b in range(batch_size):
        # Get masked positions for this sequence
        seq_mask = loss_mask[b]
        seq_logits = logits[b][seq_mask]
        seq_labels = labels[b][seq_mask]
        
        if seq_logits.numel() == 0:
            continue
        
        # Get indices of positive and negative samples
        # For binary labels, threshold at 0.5 to handle label smoothing
        pos_mask = seq_labels > 0.5
        neg_mask = seq_labels <= 0.5
        
        pos_logits = seq_logits[pos_mask]  # [|P_b|]
        neg_logits = seq_logits[neg_mask]  # [|H_b|]
        
        num_pos = pos_logits.numel()
        num_neg = neg_logits.numel()
        
        # Need at least one of each class for pairwise comparison
        if num_pos == 0 or num_neg == 0:
            continue
        
        # Compute pairwise differences: s_j - s_i for all pairs
        # pos_logits: [|P_b|] -> [|P_b|, 1]
        # neg_logits: [|H_b|] -> [1, |H_b|]
        # diff: [|P_b|, |H_b|] where diff[i,j] = neg_logits[j] - pos_logits[i]
        diff = neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)  # [|P_b|, |H_b|]
        
        # RankNet loss: log(1 + exp(s_j - s_i))
        # Use softplus for numerical stability: log(1 + exp(x)) = softplus(x)
        pairwise_loss = F.softplus(diff)  # [|P_b|, |H_b|]
        
        # Average over all pairs in this sequence
        seq_loss = pairwise_loss.sum() / (num_pos * num_neg)
        total_loss = total_loss + seq_loss
        valid_seqs += 1
    
    # Average across sequences
    if valid_seqs > 0:
        return total_loss / valid_seqs
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_remasker_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    use_class_reweighting: bool,
    label_smoothing_alpha: float,
    device: torch.device,
    pos_class_weight: float = 1.0,
    use_ranknet_pairwise_loss: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Compute loss for remasker training.
    
    Supports two loss types:
    - BCE loss (default): Binary cross-entropy with optional class reweighting and label smoothing
    - RankNet loss: Pairwise ranking loss that encourages correct tokens to rank higher
    
    Args:
        logits: Model output logits [B, L]
        labels: Ground truth labels [B, L]
        loss_mask: Mask indicating which positions to include in loss [B, L]
        use_class_reweighting: Whether to apply class reweighting based on batch statistics (BCE only)
        label_smoothing_alpha: Label smoothing factor (0 = no smoothing)
        device: Device for tensor operations
        pos_class_weight: Manual weight multiplier for positive class (BCE only, default 1.0).
            - 1.0 = equal importance of positive and negative classes
            - <1.0 = penalize false positives more (e.g., 0.5 means FP is 2x worse than FN)
            - >1.0 = penalize false negatives more (e.g., 2.0 means FN is 2x worse than FP)
        use_ranknet_pairwise_loss: If True, use RankNet pairwise ranking loss instead of BCE.
    
    Returns:
        loss: Computed loss tensor
        pos_weight: The positive class weight used (for logging, 1.0 for RankNet)
    """
    # Use RankNet pairwise loss if enabled
    if use_ranknet_pairwise_loss:
        # Apply label smoothing to labels for RankNet
        labels_for_ranknet = labels
        if label_smoothing_alpha > 0:
            labels_for_ranknet = labels * (1 - 2 * label_smoothing_alpha) + label_smoothing_alpha
        
        loss = compute_ranknet_pairwise_loss(
            logits=logits,
            labels=labels_for_ranknet,
            loss_mask=loss_mask,
            device=device,
        )
        return loss, 1.0  # pos_weight not applicable for RankNet
    
    # Otherwise use BCE loss
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
            pos_weight = (num_negative / num_positive) * pos_class_weight
        else:
            pos_weight = torch.tensor(pos_class_weight, device=device)
        
        loss = F.binary_cross_entropy_with_logits(
            masked_logits,
            masked_labels,
            pos_weight=pos_weight,
            reduction="mean",
        )
        batch_pos_weight = pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight
    elif pos_class_weight != 1.0 and masked_labels.numel() > 0:
        # No auto-balancing, but manual pos_class_weight is set
        pos_weight = torch.tensor(pos_class_weight, device=device)
        loss = F.binary_cross_entropy_with_logits(
            masked_logits,
            masked_labels,
            pos_weight=pos_weight,
            reduction="mean",
        )
        batch_pos_weight = pos_class_weight
    else:
        # No reweighting at all
        loss = F.binary_cross_entropy_with_logits(
            masked_logits,
            masked_labels,
            reduction="mean",
        )
        batch_pos_weight = 1.0
    
    return loss, batch_pos_weight
