# veomni/models/transformers/qwen2/remasking/metrics.py

"""Metrics computation for remasker training."""

from typing import Dict

import torch


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute classification metrics for tokens that propagate loss.
    
    Args:
        logits: Model output logits [B, L]
        labels: Ground truth labels (1=correct, 0=corrupted) [B, L]
        loss_mask: Mask for positions to compute metrics on [B, L]
    
    Returns:
        Dictionary with positive_ratio, pred_positive_ratio, pred_avg_prob, precision, recall
    """
    # Get probabilities and predictions
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    # Only consider masked positions
    masked_probs = probs[loss_mask]
    masked_preds = preds[loss_mask]
    masked_labels = labels[loss_mask]
    
    total_tokens = masked_labels.numel()
    if total_tokens == 0:
        return {"positive_ratio": 0.0, "pred_positive_ratio": 0.0, "pred_avg_prob": 0.0, "precision": 0.0, "recall": 0.0}
    
    # Positive class ratio in ground truth (ratio of correct/non-corrupted tokens)
    positive_ratio = masked_labels.sum().item() / total_tokens
    
    # Positive class ratio in predictions (ratio of tokens predicted as correct)
    pred_positive_ratio = masked_preds.sum().item() / total_tokens
    
    # Average probability output by classifier
    pred_avg_prob = masked_probs.mean().item()
    
    # True positives: predicted correct AND actually correct
    tp = ((masked_preds == 1) & (masked_labels == 1)).sum().item()
    # False positives: predicted correct BUT actually corrupted
    fp = ((masked_preds == 1) & (masked_labels == 0)).sum().item()
    # False negatives: predicted corrupted BUT actually correct
    fn = ((masked_preds == 0) & (masked_labels == 1)).sum().item()
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "positive_ratio": positive_ratio,
        "pred_positive_ratio": pred_positive_ratio,
        "pred_avg_prob": pred_avg_prob,
        "precision": precision,
        "recall": recall,
    }

