# veomni/models/transformers/qwen2/remasking/training/corruption_mode.py

"""Corruption-based training mode for remasker."""

from typing import Optional, Tuple

import torch

from .loss import compute_remasker_loss


def forward_corruption_mode(
    model,
    backbone,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    config,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Forward pass for corruption-based training mode.
    
    This is the original training mode where the dataset provides pre-corrupted
    tokens and labels indicating which tokens are correct.
    
    Args:
        model: The remasker model
        backbone: The backbone model (for hidden states)
        input_ids: Input token ids [B, L]
        labels: Ground truth labels [B, L]
        loss_mask: Mask indicating which positions to include in loss [B, L]
        attention_mask: Attention mask [B, L]
        config: Training configuration
    
    Returns:
        loss: Computed loss tensor (already divided by gradient_accumulation_steps)
        logits: Model output logits [B, L]
        pos_weight: The positive class weight used (for logging)
    """
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
    # Note: timestep and confidence are None for corruption-based training (no denoising process)
    # If use_time_conditioning=True or use_confidence_conditioning=True, this will raise an error
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
        logits = model(
            x_0=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask.float(),
            timestep=None,
            confidence=None,
        )
        
        # Compute loss
        loss, pos_weight = compute_remasker_loss(
            logits=logits,
            labels=labels,
            loss_mask=loss_mask,
            use_class_reweighting=config.use_class_reweighting,
            label_smoothing_alpha=config.label_smoothing_alpha,
            device=config.device,
            pos_class_weight=config.pos_class_weight,
            use_ranknet_pairwise_loss=config.use_ranknet_pairwise_loss,
        )
        loss = loss / config.gradient_accumulation_steps
    
    return loss, logits, pos_weight
