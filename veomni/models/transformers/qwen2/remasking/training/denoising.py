# veomni/models/transformers/qwen2/remasking/training/denoising.py

"""Denoising-specific logic for remasker training."""

from typing import List, Optional, Tuple

import torch

from ..corruption import corrupt_completion, sample_tokens_from_logits, multi_step_denoise
from ..scheduling import compute_alpha


def create_x_t_and_mask(
    ground_truth_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    alpha: float,
    mask_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create masked input x_t from ground truth by masking (1-alpha) fraction of completion tokens.
    
    Args:
        ground_truth_ids: Ground truth token ids [B, L]
        attention_mask: Attention mask [B, L]
        prompt_lens: Length of prompt for each sample [B]
        alpha: Fraction of tokens to keep unmasked
        mask_token_id: Token id to use for masking
        device: Device for tensor operations
    
    Returns:
        x_t: Masked sequence [B, L]
        mask_positions: Boolean mask of masked positions [B, L]
        fix_mask: Boolean mask of prompt positions (never modify) [B, L]
    """
    batch_size = ground_truth_ids.shape[0]
    
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
            perm = torch.randperm(completion_len, device=device)
            mask_indices = perm[:num_to_mask]
            
            # Apply masking
            x_t[b, prompt_len + mask_indices] = mask_token_id
            mask_positions[b, prompt_len + mask_indices] = True
    
    # Create fix_mask: True for prompt positions (never modify these)
    fix_mask = torch.zeros_like(x_t, dtype=torch.bool)
    for b in range(batch_size):
        prompt_len = prompt_lens[b].item()
        fix_mask[b, :prompt_len] = True
    
    return x_t, mask_positions, fix_mask


def get_x0_predictions(
    x_t: torch.Tensor,
    ground_truth_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    fix_mask: torch.Tensor,
    backbone,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    mask_token_id: int,
    config,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Get x_0 predictions from backbone via single-step or multi-step denoising.
    
    Args:
        x_t: Masked input sequence [B, L]
        ground_truth_ids: Ground truth token ids [B, L]
        mask_positions: Boolean mask of masked positions [B, L]
        fix_mask: Boolean mask of prompt positions [B, L]
        backbone: The backbone model
        attention_mask: Attention mask [B, L]
        prompt_lens: Length of prompt for each sample [B]
        mask_token_id: Token id used for masking
        config: Training configuration
    
    Returns:
        x_0_full: Predicted tokens with ground truth for prompts [B, L]
        hidden_states: Hidden states from backbone if use_hidden_states, else None
        confidence_full: Confidence scores if use_confidence_conditioning, else None
    """
    batch_size = x_t.shape[0]
    
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
        
        # Compute confidence for multi-step: run backbone on final x_0_full to get logits
        if config.use_confidence_conditioning:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                final_outputs = backbone(
                    input_ids=x_0_full,
                    attention_mask=attention_mask,
                    is_causal=False,
                )
                final_logits = final_outputs.logits
                final_logits = torch.cat([final_logits[:, :1], final_logits[:, :-1]], dim=1)
                probs = torch.softmax(final_logits.float(), dim=-1)
                confidence_full = torch.gather(probs, -1, x_0_full.unsqueeze(-1)).squeeze(-1)
                # Prompt positions: set to 1.0
                for b in range(batch_size):
                    prompt_len = prompt_lens[b].item()
                    confidence_full[b, :prompt_len] = 1.0
        else:
            confidence_full = None
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
            
            # Compute confidence for all positions (p2-style: probability of token in x_0_full)
            # The backbone predicts logits for ALL positions, even unmasked ones
            if config.use_confidence_conditioning:
                probs = torch.softmax(backbone_logits.float(), dim=-1)
                confidence_full = torch.gather(probs, -1, x_0_full.unsqueeze(-1)).squeeze(-1)
                # Prompt positions: set to 1.0 (not in backbone's prediction scope)
                for b in range(batch_size):
                    prompt_len = prompt_lens[b].item()
                    confidence_full[b, :prompt_len] = 1.0
            else:
                confidence_full = None
    
    return x_0_full, hidden_states, confidence_full


def apply_x_t_conditioning(
    x_0_full: torch.Tensor,
    ground_truth_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    alpha: float,
    mask_token_id: int,
    backbone,
    config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Apply double denoising scheme for x_t conditioning.
    
    This creates a second denoising step: pred_x_0 -> pred_x_t -> pred_pred_x_0
    which provides the remasker with both x_0 and x_t as conditioning signals.
    
    Args:
        x_0_full: First-step x_0 predictions [B, L]
        ground_truth_ids: Ground truth token ids [B, L]
        attention_mask: Attention mask [B, L]
        prompt_lens: Length of prompt for each sample [B]
        alpha: Fraction of tokens to keep unmasked
        mask_token_id: Token id used for masking
        backbone: The backbone model
        config: Training configuration
    
    Returns:
        x_0_for_remasker: Input x_0 for remasker [B, L]
        x_t_for_remasker: Input x_t for remasker cross-attention [B, L]
        labels: Labels for training (1 if correct, 0 if wrong) [B, L]
        hidden_states: Updated hidden states if use_hidden_states
        confidence_full: Updated confidence if use_confidence_conditioning
    """
    batch_size = x_0_full.shape[0]
    device = x_0_full.device
    
    # x_0_full is now pred_x_0 (first denoising step result)
    pred_x_0 = x_0_full.clone()
    
    # Step 3: pred_x_0 -> pred_x_t (apply masking to pred_x_0)
    pred_x_t = pred_x_0.clone()
    pred_mask_positions = torch.zeros_like(pred_x_t, dtype=torch.bool)
    
    for b in range(batch_size):
        prompt_len = prompt_lens[b].item()
        completion_len = (attention_mask[b].sum().item()) - prompt_len
        if completion_len <= 0:
            continue
        
        # Apply same masking ratio to pred_x_0
        num_to_mask = int(completion_len * (1 - alpha))
        if num_to_mask > 0:
            perm = torch.randperm(completion_len, device=device)
            mask_indices = perm[:num_to_mask]
            pred_x_t[b, prompt_len + mask_indices] = mask_token_id
            pred_mask_positions[b, prompt_len + mask_indices] = True
    
    # Step 4: pred_x_t -> pred_pred_x_0 (denoise pred_x_t)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
        pred_backbone_outputs = backbone(
            input_ids=pred_x_t,
            attention_mask=attention_mask,
            output_hidden_states=config.use_hidden_states,
            return_dict=True,
            is_causal=False,
        )
        pred_logits = pred_backbone_outputs.logits
        pred_logits = torch.cat([pred_logits[:, :1], pred_logits[:, :-1]], dim=1)
        
        # Sample pred_pred_x_0
        pred_pred_x_0 = sample_tokens_from_logits(
            pred_logits,
            temperature=config.denoising_temperature
        )
        
        # Build pred_pred_x_0_full: predictions for masked positions, keep unmasked from pred_x_0
        pred_pred_x_0_full = pred_x_0.clone()
        pred_pred_x_0_full[pred_mask_positions] = pred_pred_x_0[pred_mask_positions]
        
        # Update hidden_states to be from pred_x_t if using hidden states
        if config.use_hidden_states:
            hidden_states = pred_backbone_outputs.hidden_states[-1]
        else:
            hidden_states = None
        
        # Compute confidence for pred_pred_x_0_full if confidence conditioning is enabled
        if config.use_confidence_conditioning:
            probs = torch.softmax(pred_logits.float(), dim=-1)
            confidence_full = torch.gather(probs, -1, pred_pred_x_0_full.unsqueeze(-1)).squeeze(-1)
            for b in range(batch_size):
                prompt_len = prompt_lens[b].item()
                confidence_full[b, :prompt_len] = 1.0
        else:
            confidence_full = None
    
    # Now use pred_pred_x_0_full as the input to remasker, with pred_x_t as cross-attention context
    x_0_for_remasker = pred_pred_x_0_full
    x_t_for_remasker = pred_x_t
    
    # Labels: compare pred_pred_x_0_full to ground_truth
    # (we want remasker to identify which tokens are correct)
    prediction_correct = (x_0_for_remasker == ground_truth_ids)
    labels = prediction_correct.float()
    
    # Prompt tokens are always labeled as correct
    for b in range(batch_size):
        prompt_len = prompt_lens[b].item()
        labels[b, :prompt_len] = 1.0
    
    return x_0_for_remasker, x_t_for_remasker, labels, hidden_states, confidence_full


def apply_augmentations(
    x_0_full: torch.Tensor,
    ground_truth_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    vocab_size: int,
    random_corruption_ratio: float,
    repeat_corruption_ratio: float,
    special_token_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentations (random/repeat corruption) to completion tokens.
    
    Args:
        x_0_full: Predicted tokens [B, L]
        ground_truth_ids: Ground truth token ids [B, L]
        attention_mask: Attention mask [B, L]
        prompt_lens: Length of prompt for each sample [B]
        vocab_size: Size of vocabulary
        random_corruption_ratio: Fraction of tokens to replace with random tokens
        repeat_corruption_ratio: Fraction of tokens to replace with repeated tokens
        special_token_ids: Token ids to exclude from corruption
    
    Returns:
        x_0_for_remasker: Augmented x_0 for remasker input [B, L]
        labels: Labels (1 if correct and not augmented, 0 otherwise) [B, L]
    """
    batch_size = x_0_full.shape[0]
    
    # Apply augmentations per-sample since completion lengths vary
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
            vocab_size=vocab_size,
            random_ratio=random_corruption_ratio,
            repeat_ratio=repeat_corruption_ratio,
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
    
    return x_0_full, labels
