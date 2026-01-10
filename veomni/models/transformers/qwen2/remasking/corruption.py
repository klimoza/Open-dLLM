# veomni/models/transformers/qwen2/remasking/corruption.py

"""Corruption utilities for remasker training."""

from typing import List, Optional, Tuple

import torch

from .scheduling import compute_alpha


def corrupt_completion(
    completion_ids: torch.Tensor,
    vocab_size: int,
    random_ratio: float,
    repeat_ratio: float,
    special_token_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Corrupt completion tokens.
    
    Args:
        completion_ids: Token ids of the completion [L]
        vocab_size: Size of vocabulary
        random_ratio: Fraction of tokens to replace with random tokens
        repeat_ratio: Fraction of tokens to replace with repeating tokens from completion
        special_token_ids: Token ids to exclude from corruption (e.g., pad, eos)
    
    Returns:
        corrupted_ids: Corrupted token ids [L]
        corruption_mask: Boolean mask, True where token was corrupted [L]
    """
    seq_len = completion_ids.shape[0]
    device = completion_ids.device
    
    if special_token_ids is None:
        special_token_ids = []
    
    # Create mask of positions that can be corrupted (exclude special tokens)
    can_corrupt = torch.ones(seq_len, dtype=torch.bool, device=device)
    for token_id in special_token_ids:
        can_corrupt &= (completion_ids != token_id)
    
    num_corruptible = can_corrupt.sum().item()
    if num_corruptible == 0:
        return completion_ids.clone(), torch.zeros(seq_len, dtype=torch.bool, device=device)
    
    # Calculate number of tokens to corrupt
    num_random = int(num_corruptible * random_ratio)
    num_repeat = int(num_corruptible * repeat_ratio)
    
    # Get indices of corruptible positions
    corruptible_indices = torch.where(can_corrupt)[0]
    
    # Shuffle and select indices for corruption
    perm = torch.randperm(num_corruptible, device=device)
    random_indices = corruptible_indices[perm[:num_random]]
    repeat_indices = corruptible_indices[perm[num_random:num_random + num_repeat]]
    
    # Create corrupted version
    corrupted_ids = completion_ids.clone()
    corruption_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    
    # Apply random corruption
    if num_random > 0:
        random_tokens = torch.randint(0, vocab_size, (num_random,), device=device)
        corrupted_ids[random_indices] = random_tokens
        corruption_mask[random_indices] = True
    
    # Apply repeat corruption (use tokens from elsewhere in completion)
    if num_repeat > 0 and seq_len > 1:
        # Sample source positions (different from target positions)
        source_indices = torch.randint(0, seq_len, (num_repeat,), device=device)
        # Make sure we pick different tokens (at least try)
        for attempt in range(3):  # Try a few times to get different tokens
            same_mask = (corrupted_ids[source_indices] == completion_ids[repeat_indices])
            if not same_mask.any():
                break
            source_indices[same_mask] = torch.randint(0, seq_len, (same_mask.sum().item(),), device=device)
        
        corrupted_ids[repeat_indices] = completion_ids[source_indices]
        # Only mark as corrupted if actually different
        actually_changed = corrupted_ids[repeat_indices] != completion_ids[repeat_indices]
        corruption_mask[repeat_indices] = actually_changed
    
    return corrupted_ids, corruption_mask


def create_masked_sequence(
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    mask_token_id: int,
    t: float,
    t_on: float = 0.55,
    t_off: float = 0.05,
    alpha_on: float = 0.9,
    schedule: str = "linear",
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a masked sequence x_t by masking (1-alpha) fraction of completion tokens.
    
    This simulates the denoising process at timestep t, where alpha(t) determines
    how many tokens are unmasked/revealed.
    
    Args:
        prompt_ids: Token ids of the prompt [P]
        completion_ids: Token ids of the completion (ground truth) [C]
        mask_token_id: The mask token id to use for masked positions
        t: Timestep value (typically sampled from [t_off, t_on])
        t_on: Upper bound of remasking interval
        t_off: Lower bound of remasking interval
        alpha_on: Alpha value during plateau phase (for loop schedule)
        schedule: Either "loop" or "linear"
        eps: Small value representing final timestep
    
    Returns:
        x_t: Masked sequence [P + C] with some completion tokens replaced by mask_token_id
        mask_positions: Boolean mask [P + C], True where completion tokens are masked
    """
    device = completion_ids.device
    prompt_len = prompt_ids.shape[0]
    completion_len = completion_ids.shape[0]
    
    # Compute alpha (fraction of tokens to keep unmasked)
    alpha = compute_alpha(
        t=t,
        schedule=schedule,
        t_on=t_on,
        t_off=t_off,
        alpha_on=alpha_on,
        eps=eps
    )
    
    # Number of completion tokens to keep unmasked
    num_to_keep = int(completion_len * alpha)
    num_to_mask = completion_len - num_to_keep
    
    # Randomly select which positions to mask in completion
    perm = torch.randperm(completion_len, device=device)
    mask_indices = perm[:num_to_mask]  # Indices within completion to mask
    
    # Create masked completion
    masked_completion = completion_ids.clone()
    masked_completion[mask_indices] = mask_token_id
    
    # Combine prompt + masked completion
    x_t = torch.cat([prompt_ids, masked_completion])
    
    # Create mask indicating which positions are masked (in full sequence)
    mask_positions = torch.zeros(prompt_len + completion_len, dtype=torch.bool, device=device)
    mask_positions[prompt_len + mask_indices] = True
    
    return x_t, mask_positions


def sample_tokens_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
) -> torch.Tensor:
    """
    Sample tokens from logits.
    
    Args:
        logits: Logits tensor [*, vocab_size]
        temperature: Sampling temperature (0 = greedy)
    
    Returns:
        Sampled token ids [*]
    """
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(probs.shape[:-1])
    else:
        return logits.argmax(dim=-1)


def _sample_tokens_with_confidence(
    logits: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tokens from logits and compute entropy-based confidence.
    
    Args:
        logits: Logits tensor [B, L, vocab_size]
        temperature: Sampling temperature (0 = greedy)
    
    Returns:
        confidence: Confidence scores [B, L] (higher = more confident, using negative entropy)
        x0: Sampled token ids [B, L]
    """
    if temperature > 0:
        scaled_logits = logits / temperature
    else:
        scaled_logits = logits
    
    probs = torch.softmax(scaled_logits.float(), dim=-1)
    
    if temperature > 0:
        # Sample from distribution
        flat_probs = probs.view(-1, probs.shape[-1])
        x0 = torch.multinomial(flat_probs, num_samples=1).view(probs.shape[:-1])
    else:
        # Greedy selection
        x0 = probs.argmax(dim=-1)
    
    # Compute entropy-based confidence (negative entropy = higher is more confident)
    log_probs = torch.log(probs.clamp(min=1e-10))
    confidence = (probs * log_probs).sum(dim=-1)  # Negative entropy
    
    return confidence, x0


def multi_step_denoise(
    x_t: torch.Tensor,
    backbone,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    num_steps: int = 4,
    temperature: float = 0.0,
    fix_mask: torch.Tensor = None,
    fp16: bool = True,
) -> torch.Tensor:
    """
    Multi-step entropy-based denoising from x_t to x_0.
    
    Runs iterative denoising using the entropy algorithm: at each step,
    tokens are sampled and the highest-confidence ones are unmasked.
    
    Args:
        x_t: Input sequence with some positions masked [B, L]
        backbone: The backbone model for computing logits
        attention_mask: Attention mask [B, L]
        mask_token_id: The mask token id
        num_steps: Number of denoising steps
        temperature: Sampling temperature (0 = greedy)
        fix_mask: Boolean mask [B, L], True = fixed positions (prompt), never change
        fp16: Whether to use fp16 for backbone forward pass
    
    Returns:
        x_0: Denoised sequence [B, L]
    """
    device = x_t.device
    x = x_t.clone()
    
    # If fix_mask not provided, infer from initial non-mask positions
    if fix_mask is None:
        fix_mask = (x_t != mask_token_id)
    
    for step in range(num_steps):
        # Find current mask positions
        mask_index = (x == mask_token_id)
        
        if not mask_index.any():
            # All tokens unmasked, done
            break
        
        # Forward pass through backbone
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            outputs = backbone(
                input_ids=x,
                attention_mask=attention_mask,
                is_causal=False,  # Bidirectional attention for MDM
            )
            logits = outputs.logits
        
        # CRITICAL: Shift logits to predict the next token (matching inference)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        
        # Sample tokens and get confidence for masked positions
        confidence, x0_full = _sample_tokens_with_confidence(logits, temperature=temperature)
        confidence = confidence.to(logits.dtype)
        
        # Only consider masked positions for confidence ranking
        # Set fixed positions to -inf so they're never selected
        full_confidence = torch.full_like(x, float('-inf'), dtype=logits.dtype)
        full_confidence[mask_index] = confidence[mask_index]
        
        # Calculate how many tokens to transfer this step
        num_mask_tokens_per_sample = mask_index.sum(dim=1)  # [B]
        
        # Linear schedule: unmask proportionally more tokens each step
        # At step i (0-indexed), we want (i+1)/num_steps fraction to be unmasked
        # So we transfer enough to reach that target
        if step < num_steps - 1:
            # Transfer fraction for this step
            transfer_ratio = 1.0 / (num_steps - step)
            num_to_transfer = (num_mask_tokens_per_sample.float() * transfer_ratio).long()
        else:
            # Last step: transfer all remaining
            num_to_transfer = num_mask_tokens_per_sample
        
        # Get maximum tokens to transfer for batching
        max_transfer = int(num_to_transfer.max().item())
        
        if max_transfer > 0:
            # Use deterministic top-k selection (alg_temp=0)
            _, transfer_indices = torch.topk(full_confidence, max_transfer, dim=1)  # [B, max_transfer]
            
            # Create mask for valid transfers per sample
            batch_size = x.size(0)
            valid_mask = torch.arange(max_transfer, device=device).unsqueeze(0) < num_to_transfer.unsqueeze(1)
            
            # Get valid indices
            valid_transfer_indices = transfer_indices[valid_mask]
            valid_batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(transfer_indices)[valid_mask]
            
            # Transfer tokens: update x with sampled tokens at selected positions
            x[valid_batch_indices, valid_transfer_indices] = x0_full[valid_batch_indices, valid_transfer_indices]
    return x

