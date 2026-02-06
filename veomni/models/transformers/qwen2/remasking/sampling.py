# veomni/models/transformers/qwen2/remasking/sampling.py

"""Sampling utilities for remasking (Gumbel sampling, logits generation)."""

import torch


def sample_indices_gumbel(logits: torch.Tensor, num_to_select: torch.Tensor) -> torch.Tensor:
    """
    Sample indices using Gumbel-top-k trick (sampling without replacement).
    
    Uses the Gumbel-max trick: adding Gumbel noise to log-probabilities and taking
    the top-k gives samples without replacement from the categorical distribution.
    
    Args:
        logits: Tensor of shape [B, L] containing logits/scores for each position
        num_to_select: Tensor of shape [B] containing number of indices to select per sample
    
    Returns:
        selected_mask: Boolean tensor of shape [B, L], True for selected positions
    """
    batch_size, seq_len = logits.shape
    device = logits.device
    
    # Add Gumbel noise for sampling without replacement
    # Gumbel(0, 1) = -log(-log(U)), U ~ Uniform(0, 1)
    uniform = torch.rand_like(logits).clamp_(min=1e-20, max=1 - 1e-20)
    gumbel_noise = -torch.log(-torch.log(uniform))
    
    # Add noise to logits
    noisy_logits = logits + gumbel_noise
    
    # Get max number to select for efficient batching
    max_k = int(num_to_select.max().item())
    
    if max_k <= 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    
    # Clamp max_k to not exceed sequence length
    max_k = min(max_k, seq_len)
    
    # Get top-k indices
    _, topk_indices = torch.topk(noisy_logits, max_k, dim=1)  # [B, max_k]
    
    # Create mask for valid selections (handle variable num_to_select per sample)
    valid_mask = torch.arange(max_k, device=device).unsqueeze(0) < num_to_select.unsqueeze(1)  # [B, max_k]
    
    # Build selection mask
    selected_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Get valid indices
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(topk_indices)
    valid_batch_idx = batch_indices[valid_mask]
    valid_seq_idx = topk_indices[valid_mask]
    
    selected_mask[valid_batch_idx, valid_seq_idx] = True
    
    return selected_mask


def get_remasking_logits(
    batch_size: int,
    seq_len: int,
    candidate_mask: torch.Tensor,
    source: str = "random",
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    temperature: float = 1.0,
    # Additional parameters for model-based remasking
    x_0: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    remasker_model = None,
    attention_mask: torch.Tensor = None,
    timestep: torch.Tensor = None,
    confidence: torch.Tensor = None,
    x_t: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate logits for remasking selection.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        candidate_mask: Boolean tensor [B, L] indicating which positions are candidates
                       (True = can be selected for unmasking)
        source: Source of logits. Options:
                - "random": uniform random logits
                - "model": use trained remasker model
                - "backbone": use backbone confidence directly (equivalent to p2 algorithm)
        device: Device to create tensor on
        dtype: Data type for the logits tensor
        temperature: Temperature for scaling logits (default 1.0). Higher values make
                    selection more uniform, lower values make it more deterministic.
                    When temperature=0, selection is fully deterministic (top-k by logits).
        x_0: Predicted token ids [B, L] (required for source="model")
        hidden_states: Hidden states from backbone [B, L, D] (required for source="model")
        remasker_model: Trained Remasker model instance (required for source="model")
        attention_mask: Optional attention mask [B, L]
        timestep: Timestep/noise level [B] (optional, for time-conditioned remasker models)
        confidence: Backbone prediction confidence [B, L] (optional, for confidence-conditioned models)
        x_t: Noisy/masked token ids [B, L] (optional, for x_t-conditioned remasker models)
    
    Returns:
        logits: Tensor of shape [B, L] with logits for selection.
                Higher logits = more likely to be correct = more likely to be kept unmasked.
                Non-candidate positions have -inf to exclude them.
    """
    if source == "random":
        # Random logits (uniform distribution for Gumbel sampling)
        logits = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
    
    elif source == "model":
        # Use trained remasker model to predict correctness
        if remasker_model is None:
            raise ValueError("remasker_model must be provided when source='model'")
        if x_0 is None:
            raise ValueError("x_0 must be provided when source='model'")
        if hidden_states is None:
            raise ValueError("hidden_states must be provided when source='model'")
        
        # Run remasker model
        with torch.no_grad():
            logits = remasker_model(
                x_0=x_0,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                timestep=timestep,
                confidence=confidence,
                x_t=x_t,
            )
        
        # Ensure correct dtype
        logits = logits.to(dtype)
        
        # Apply temperature scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            # Temperature=0: deterministic selection (top-k without randomness)
            # Scale logits high to make Gumbel noise negligible
            logits = logits * 1e6
    
    elif source in ["backbone", "confs"]:
        # Use backbone confidence directly as remasking logits
        # Higher confidence = higher logits = more likely to be kept unmasked
        # This makes the algorithm equivalent to p2 when t_on=1, t_off=0
        if confidence is None:
            raise ValueError(f"confidence must be provided when source='{source}'")
        
        # Use confidence directly as logits
        logits = confidence.clone().to(dtype)
        
        # Apply temperature scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            # Temperature=0: deterministic selection (top-k without randomness)
            # Scale logits high to make Gumbel noise negligible
            logits = logits * 1e6
    
    else:
        raise NotImplementedError(f"Remasking logits source '{source}' not implemented. Use 'random', 'model', or 'backbone'.")
    
    # Mask out non-candidate positions with -inf
    logits = logits.masked_fill(~candidate_mask, float('-inf'))
    
    return logits
