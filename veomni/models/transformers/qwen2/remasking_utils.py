# veomni/models/transformers/qwen2/remasking_utils.py

import torch


def compute_alpha(t: float, schedule: str, t_on: float, t_off: float, alpha_on: float, eps: float = 1e-3) -> float:
    """
    Compute alpha (ratio of unmasked tokens) based on schedule type.
    
    Args:
        t: Current timestep, goes from 1 (start) to eps (end)
        schedule: Either "loop" or "linear"
        t_on: Upper bound of remasking interval (t_on > t_off)
        t_off: Lower bound of remasking interval
        alpha_on: Alpha value during the plateau phase (for loop mode)
        eps: Small value close to 0 representing the final timestep
    
    Returns:
        alpha: Ratio of tokens to keep unmasked (0 to 1)
    
    Loop mode:
        - t ∈ [1, t_on]: α increases linearly from 0 to α_on
        - t ∈ [t_on, t_off]: α = α_on (constant plateau)
        - t ∈ [t_off, eps]: α increases linearly from α_on to 1
    
    Linear mode:
        - α increases linearly from 0 to 1 as t goes from 1 to eps
    """
    if schedule == "linear":
        # Linear interpolation: alpha goes from 0 (at t=1) to 1 (at t=eps)
        alpha = (1.0 - t) / (1.0 - eps)
        return min(max(alpha, 0.0), 1.0)
    
    elif schedule == "loop":
        if t >= t_on:
            # Phase 1: t ∈ [1, t_on], alpha goes from 0 to alpha_on
            # At t=1, alpha=0; at t=t_on, alpha=alpha_on
            alpha = alpha_on * (1.0 - t) / (1.0 - t_on)
            return min(max(alpha, 0.0), alpha_on)
        
        elif t >= t_off:
            # Phase 2: t ∈ [t_on, t_off], alpha stays at alpha_on
            return alpha_on
        
        else:
            # Phase 3: t ∈ [t_off, eps], alpha goes from alpha_on to 1
            # At t=t_off, alpha=alpha_on; at t=eps, alpha=1
            alpha = alpha_on + (1.0 - alpha_on) * (t_off - t) / (t_off - eps)
            return min(max(alpha, alpha_on), 1.0)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule}. Expected 'loop' or 'linear'.")


def is_remasking_active(t: float, t_on: float, t_off: float) -> bool:
    """
    Check if remasking should be applied at timestep t.
    
    Remasking is active when t is in the interval [t_off, t_on].
    Note: t_on > t_off since t decreases from 1 to 0.
    
    Args:
        t: Current timestep
        t_on: Upper bound of remasking interval
        t_off: Lower bound of remasking interval
    
    Returns:
        True if remasking should be applied, False otherwise
    """
    return t_off <= t <= t_on


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


def load_remasker_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load a trained remasker model from checkpoint.
    
    Args:
        checkpoint_path: Path to the remasker checkpoint directory
        device: Device to load model on
    
    Returns:
        Loaded Remasker model
    """
    from .remasker_model import Remasker
    model = Remasker.from_pretrained(checkpoint_path, device=device)
    model.eval()
    return model


def get_remasking_logits(
    batch_size: int,
    seq_len: int,
    candidate_mask: torch.Tensor,
    source: str = "random",
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    # Additional parameters for model-based remasking
    x_0: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    remasker_model = None,
    attention_mask: torch.Tensor = None,
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
        device: Device to create tensor on
        dtype: Data type for the logits tensor
        x_0: Predicted token ids [B, L] (required for source="model")
        hidden_states: Hidden states from backbone [B, L, D] (required for source="model")
        remasker_model: Trained Remasker model instance (required for source="model")
        attention_mask: Optional attention mask [B, L]
    
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
            )
        
        # Ensure correct dtype
        logits = logits.to(dtype)
    
    elif source == "confs":
        # Use model confidence scores (to be implemented)
        # This would use the confidence from sample_tokens
        raise NotImplementedError("Remasking logits source 'confs' not yet implemented.")
    
    else:
        raise NotImplementedError(f"Remasking logits source '{source}' not implemented. Use 'random' or 'model'.")
    
    # Mask out non-candidate positions with -inf
    logits = logits.masked_fill(~candidate_mask, float('-inf'))
    
    return logits


