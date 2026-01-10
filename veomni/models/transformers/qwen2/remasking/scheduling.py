# veomni/models/transformers/qwen2/remasking/scheduling.py

"""Scheduling utilities for remasking (alpha computation, timestep management)."""


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

