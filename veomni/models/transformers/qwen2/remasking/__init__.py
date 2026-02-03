# veomni/models/transformers/qwen2/remasking/__init__.py

"""
Remasking module for masked diffusion models.

This module provides:
- Remasker model for predicting token correctness
- Training utilities (dataset, training loops, metrics)
- Inference utilities (scheduling, sampling)
"""

from .config import RemaskerConfig, RemaskerTrainingConfig
from .model import Remasker
from .corruption import corrupt_completion, create_masked_sequence, sample_tokens_from_logits, multi_step_denoise
from .scheduling import compute_alpha, is_remasking_active
from .sampling import sample_indices_gumbel, get_remasking_logits
from .dataset import RemaskerDataset, collate_fn, load_data
from .metrics import compute_classification_metrics
from .logging import log_train_step, log_timestep_eval, evaluate_at_timesteps
from .training import train_epoch, evaluate


def load_remasker_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load a trained remasker model from checkpoint.
    
    Args:
        checkpoint_path: Path to the remasker checkpoint directory
        device: Device to load model on
    
    Returns:
        Loaded Remasker model
    """
    model = Remasker.from_pretrained(checkpoint_path, device=device)
    model.eval()
    return model


__all__ = [
    # Config
    "RemaskerConfig",
    "RemaskerTrainingConfig",
    # Model
    "Remasker",
    "load_remasker_model",
    # Corruption
    "corrupt_completion",
    "create_masked_sequence",
    "sample_tokens_from_logits",
    "multi_step_denoise",
    # Scheduling
    "compute_alpha",
    "is_remasking_active",
    # Sampling
    "sample_indices_gumbel",
    "get_remasking_logits",
    # Dataset
    "RemaskerDataset",
    "collate_fn",
    "load_data",
    # Metrics
    "compute_classification_metrics",
    # Logging
    "log_train_step",
    "log_timestep_eval",
    "evaluate_at_timesteps",
    # Training
    "train_epoch",
    "evaluate",
]

