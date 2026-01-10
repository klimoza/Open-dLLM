# veomni/models/transformers/qwen2/remasking/config.py

"""Configuration classes for the Remasker model and training."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


@dataclass
class RemaskerConfig:
    """Configuration for the Remasker model."""
    # Model architecture
    num_layers: int = 4
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    backbone_hidden_size: int = 896  # Hidden size of the backbone model
    use_hidden_states: bool = True  # Whether to condition on backbone hidden states
    
    # Attention settings
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    
    # Training settings
    initializer_range: float = 0.02
    
    def to_qwen2_config(self) -> Qwen2Config:
        """Convert to Qwen2Config for reusing Qwen2DecoderLayer."""
        return Qwen2Config(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            attention_dropout=self.attention_dropout,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            use_sliding_window=self.use_sliding_window,
            sliding_window=self.sliding_window,
            initializer_range=self.initializer_range,
            _attn_implementation="eager",
        )
    
    def to_dict(self):
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "backbone_hidden_size": self.backbone_hidden_size,
            "use_hidden_states": self.use_hidden_states,
            "attention_dropout": self.attention_dropout,
            "hidden_act": self.hidden_act,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window": self.sliding_window,
            "initializer_range": self.initializer_range,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "RemaskerConfig":
        return cls(**d)


@dataclass
class RemaskerTrainingConfig:
    """Configuration for training the Remasker."""
    # Model paths
    backbone_path: str = "./models/qwen2-0.5b"
    checkpoint_name: str = "remasker_v1"
    checkpoint_dir: str = "./checkpoints"
    
    # Remasker architecture
    remasker_num_layers: int = 4
    remasker_hidden_size: Optional[int] = None  # If None, use backbone hidden size
    remasker_intermediate_size: Optional[int] = None
    remasker_num_attention_heads: Optional[int] = None
    remasker_num_key_value_heads: Optional[int] = None
    
    # Remasker initialization from backbone
    init_from_backbone: bool = False  # Initialize remasker layers from backbone
    init_layer_offset: int = -1  # Which backbone layer to start from (-1 = auto: use last N layers)
    use_hidden_states: bool = True  # Whether to condition remasker on backbone hidden states
    
    # Corruption settings
    random_corruption_ratio: float = 0.1  # a% of tokens changed to random
    repeat_corruption_ratio: float = 0.1  # b% of tokens changed to repeating
    
    # Dataset
    dataset_path: str = "nvidia/OpenCodeInstruct"
    dataset_name: Optional[str] = None
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    max_seq_length: int = 2048
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine" or "linear"
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "remasker-training"
    wandb_run_name: Optional[str] = None
    
    # Class reweighting
    use_class_reweighting: bool = True  # Reweight loss to handle class imbalance
    
    # Label smoothing
    label_smoothing_alpha: float = 0.0  # If > 0, use soft labels: 0 -> alpha, 1 -> 1-alpha
    
    # Denoising training mode
    use_denoising_training: bool = False  # If True, use denoising-based training that matches inference
    denoising_t_on: float = 0.1  # Upper bound for timestep sampling
    denoising_t_off: float = 0.1  # Lower bound for timestep sampling
    denoising_temperature: float = 0.0  # Temperature for sampling x_0 from logits (0 = greedy)
    denoising_num_steps: int = 4  # Number of denoising steps (1 = single-step, >1 = multi-step entropy-based)
    
    # Other
    seed: int = 42
    num_workers: int = 4
    save_every_n_steps: int = 1000  # Save checkpoint every N optimization steps
    eval_ratio: float = 0.05  # Fraction of data for evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True

