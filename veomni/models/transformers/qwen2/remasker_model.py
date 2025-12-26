# veomni/models/transformers/qwen2/remasker_model.py

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
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


class Remasker(nn.Module):
    """
    Remasker model that predicts token correctness.
    
    Takes as input:
        - x_0: predicted tokens from denoiser [B, L]
        - hidden_states: hidden states from backbone [B, L, backbone_hidden_size]
    
    Outputs:
        - correctness_logits: logits indicating token correctness [B, L]
          (higher = more likely correct, used for Gumbel sampling)
    """
    
    def __init__(self, config: RemaskerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding for x_0
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Projection for backbone hidden states (if different size)
        if config.backbone_hidden_size != config.hidden_size:
            self.hidden_proj = nn.Linear(config.backbone_hidden_size, config.hidden_size)
        else:
            self.hidden_proj = nn.Identity()
        
        # Combination layer (embedding + projected hidden states)
        self.combine_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # Get Qwen2 config for decoder layers
        qwen2_config = config.to_qwen2_config()
        
        # Import here to avoid circular imports
        from .modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(qwen2_config, layer_idx=i) 
            for i in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Rotary embeddings
        self.rotary_emb = Qwen2RotaryEmbedding(config=qwen2_config)
        
        # Binary classification head (outputs 1 logit per token)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        x_0: torch.LongTensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the remasker.
        
        Args:
            x_0: Predicted token ids [B, L]
            hidden_states: Hidden states from backbone [B, L, backbone_hidden_size]
            attention_mask: Optional attention mask [B, L]
        
        Returns:
            correctness_logits: Logits indicating token correctness [B, L]
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # Embed x_0 tokens
        token_embeds = self.token_embedding(x_0)  # [B, L, hidden_size]
        
        # Project backbone hidden states
        projected_hidden = self.hidden_proj(hidden_states)  # [B, L, hidden_size]
        
        # Combine embeddings and hidden states
        combined = torch.cat([token_embeds, projected_hidden], dim=-1)  # [B, L, hidden_size * 2]
        hidden = self.combine_proj(combined)  # [B, L, hidden_size]
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get rotary embeddings
        position_embeddings = self.rotary_emb(hidden, position_ids)
        
        # Create causal mask if needed (we use bidirectional attention for remasking)
        causal_mask = None
        if attention_mask is not None:
            # Expand attention mask for all heads
            causal_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_len, seq_len
            ).to(hidden.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(hidden.dtype).min
        
        # Pass through decoder layers (bidirectional)
        for layer in self.layers:
            layer_outputs = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                is_causal=False,  # Bidirectional attention
            )
            hidden = layer_outputs[0]
        
        # Final layer norm
        hidden = self.norm(hidden)
        
        # Binary classification
        logits = self.classifier(hidden).squeeze(-1)  # [B, L]
        
        return logits
    
    def save_pretrained(self, save_path: str):
        """Save model and config."""
        import os
        import json
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save model weights
        model_path = os.path.join(save_path, "model.pt")
        torch.save(self.state_dict(), model_path)
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cpu") -> "Remasker":
        """Load model from checkpoint."""
        import os
        import json
        
        # Load config
        config_path = os.path.join(load_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = RemaskerConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model_path = os.path.join(load_path, "model.pt")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        return model
    
    @classmethod
    def from_backbone(
        cls, 
        backbone_model, 
        config: RemaskerConfig,
        init_embedding: bool = True,
        init_layers: bool = True,
        layer_offset: int = 0,
    ) -> "Remasker":
        """
        Initialize remasker from a pretrained backbone (Qwen2) model.
        
        Args:
            backbone_model: Pretrained Qwen2 model (from transformers)
            config: RemaskerConfig for the remasker
            init_embedding: Whether to initialize token_embedding from backbone
            init_layers: Whether to initialize transformer layers from backbone
            layer_offset: Which backbone layer to start copying from.
                         E.g., if backbone has 24 layers and remasker has 4,
                         layer_offset=20 copies layers 20-23 (last 4 layers).
        
        Returns:
            Remasker model with weights initialized from backbone
        """
        # Create remasker with random init
        model = cls(config)
        
        # Get backbone's model component (handle different wrapper structures)
        if hasattr(backbone_model, 'model'):
            backbone = backbone_model.model
        else:
            backbone = backbone_model
        
        # Initialize token embedding from backbone
        if init_embedding and hasattr(backbone, 'embed_tokens'):
            if backbone.embed_tokens.weight.shape == model.token_embedding.weight.shape:
                model.token_embedding.weight.data.copy_(backbone.embed_tokens.weight.data)
                print(f"Initialized token_embedding from backbone embed_tokens")
            else:
                print(f"Warning: Embedding shapes don't match, skipping. "
                      f"Backbone: {backbone.embed_tokens.weight.shape}, "
                      f"Remasker: {model.token_embedding.weight.shape}")
        
        # Initialize transformer layers from backbone
        if init_layers and hasattr(backbone, 'layers'):
            backbone_num_layers = len(backbone.layers)
            remasker_num_layers = len(model.layers)
            
            if layer_offset + remasker_num_layers > backbone_num_layers:
                print(f"Warning: layer_offset={layer_offset} + remasker_layers={remasker_num_layers} "
                      f"> backbone_layers={backbone_num_layers}. Adjusting offset.")
                layer_offset = max(0, backbone_num_layers - remasker_num_layers)
            
            for i in range(remasker_num_layers):
                backbone_layer_idx = layer_offset + i
                try:
                    # Copy layer weights
                    backbone_layer_state = backbone.layers[backbone_layer_idx].state_dict()
                    model.layers[i].load_state_dict(backbone_layer_state)
                    print(f"Initialized remasker layer {i} from backbone layer {backbone_layer_idx}")
                except Exception as e:
                    print(f"Warning: Could not copy layer {backbone_layer_idx} -> {i}: {e}")
        
        # Initialize final norm from backbone
        if hasattr(backbone, 'norm'):
            try:
                model.norm.load_state_dict(backbone.norm.state_dict())
                print(f"Initialized norm from backbone")
            except Exception as e:
                print(f"Warning: Could not copy norm: {e}")
        
        # Initialize classifier with zeros (new layer, start neutral)
        nn.init.zeros_(model.classifier.weight)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print(f"Initialized classifier with zeros")
        
        return model

