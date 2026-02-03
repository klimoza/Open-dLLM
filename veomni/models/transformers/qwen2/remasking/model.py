# veomni/models/transformers/qwen2/remasking/model.py

"""Remasker model for predicting token correctness."""

import math
from typing import Optional

import torch
import torch.nn as nn

from .config import RemaskerConfig


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings (DDPM-style).
    
    Args:
        timesteps: Tensor of shape [B] containing timestep values (typically in [0, 1])
        embedding_dim: Dimension of the output embedding
    
    Returns:
        Embeddings of shape [B, embedding_dim]
    """
    assert len(timesteps.shape) == 1, "timesteps must be 1D tensor"
    
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    # Handle odd embedding dimensions
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    
    return emb


class TimestepFiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) conditioning from timestep.
    
    Predicts scale (γ) and shift (β) from timestep to modulate features:
        output = γ * LayerNorm(x) + β
    
    Initialized so that γ=1 and β=0, meaning the output starts as identity.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        
        # MLP to predict scale and shift (2 * hidden_size output)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),  # [scale, shift]
        )
        
        # Layer norm for the input features
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        
        # Initialize final layer to zero so scale=1, shift=0 at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, hidden_size]
            timesteps: Tensor of shape [B] containing timestep values
        
        Returns:
            Modulated features [B, L, hidden_size]
        """
        # Get FiLM parameters from timestep
        t_freq = get_timestep_embedding(timesteps, self.frequency_embedding_size)
        film_params = self.mlp(t_freq)  # [B, hidden_size * 2]
        
        # Split into scale and shift
        scale, shift = film_params.chunk(2, dim=-1)  # [B, hidden_size] each
        
        # scale starts at 0, we add 1 so it starts at 1 (identity)
        scale = 1 + scale
        
        # Broadcast over sequence length: [B, hidden_size] -> [B, 1, hidden_size]
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        
        # Apply FiLM: γ * LayerNorm(x) + β
        x_norm = self.norm(x)
        return scale * x_norm + shift


class ConfidenceFiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) conditioning from confidence scores.
    
    Predicts per-token scale (γ) and shift (β) from confidence to modulate features:
        output = γ * LayerNorm(x) + β
    
    Initialized so that γ=1 and β=0, meaning the output starts as identity.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        
        # MLP to predict scale and shift (2 * hidden_size output)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),  # [scale, shift]
        )
        
        # Layer norm for the input features
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        
        # Initialize final layer to zero so scale=1, shift=0 at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, hidden_size]
            confidence: Tensor of shape [B, L] containing confidence values (0 to 1)
        
        Returns:
            Modulated features [B, L, hidden_size]
        """
        # Get FiLM parameters from confidence
        # confidence: [B, L] -> [B, L, 1]
        film_params = self.mlp(confidence.unsqueeze(-1))  # [B, L, hidden_size * 2]
        
        # Split into scale and shift
        scale, shift = film_params.chunk(2, dim=-1)  # [B, L, hidden_size] each
        
        # scale starts at 0, we add 1 so it starts at 1 (identity)
        scale = 1 + scale
        
        # Apply FiLM: γ * LayerNorm(x) + β
        x_norm = self.norm(x)
        return scale * x_norm + shift


class CrossAttention(nn.Module):
    """
    Cross-attention module for conditioning on x_t.
    
    Query comes from the main hidden states (x_0 processing),
    Key/Value come from x_t embeddings.
    
    Initialized to be close to identity (residual connection dominates at start).
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_attention_heads: int,
        num_key_value_heads: int = None,
        attention_dropout: float = 0.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = attention_dropout
        
        # Query projection (from main hidden states)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        # Key and Value projections (from x_t embeddings)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # Output projection
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        
        # Layer norm before cross-attention
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        
        # Initialize output projection to zero so cross-attention starts as identity
        nn.init.zeros_(self.o_proj.weight)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match number of query heads."""
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        x_t_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Main hidden states [B, L, hidden_size] (query source)
            x_t_embeds: Embedded x_t tokens [B, L, hidden_size] (key/value source)
            attention_mask: Optional attention mask [B, L]
        
        Returns:
            Output hidden states [B, L, hidden_size]
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries from main hidden states
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)  # [B, num_heads, L, head_dim]
        
        # Project keys and values from x_t embeddings
        key_states = self.k_proj(x_t_embeds)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        key_states = key_states.transpose(1, 2)  # [B, num_kv_heads, L, head_dim]
        
        value_states = self.v_proj(x_t_embeds)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.transpose(1, 2)  # [B, num_kv_heads, L, head_dim]
        
        # Repeat KV heads if needed (grouped-query attention)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for cross-attention: [B, L] -> [B, 1, 1, L]
            cross_attn_mask = attention_mask[:, None, None, :].to(attn_weights.dtype)
            cross_attn_mask = (1.0 - cross_attn_mask) * torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights + cross_attn_mask
        
        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)  # [B, num_heads, L, head_dim]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Residual connection
        return residual + attn_output


class Remasker(nn.Module):
    """
    Remasker model that predicts token correctness.
    
    Takes as input:
        - x_0: predicted tokens from denoiser [B, L]
        - hidden_states: hidden states from backbone [B, L, backbone_hidden_size]
          (optional if use_hidden_states=False in config)
    
    Outputs:
        - correctness_logits: logits indicating token correctness [B, L]
          (higher = more likely correct, used for Gumbel sampling)
    
    When use_hidden_states=False, the model only uses token embeddings without
    conditioning on backbone hidden states.
    """
    
    def __init__(self, config: RemaskerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding for x_0
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Projection for backbone hidden states (if using them)
        if config.use_hidden_states:
            if config.backbone_hidden_size != config.hidden_size:
                self.hidden_proj = nn.Linear(config.backbone_hidden_size, config.hidden_size)
            else:
                self.hidden_proj = nn.Identity()
            
            # Combination layer (embedding + projected hidden states)
            self.combine_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            self.hidden_proj = None
            self.combine_proj = None
        
        # Get Qwen2 config for decoder layers
        qwen2_config = config.to_qwen2_config()
        
        # Import here to avoid circular imports
        from ..modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
        
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
        
        # Time conditioning via FiLM (optional)
        if config.use_time_conditioning:
            self.time_film = TimestepFiLM(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.time_film = None
        
        # Confidence conditioning via FiLM (optional)
        if config.use_confidence_conditioning:
            self.confidence_film = ConfidenceFiLM(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.confidence_film = None
        
        # x_t cross-attention conditioning (optional) - one per layer for richer conditioning
        if config.use_x_t_conditioning:
            self.x_t_cross_attns = nn.ModuleList([
                CrossAttention(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    attention_dropout=config.attention_dropout,
                    eps=config.rms_norm_eps,
                )
                for _ in range(config.num_layers)
            ])
        else:
            self.x_t_cross_attns = None
        
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
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        x_t: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the remasker.
        
        Args:
            x_0: Predicted token ids [B, L]
            hidden_states: Hidden states from backbone [B, L, backbone_hidden_size]
                          (optional if use_hidden_states=False in config)
            attention_mask: Optional attention mask [B, L]
            timestep: Timestep/noise level [B] (optional if use_time_conditioning=False in config)
            confidence: Backbone prediction confidence [B, L] (optional if use_confidence_conditioning=False)
            x_t: Noisy/masked token ids [B, L] (optional if use_x_t_conditioning=False in config)
        
        Returns:
            correctness_logits: Logits indicating token correctness [B, L]
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # Embed x_0 tokens
        token_embeds = self.token_embedding(x_0)  # [B, L, hidden_size]
        
        if self.config.use_hidden_states:
            if hidden_states is None:
                raise ValueError("hidden_states must be provided when use_hidden_states=True")
            # Project backbone hidden states
            projected_hidden = self.hidden_proj(hidden_states)  # [B, L, hidden_size]
            
            # Combine embeddings and hidden states
            combined = torch.cat([token_embeds, projected_hidden], dim=-1)  # [B, L, hidden_size * 2]
            hidden = self.combine_proj(combined)  # [B, L, hidden_size]
        else:
            # Use only token embeddings (no hidden state conditioning)
            hidden = token_embeds
        
        # Apply time conditioning via FiLM if enabled
        if self.config.use_time_conditioning:
            if timestep is None:
                raise ValueError("timestep must be provided when use_time_conditioning=True")
            # Apply FiLM: γ * LayerNorm(hidden) + β, where γ and β depend on timestep
            hidden = self.time_film(hidden, timestep)  # [B, L, hidden_size]
        
        # Apply confidence conditioning via FiLM if enabled
        if self.config.use_confidence_conditioning:
            if confidence is None:
                raise ValueError("confidence must be provided when use_confidence_conditioning=True")
            # Apply FiLM: γ * LayerNorm(hidden) + β, where γ and β depend on confidence
            hidden = self.confidence_film(hidden, confidence)  # [B, L, hidden_size]
        
        # Embed x_t tokens if x_t conditioning is enabled (used for per-layer cross-attention)
        x_t_embeds = None
        if self.config.use_x_t_conditioning:
            if x_t is None:
                raise ValueError("x_t must be provided when use_x_t_conditioning=True")
            # Embed x_t tokens (reuse token_embedding)
            x_t_embeds = self.token_embedding(x_t)  # [B, L, hidden_size]
        
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
        
        # Pass through decoder layers with per-layer cross-attention (bidirectional)
        for layer_idx, layer in enumerate(self.layers):
            # Self-attention layer
            layer_outputs = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                is_causal=False,  # Bidirectional attention
            )
            hidden = layer_outputs[0]
            
            # Apply cross-attention to x_t after each self-attention layer
            if self.config.use_x_t_conditioning:
                hidden = self.x_t_cross_attns[layer_idx](hidden, x_t_embeds, attention_mask)
        
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

