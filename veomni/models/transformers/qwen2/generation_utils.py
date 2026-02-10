# veomni/models/transformers/qwen2/generation_utils.py

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

from .remasking import compute_alpha, is_remasking_active, sample_indices_gumbel, get_remasking_logits, load_remasker_model

logger = logging.get_logger(__name__)

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    if top_k is None or top_k == 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, alg="origin"):
    # original_dtype = logits.dtype
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
    else:
        _, x0 = probs.max(dim=-1)
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if alg == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        confidence = top1_probs - top2_probs
    elif alg == "entropy":
        log_probs = torch.log(probs.clamp(min=1e-10))
        confidence = (probs * log_probs).sum(dim=-1)
    elif alg in ["maskgit_plus", "origin", "p2"]:
        pass
    else:
        raise NotImplementedError(f"Algorithm {alg} not implemented.")
    
    return confidence, x0


@dataclass
class MDMModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class RemaskingConfig:
    def __init__(self, **kwargs):
        self.schedule = kwargs.pop("schedule", "loop")
        # schedule: loop, linear
        self.remasking_t_on = kwargs.pop("remasking_t_on", 0.55)
        self.remasking_t_off = kwargs.pop("remasking_t_off", 0.05)
        self.remasking_alpha_on = kwargs.pop("remasking_alpha_on", 0.9)
        
        self.remasking_logits_source = kwargs.pop("remasking_logits_source", "random")
        # remasking_logits_source: random, model, backbone (backbone = p2-style confidence-based selection)
        self.remasking_temperature = kwargs.pop("remasking_temperature", 1.0)

        self.non_remasking_sampling_algorithm = kwargs.pop("non_remasking_sampling_algorithm", "origin")
        # non_remasking_sampling_algorithm: origin, topk_margin, entropy, maskgit_plus, p2
        
        # Remasker model checkpoint path (for remasking_logits_source="model")
        self.remasker_checkpoint_path = kwargs.pop("remasker_checkpoint_path", None)
        # Cached remasker model instance (loaded lazily)
        self._remasker_model = None

        # Threshold-based adaptive remasking (used with alg="remasking_threshold")
        # Sigmoid is applied to remasking logits, then positions with sigmoid(logit) > threshold are kept.
        # Threshold is in [0, 1] (probability space).
        self.remasking_threshold = kwargs.pop("remasking_threshold", 0.5)
        # Minimum increase in unmasked token count per step (safety floor).
        # After each step, the number of unmasked completion tokens must grow by at least this many.
        # Prevents the algorithm from getting stuck when nothing passes the threshold.
        self.remasking_min_unmask_tokens = kwargs.pop("remasking_min_unmask_tokens", 0)


class MDMGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        # Set do_sample=True as default for MDM (since MDM handles its own sampling)
        if 'do_sample' not in kwargs:
            kwargs['do_sample'] = True
        
        super().__init__(**kwargs)
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'entropy')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", 0.0)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)

        self.remasking_config = kwargs.pop("remasking_config", RemaskingConfig())


class MDMGenerationMixin:
    """
    Mixin class for Masked Diffusion Model generation, adapted from the Dream model's generation utils.
    """
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        if expand_size == 1:
            return input_ids, attention_mask
        
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _mdm_prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs
    ) -> MDMGenerationConfig:
        if generation_config is None:
            generation_config = self.generation_config
        
        # Use MDMGenerationConfig as the target class
        if not isinstance(generation_config, MDMGenerationConfig):
            generation_config = MDMGenerationConfig.from_dict(generation_config.to_dict())

        # Update with kwargs
        generation_config.update(**kwargs)
        return generation_config

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[MDMGenerationConfig] = None,
        **kwargs,
    ) -> Union[MDMModelOutput, torch.LongTensor]:
        
        # 1. Prepare generation config
        generation_config = self._mdm_prepare_generation_config(generation_config, **kwargs)
        # 2. Prepare inputs
        input_ids = inputs
        attention_mask = kwargs.get("attention_mask", None)

        if input_ids is None:
            raise ValueError("`inputs` must be provided for diffusion generation.")

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = input_ids.shape[-1] + generation_config.max_new_tokens
        # 3. Expand inputs for multi-sequence generation
        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        mask_token_id = generation_config.mask_token_id
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be set in the generation config.")
        
        input_ids = F.pad(input_ids, (0, generation_config.max_length - input_ids.shape[1]), value=generation_config.mask_token_id)
        attention_mask = None

        # 4. Run the sampling loop
        return self._mdm_sample(
            x=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
    
    def _mdm_sample(
        self,
        x: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: MDMGenerationConfig
    ) -> Union[MDMModelOutput, torch.LongTensor]:
        
        # Extract params from config

        # import pdb; pdb.set_trace()
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        if mask_token_id is None:
            raise ValueError("`mask_token_id` must be set in the generation config.")

        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if generation_config.output_history else None

        # Pad input_ids to max_length with mask tokens
        # x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        # Fixed tokens = input context (should never be remasked in p2)
        fix_mask = (x != mask_token_id)
        # fix_mask = F.pad(fix_mask, (0, max_length - fix_mask.shape[1]), value=0)

        # The model expects a bidirectional mask, so we just use the presence of pad_token_id
        gen_attention_mask = (x != self.config.pad_token_id).long() if self.config.pad_token_id is not None else None

        # ── Threshold-based remasking: own while-loop (not bound by steps) ──
        if alg == "remasking_threshold":
            remasking_cfg = generation_config.remasking_config
            need_hidden_states = remasking_cfg.remasking_logits_source == "model"

            # Lazy-load remasker model once before the loop
            remasker_model = None
            if remasking_cfg.remasking_logits_source == "model":
                if remasking_cfg._remasker_model is None:
                    if remasking_cfg.remasker_checkpoint_path is None:
                        raise ValueError(
                            "remasker_checkpoint_path must be set in remasking_config when "
                            "remasking_logits_source='model'"
                        )
                    remasking_cfg._remasker_model = load_remasker_model(
                        remasking_cfg.remasker_checkpoint_path,
                        device=str(self.device),
                    )
                    remasking_cfg._remasker_model.to(self.device)
                remasker_model = remasking_cfg._remasker_model

            num_completion_tokens = (~fix_mask).sum(dim=1)  # [B] — constant across iterations
            step = 0
            while True:
                mask_index = (x == mask_token_id)
                if not mask_index.any():
                    break  # All tokens unmasked — generation finished

                # Forward pass
                outputs = self(
                    input_ids=x,
                    attention_mask=gen_attention_mask,
                    is_causal=False,
                    output_hidden_states=need_hidden_states,
                )
                logits = outputs.logits
                hidden_states = outputs.hidden_states[-1] if need_hidden_states else None

                # CRITICAL: Shift logits to predict the next token, aligning with training
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                mask_logits = logits[mask_index]

                # Synthetic timestep for remasker conditioning: fraction of tokens still masked
                t_synthetic = mask_index.sum(dim=1).float() / num_completion_tokens.float().clamp(min=1)  # [B]

                # 1. Sample x_0 for all masked positions
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg="origin"
                )

                # Build full x0 predictions
                x0_full = x.clone()
                x0_full[mask_index] = x0

                # 2. Compute backbone confidence for all positions
                probs_full = torch.softmax(logits.float(), dim=-1)
                confidence_full = torch.gather(probs_full, -1, x0_full.unsqueeze(-1)).squeeze(-1)
                confidence_full[fix_mask] = 1.0

                # 3. Compute remasking logits
                remasking_logits = get_remasking_logits(
                    batch_size=x.size(0),
                    seq_len=x.size(1),
                    candidate_mask=~fix_mask,
                    source=remasking_cfg.remasking_logits_source,
                    device=x.device,
                    dtype=logits.dtype,
                    temperature=remasking_cfg.remasking_temperature,
                    x_0=x0_full,
                    hidden_states=hidden_states,
                    remasker_model=remasker_model,
                    attention_mask=gen_attention_mask.float() if gen_attention_mask is not None else None,
                    timestep=t_synthetic,
                    confidence=confidence_full,
                    x_t=x,
                )

                # 4. Apply sigmoid then threshold
                remasking_probs = torch.sigmoid(remasking_logits)
                above_threshold = (remasking_probs > remasking_cfg.remasking_threshold) & (~fix_mask)

                # 5. Enforce minimum increase in unmasked count per step (safety floor)
                # previously_unmasked: completion positions that were already unmasked before this step
                # after this step, unmasked count = above_threshold.sum()
                # increase = above_threshold.sum() - previously_unmasked
                # we need: increase >= remasking_min_unmask_tokens
                # i.e.: above_threshold.sum() >= previously_unmasked + remasking_min_unmask_tokens
                if remasking_cfg.remasking_min_unmask_tokens > 0:
                    previously_unmasked = (~mask_index & ~fix_mask).sum(dim=1)  # [B]
                    num_above = above_threshold.sum(dim=1)  # [B]
                    num_masked = mask_index.sum(dim=1)  # [B]
                    required_total = (previously_unmasked + remasking_cfg.remasking_min_unmask_tokens)
                    # Can't require more than total completion tokens
                    required_total = torch.minimum(required_total, num_completion_tokens)

                    needs_topup = num_above < required_total  # [B]
                    if needs_topup.any():
                        # Top-up from completion positions not already selected (rank by sigmoid probs)
                        topup_logits = remasking_probs.clone()
                        topup_logits[above_threshold | fix_mask] = float('-inf')
                        max_deficit = int((required_total - num_above).clamp(min=0).max().item())
                        if max_deficit > 0:
                            _, topup_indices = torch.topk(topup_logits, max_deficit, dim=1)
                            deficit_per_sample = (required_total - num_above).clamp(min=0)
                            valid_topup = torch.arange(max_deficit, device=x.device).unsqueeze(0) < deficit_per_sample.unsqueeze(1)
                            valid_topup = valid_topup & needs_topup.unsqueeze(1)
                            batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topup_indices)
                            above_threshold[batch_idx[valid_topup], topup_indices[valid_topup]] = True

                # 6. Apply: mask all completion positions, then unmask those above threshold
                x[~fix_mask] = mask_token_id
                x[above_threshold] = x0_full[above_threshold]

                if histories is not None:
                    histories.append(x.clone())
                step += 1

        # ── All other algorithms: fixed-step for-loop ──
        else:
            timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

            for i in range(steps):
                mask_index = (x == mask_token_id)
                if not mask_index.any(): # Stop if no tokens are masked
                    break

                # is_causal=False is crucial for bidirectional attention
                # Output hidden states when using remasking with model source
                need_hidden_states = (
                    alg == "remasking" and 
                    generation_config.remasking_config.remasking_logits_source == "model"
                )
                outputs = self(
                    input_ids=x, 
                    attention_mask=gen_attention_mask, 
                    is_causal=False,
                    output_hidden_states=need_hidden_states,
                )
                logits = outputs.logits
                hidden_states = outputs.hidden_states[-1] if need_hidden_states else None

                # CRITICAL: Shift logits to predict the next token, aligning with training
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                mask_logits = logits[mask_index]
                t = timesteps[i]
                s = timesteps[i + 1]

                if alg == "origin":
                    p_transfer = 1 - s / t if i < steps - 1 else 1
                    x0 = torch.full_like(x[mask_index], fill_value=mask_token_id, device=self.device, dtype=torch.long)
                    transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    _, sampled_tokens = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)
                    x0[transfer_index_t_s] = sampled_tokens
                    x[mask_index] = x0

                elif alg == "p2":
                    # Use sample_tokens to obtain confidence and candidate tokens for the whole sequence
                    # kappa_t: fraction of tokens to keep unmasked (can be replaced with custom schedule)
                    kappa_t = (i + 1) / steps

                    # Compute confidence and sampled tokens for the entire sequence:
                    #   conf_full: [B, L], confidence of the sampled token at each position
                    #   x0_full:  [B, L], sampled token IDs for each position
                    conf_full, x0_full = sample_tokens(
                        logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg
                    )

                    # Construct full_conf matrix and mask out fixed positions
                    # Only positions in (~fix_mask) are candidates for masking/unmasking
                    full_conf = conf_full.clone()
                    full_conf[fix_mask] = float("inf")
                    # Prevent NaNs or extreme values from interfering
                    full_conf = torch.where(
                        torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf"))
                    )

                    # Calculate how many positions to re-mask per sample
                    # = number of variable positions * (1 - kappa_t)
                    num_positions = (~fix_mask).sum(dim=1)  # [B]
                    num_to_mask = (num_positions.float() * (1.0 - kappa_t)).floor().to(torch.long)
                    # Boundaries: at least 0, at most total number of variable positions
                    num_to_mask = num_to_mask.clamp_min(0)
                    num_to_mask = torch.minimum(num_to_mask, num_positions)

                    # Select the lowest-confidence positions within (~fix_mask) for re-masking
                    sorted_idx = torch.argsort(full_conf, dim=1, descending=False)  # [B, L]
                    max_k = int(num_to_mask.max().item())
                    if max_k > 0:
                        topk_idx = sorted_idx[:, :max_k]  # [B, max_k]
                        row_mask = torch.arange(max_k, device=x.device).unsqueeze(0) < num_to_mask.unsqueeze(1)  # [B, max_k]

                        to_mask = torch.zeros_like(x, dtype=torch.bool)
                        batch_arange = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topk_idx)  # [B, max_k]
                        valid_batch = batch_arange[row_mask]  # [sum_k]
                        valid_col   = topk_idx[row_mask]      # [sum_k]
                        to_mask[valid_batch, valid_col] = True
                    else:
                        to_mask = torch.zeros_like(x, dtype=torch.bool)

                    # Apply re-masking: set selected positions back to mask_token_id
                    x[to_mask] = mask_token_id

                    # For positions that started as mask and were not re-masked, unmask them with sampled tokens
                    keep_unmask = mask_index & (~to_mask)
                    x[keep_unmask] = x0_full[keep_unmask]
                    


                elif alg in ["maskgit_plus", "entropy", "topk_margin"]:
                    # Confidence-based sampling (maskgit, entropy, etc.)
                    
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)
                    confidence = confidence.to(mask_logits.dtype)

                    # Calculate number of mask tokens per sample
                    num_mask_tokens_per_sample = mask_index.sum(dim=1)  # [batch_size]
                    
                    # Calculate transfer tokens per sample
                    if i < steps - 1:
                        number_transfer_tokens_per_sample = (num_mask_tokens_per_sample.float() * (1 - s / t)).long()
                    else:
                        number_transfer_tokens_per_sample = num_mask_tokens_per_sample
                    
                    # Build full confidence matrix
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    
                    # Get maximum transfer tokens for efficient batching
                    max_transfer_tokens = number_transfer_tokens_per_sample.max().item()
                    
                    if max_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            # Use topk for each sample
                            _, all_transfer_indices = torch.topk(full_confidence, max_transfer_tokens, dim=1)  # [batch_size, max_transfer_tokens]
                        else:
                            # Robust vectorized sampling via Gumbel-TopK (no replacement)
                            # Handles rows with fewer valid positions than requested and rows with no valid positions
                            # full_confidence has -inf for invalid positions; keep them -inf so they won't be selected
                            scaled_logits = full_confidence / alg_temp
                            # Uniform in (0,1) to avoid log(0)
                            uniform = torch.rand_like(scaled_logits).clamp_(min=1e-20, max=1 - 1e-20)
                            gumbel_noise = -torch.log(-torch.log(uniform))
                            scores = scaled_logits + gumbel_noise
                            _, all_transfer_indices = torch.topk(scores, max_transfer_tokens, dim=1)  # [batch_size, max_transfer_tokens]
                        
                        # Create mask for valid transfers (handle variable number of transfers per sample)
                        batch_size = x.size(0)
                        valid_mask = torch.arange(max_transfer_tokens, device=x.device).unsqueeze(0) < number_transfer_tokens_per_sample.unsqueeze(1)  # [batch_size, max_transfer_tokens]
                        
                        # Get valid transfer indices and corresponding batch indices
                        valid_transfer_indices = all_transfer_indices[valid_mask]  # [total_valid_transfers]
                        valid_batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(all_transfer_indices)[valid_mask]  # [total_valid_transfers]
                        
                        # Prepare the transfer data
                        x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        
                        # Batch update using advanced indexing
                        x[valid_batch_indices, valid_transfer_indices] = x_[valid_batch_indices, valid_transfer_indices]
                
                elif alg == "remasking":
                    # Remasking algorithm with configurable alpha schedule
                    remasking_cfg = generation_config.remasking_config
                    
                    # Check if remasking could ever be active (t_on > t_off means there's a remasking window)
                    remasking_ever_active = abs(remasking_cfg.remasking_t_on - remasking_cfg.remasking_t_off) >= 1e-6
                    
                    # Check if remasking is active at current timestep
                    remasking_active = remasking_ever_active and is_remasking_active(
                        t=t.item(),
                        t_on=remasking_cfg.remasking_t_on,
                        t_off=remasking_cfg.remasking_t_off
                    )
                    
                    # Only compute x0_full when remasking is active (needed for remasking logic)
                    # This avoids consuming random numbers when we just want to match the base algorithm
                    x0_full = None
                    remasker_model = None
                    alpha = None
                    
                    if remasking_active:
                        # Predict x_0 for all masked positions
                        confidence, x0 = sample_tokens(
                            mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg="origin"
                        )
                        
                        # Build the full x0 predictions (needed for model-based remasking)
                        x0_full = x.clone()
                        x0_full[mask_index] = x0
                        
                        # Compute confidence for ALL positions (p2-style) for confidence-conditioned remasker
                        # confidence_full: probability backbone assigns to the token at each position in x0_full
                        probs_full = torch.softmax(logits.float(), dim=-1)
                        confidence_full = torch.gather(probs_full, -1, x0_full.unsqueeze(-1)).squeeze(-1)
                        # Prompt positions (fix_mask): set to 1.0 (ground truth, not in backbone's prediction scope)
                        confidence_full[fix_mask] = 1.0
                        
                        # Load remasker model if using model-based remasking (lazy loading)
                        if remasking_cfg.remasking_logits_source == "model":
                            if remasking_cfg._remasker_model is None:
                                if remasking_cfg.remasker_checkpoint_path is None:
                                    raise ValueError(
                                        "remasker_checkpoint_path must be set in remasking_config when "
                                        "remasking_logits_source='model'"
                                    )
                                remasking_cfg._remasker_model = load_remasker_model(
                                    remasking_cfg.remasker_checkpoint_path,
                                    device=str(self.device),
                                )
                                remasking_cfg._remasker_model.to(self.device)
                            remasker_model = remasking_cfg._remasker_model
                        
                        # Compute alpha (ratio of tokens to keep unmasked) based on schedule
                        # Use s (target timestep) instead of t to match p2's kappa_t = (i+1)/steps
                        alpha = compute_alpha(
                            t=s.item(),
                            schedule=remasking_cfg.schedule,
                            t_on=remasking_cfg.remasking_t_on,
                            t_off=remasking_cfg.remasking_t_off,
                            alpha_on=remasking_cfg.remasking_alpha_on,
                            eps=eps
                        )
                        
                        # Calculate number of completion tokens and how many to unmask
                        num_completion_tokens = (~fix_mask).sum(dim=1)  # [B]
                        num_to_unmask = (num_completion_tokens.float() * alpha).floor().long()  # [B]
                        num_to_unmask = num_to_unmask.clamp(min=0)
                        num_to_unmask = torch.minimum(num_to_unmask, num_completion_tokens)
                        
                        # Use Gumbel sampling to select which tokens to unmask
                        # Candidates are completion positions only (~fix_mask)
                        remasking_logits = get_remasking_logits(
                            batch_size=x.size(0),
                            seq_len=x.size(1),
                            candidate_mask=~fix_mask,
                            source=remasking_cfg.remasking_logits_source,
                            device=x.device,
                            dtype=logits.dtype,
                            temperature=remasking_cfg.remasking_temperature,
                            # Additional parameters for model-based remasking
                            x_0=x0_full,
                            hidden_states=hidden_states,
                            remasker_model=remasker_model,
                            attention_mask=gen_attention_mask.float() if gen_attention_mask is not None else None,
                            timestep=t.expand(x.size(0)),  # Pass current timestep for time conditioning
                            confidence=confidence_full,  # Pass backbone confidence for confidence conditioning
                            x_t=x,  # Pass current masked sequence for x_t cross-attention conditioning
                        )
                        
                        # Sample which positions to unmask using Gumbel trick
                        unmask_selection = sample_indices_gumbel(remasking_logits, num_to_unmask)
                        
                        # Apply: unmask selected positions, mask everything else in completion
                        # First, set all completion positions to mask
                        x[~fix_mask] = mask_token_id
                        # Then unmask the selected positions
                        x[unmask_selection] = x0_full[unmask_selection]
                    
                    else:
                        # No remasking: use the non_remasking_sampling_algorithm
                        non_remasking_alg = remasking_cfg.non_remasking_sampling_algorithm
                        
                        # Calculate transfer ratio
                        # Only use simple (1-s/t) formula when:
                        #   - schedule is "linear" AND t_on == t_off (no remasking window)
                        # This ensures behavior matches the base algorithm exactly in that case.
                        # For loop schedule (or any actual remasking window), use alpha-based
                        # formula to respect the intended schedule.
                        if i < steps - 1:
                            use_simple_ratio = (
                                remasking_cfg.schedule == "linear" and 
                                not remasking_ever_active
                            )
                            if use_simple_ratio:
                                # Linear schedule with no remasking window: match original algorithm
                                # Note: p2 uses a different formula, handled in its own block
                                transfer_ratio = 1.0 - (s.item() / t.item())
                            else:
                                # Loop schedule or actual remasking window: use alpha-based transfer ratio
                                # Compute alpha for current timestep t
                                alpha_t = compute_alpha(
                                    t=t.item(),
                                    schedule=remasking_cfg.schedule,
                                    t_on=remasking_cfg.remasking_t_on,
                                    t_off=remasking_cfg.remasking_t_off,
                                    alpha_on=remasking_cfg.remasking_alpha_on,
                                    eps=eps
                                )
                                alpha_s = compute_alpha(
                                    t=s.item(),
                                    schedule=remasking_cfg.schedule,
                                    t_on=remasking_cfg.remasking_t_on,
                                    t_off=remasking_cfg.remasking_t_off,
                                    alpha_on=remasking_cfg.remasking_alpha_on,
                                    eps=eps
                                )
                                if alpha_t < 1.0:
                                    transfer_ratio = (alpha_s - alpha_t) / (1.0 - alpha_t)
                                else:
                                    transfer_ratio = 1.0
                        else:
                            transfer_ratio = 1.0
                        
                        if non_remasking_alg == "origin":
                            # Origin algorithm: probability-based transfer
                            p_transfer = transfer_ratio
                            x0_masked = torch.full_like(x[mask_index], fill_value=mask_token_id, device=self.device, dtype=torch.long)
                            transfer_index_t_s = torch.rand(*x0_masked.shape, device=self.device) < p_transfer
                            _, sampled_tokens = sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, alg="origin")
                            x0_masked[transfer_index_t_s] = sampled_tokens
                            x[mask_index] = x0_masked
                        
                        elif non_remasking_alg in ["maskgit_plus", "entropy", "topk_margin"]:
                            # Confidence-based sampling
                            confidence, x0_sampled = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=non_remasking_alg)
                            confidence = confidence.to(mask_logits.dtype)
                            
                            # Calculate number of mask tokens per sample
                            num_mask_tokens_per_sample = mask_index.sum(dim=1)  # [batch_size]
                            
                            # Calculate transfer tokens per sample using transfer ratio
                            number_transfer_tokens_per_sample = (num_mask_tokens_per_sample.float() * transfer_ratio).long()
                            
                            # Build full confidence matrix
                            full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                            full_confidence[mask_index] = confidence
                            
                            # Get maximum transfer tokens for efficient batching
                            max_transfer_tokens = number_transfer_tokens_per_sample.max().item()
                            
                            if max_transfer_tokens > 0:
                                if alg_temp is None or alg_temp == 0:
                                    # Use topk for each sample
                                    _, all_transfer_indices = torch.topk(full_confidence, max_transfer_tokens, dim=1)
                                else:
                                    # Gumbel-TopK sampling
                                    scaled_logits = full_confidence / alg_temp
                                    uniform = torch.rand_like(scaled_logits).clamp_(min=1e-20, max=1 - 1e-20)
                                    gumbel_noise = -torch.log(-torch.log(uniform))
                                    scores = scaled_logits + gumbel_noise
                                    _, all_transfer_indices = torch.topk(scores, max_transfer_tokens, dim=1)
                                
                                # Create mask for valid transfers
                                batch_size = x.size(0)
                                valid_mask = torch.arange(max_transfer_tokens, device=x.device).unsqueeze(0) < number_transfer_tokens_per_sample.unsqueeze(1)
                                
                                # Get valid transfer indices
                                valid_transfer_indices = all_transfer_indices[valid_mask]
                                valid_batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(all_transfer_indices)[valid_mask]
                                
                                # Prepare transfer data
                                x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                                x_[mask_index] = x0_sampled.clone()
                                
                                # Apply transfer
                                x[valid_batch_indices, valid_transfer_indices] = x_[valid_batch_indices, valid_transfer_indices]
                        
                        elif non_remasking_alg == "p2":
                            # Exact copy of p2 algorithm, only kappa_t is different
                            # kappa_t = fraction of variable positions that should be UNMASKED (cumulative)
                            # Use original p2 kappa_t formula when no remasking window is active
                            if remasking_cfg.schedule == "linear" and not remasking_ever_active:
                                kappa_t = (i + 1) / steps
                            else:
                                # When remasking is active, kappa_t = alpha_s (target unmasked fraction)
                                kappa_t = compute_alpha(
                                    t=s.item(),
                                    schedule=remasking_cfg.schedule,
                                    t_on=remasking_cfg.remasking_t_on,
                                    t_off=remasking_cfg.remasking_t_off,
                                    alpha_on=remasking_cfg.remasking_alpha_on,
                                    eps=eps
                                )

                            # Compute confidence and sampled tokens for the entire sequence:
                            #   conf_full: [B, L], confidence of the sampled token at each position
                            #   x0_full:  [B, L], sampled token IDs for each position
                            conf_full, x0_full = sample_tokens(
                                logits, temperature=temperature, top_p=top_p, top_k=top_k, alg="p2"
                            )

                            # Construct full_conf matrix and mask out fixed positions
                            # Only positions in (~fix_mask) are candidates for masking/unmasking
                            full_conf = conf_full.clone()
                            full_conf[fix_mask] = float("inf")
                            # Prevent NaNs or extreme values from interfering
                            full_conf = torch.where(
                                torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf"))
                            )

                            # Calculate how many positions to re-mask per sample
                            # = number of variable positions * (1 - kappa_t)
                            num_positions = (~fix_mask).sum(dim=1)  # [B]
                            num_to_mask = (num_positions.float() * (1.0 - kappa_t)).floor().to(torch.long)
                            # Boundaries: at least 0, at most total number of variable positions
                            num_to_mask = num_to_mask.clamp_min(0)
                            num_to_mask = torch.minimum(num_to_mask, num_positions)

                            # Select the lowest-confidence positions within (~fix_mask) for re-masking
                            sorted_idx = torch.argsort(full_conf, dim=1, descending=False)  # [B, L]
                            max_k = int(num_to_mask.max().item())
                            if max_k > 0:
                                topk_idx = sorted_idx[:, :max_k]  # [B, max_k]
                                row_mask = torch.arange(max_k, device=x.device).unsqueeze(0) < num_to_mask.unsqueeze(1)  # [B, max_k]

                                to_mask = torch.zeros_like(x, dtype=torch.bool)
                                batch_arange = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topk_idx)  # [B, max_k]
                                valid_batch = batch_arange[row_mask]  # [sum_k]
                                valid_col   = topk_idx[row_mask]      # [sum_k]
                                to_mask[valid_batch, valid_col] = True
                            else:
                                to_mask = torch.zeros_like(x, dtype=torch.bool)

                            # Apply re-masking: set selected positions back to mask_token_id
                            x[to_mask] = mask_token_id

                            # For positions that started as mask and were not re-masked, unmask them with sampled tokens
                            keep_unmask = mask_index & (~to_mask)
                            x[keep_unmask] = x0_full[keep_unmask]
                        
                        else:
                            raise NotImplementedError(f"Non-remasking algorithm '{non_remasking_alg}' not implemented.")

                else:
                    raise NotImplementedError(f"Algorithm {alg} not implemented.")

                if histories is not None:
                    histories.append(x.clone())

        # Final cleanup: unmask any remaining mask tokens after all steps
        remaining_mask = (x == mask_token_id)
        if remaining_mask.any():
            # Forward pass to get logits for remaining masks
            outputs = self(input_ids=x, attention_mask=gen_attention_mask, is_causal=False)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Sample tokens for remaining masks (use greedy to ensure clean output)
            mask_logits = logits[remaining_mask]
            _, sampled_tokens = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg="origin")
            x[remaining_mask] = sampled_tokens
            
            if histories is not None:
                histories.append(x.clone())

        if generation_config.return_dict_in_generate:
            return MDMModelOutput(sequences=x, history=histories)
        else:
            return x