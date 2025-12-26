#!/usr/bin/env python3
# scripts/train_remasker.py

"""
Training script for the Remasker model.

The remasker learns to identify corrupted tokens in a sequence.
It takes x_0 (predicted tokens) and hidden_states from the backbone,
and outputs binary logits indicating token correctness.

Example usage:
    python scripts/train_remasker.py \
        --backbone_path fredzzp/open-dcoder-0.5B \
        --dataset_path nvidia/OpenCodeInstruct \
        --checkpoint_name remasker_v1 \
        --num_layers 4 \
        --epochs 3 \
        --lr 1e-4 \
        --wandb_project remasker-training \
        --wandb_run_name remasker-training-open-dcoder-0.5B \
        --use_wandb

Dataset format expected:
    - HuggingFace dataset with 'instruction' and 'response' columns (OpenCodeInstruct)
    - Or 'prompt' and 'completion' columns
    - Or 'messages' column with list of dicts
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from veomni.models.transformers.qwen2.remasking_utils import compute_alpha


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
    
    # Other
    seed: int = 42
    num_workers: int = 4
    save_every_n_steps: int = 1000  # Save checkpoint every N optimization steps
    eval_ratio: float = 0.05  # Fraction of data for evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def corrupt_completion(
    completion_ids: torch.Tensor,
    vocab_size: int,
    random_ratio: float,
    repeat_ratio: float,
    special_token_ids: Optional[List[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor]:
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


class RemaskerDataset(Dataset):
    """Dataset for training the remasker."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        backbone_model,
        config: RemaskerTrainingConfig,
        is_eval: bool = False,
        mask_token_id: Optional[int] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.backbone_model = backbone_model
        self.config = config
        self.is_eval = is_eval
        self.vocab_size = tokenizer.vocab_size
        self.mask_token_id = mask_token_id  # For denoising training mode
        
        # Get special token ids to exclude from corruption
        self.special_token_ids = []
        if tokenizer.pad_token_id is not None:
            self.special_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            self.special_token_ids.append(tokenizer.eos_token_id)
        if tokenizer.bos_token_id is not None:
            self.special_token_ids.append(tokenizer.bos_token_id)
    
    def __len__(self):
        return len(self.data)
    
    def _extract_prompt_completion(self, item: Dict[str, Any]) -> tuple[str, str]:
        """Extract prompt and completion from dataset item."""
        # Try different formats
        if "instruction" in item and "response" in item:
            return item["instruction"], item["response"]
        elif "prompt" in item and "completion" in item:
            return item["prompt"], item["completion"]
        elif "messages" in item:
            messages = item["messages"]
            if len(messages) >= 2:
                # Assume first is user, second is assistant
                prompt = messages[0].get("content", "")
                completion = messages[1].get("content", "")
                return prompt, completion
        elif "input" in item and "output" in item:
            return item["input"], item["output"]
        elif "question" in item and "answer" in item:
            return item["question"], item["answer"]
        
        raise ValueError(f"Unknown dataset format: {list(item.keys())}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        try:
            prompt, completion = self._extract_prompt_completion(item)
        except ValueError:
            # Fallback: use entire text as completion
            if "text" in item:
                prompt = ""
                completion = item["text"]
            else:
                raise
        
        # Tokenize prompt and completion separately
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
        
        # Truncate if needed
        max_prompt_len = self.config.max_seq_length // 2
        max_completion_len = self.config.max_seq_length - len(prompt_tokens[:max_prompt_len])
        
        prompt_tokens = prompt_tokens[:max_prompt_len]
        completion_tokens = completion_tokens[:max_completion_len]
        
        if len(completion_tokens) == 0:
            completion_tokens = [self.tokenizer.eos_token_id or 0]
        
        # Create tensors
        prompt_ids = torch.tensor(prompt_tokens, dtype=torch.long)
        completion_ids = torch.tensor(completion_tokens, dtype=torch.long)
        
        if self.config.use_denoising_training:
            # Denoising mode: return ground truth tokens
            # Masking, denoising, and augmentation will be done in train_epoch
            full_ids = torch.cat([prompt_ids, completion_ids])
            
            # Ground truth labels (all correct for now, will be recomputed after augmentation)
            full_labels = torch.ones(len(full_ids), dtype=torch.float)
            
            # Create mask for which positions to compute loss on (only completion)
            loss_mask = torch.zeros(len(full_ids), dtype=torch.bool)
            loss_mask[len(prompt_tokens):] = True
            
            return {
                "input_ids": full_ids,  # Ground truth sequence
                "labels": full_labels,  # Will be recomputed after denoising + augmentation
                "loss_mask": loss_mask,
                "prompt_len": len(prompt_tokens),
                "ground_truth_ids": full_ids.clone(),  # Keep a copy for label computation
            }
        else:
            # Original corruption-based training mode
            # Corrupt completion
            corrupted_completion, corruption_mask = corrupt_completion(
                completion_ids,
                self.vocab_size,
                self.config.random_corruption_ratio,
                self.config.repeat_corruption_ratio,
                self.special_token_ids,
            )
            
            # Combine prompt + corrupted completion
            full_ids = torch.cat([prompt_ids, corrupted_completion])
            
            # Create labels (1 = correct, 0 = corrupted)
            # Prompt tokens are always "correct" (we don't predict on them)
            prompt_labels = torch.ones(len(prompt_tokens), dtype=torch.float)
            completion_labels = (~corruption_mask).float()  # 1 if not corrupted
            full_labels = torch.cat([prompt_labels, completion_labels])
            
            # Create mask for which positions to compute loss on (only completion)
            loss_mask = torch.zeros(len(full_ids), dtype=torch.bool)
            loss_mask[len(prompt_tokens):] = True
            
            return {
                "input_ids": full_ids,
                "labels": full_labels,
                "loss_mask": loss_mask,
                "prompt_len": len(prompt_tokens),
            }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate function for batching."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    
    input_ids = []
    labels = []
    loss_masks = []
    attention_masks = []
    prompt_lens = []
    ground_truth_ids = []
    has_ground_truth = "ground_truth_ids" in batch[0]
    
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        
        # Pad sequences
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(F.pad(item["labels"], (0, pad_len), value=1.0))  # Pad labels with 1 (correct)
        loss_masks.append(F.pad(item["loss_mask"], (0, pad_len), value=False))
        
        if has_ground_truth:
            ground_truth_ids.append(F.pad(item["ground_truth_ids"], (0, pad_len), value=pad_token_id))
        
        # Create attention mask
        attn_mask = torch.zeros(max_len, dtype=torch.bool)
        attn_mask[:seq_len] = True
        attention_masks.append(attn_mask)
        
        prompt_lens.append(item["prompt_len"])
    
    result = {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "loss_mask": torch.stack(loss_masks),
        "attention_mask": torch.stack(attention_masks),
        "prompt_lens": torch.tensor(prompt_lens),
    }
    
    if has_ground_truth:
        result["ground_truth_ids"] = torch.stack(ground_truth_ids)
    
    return result


def load_data(config: RemaskerTrainingConfig) -> tuple[List[Dict], List[Dict]]:
    """Load and split dataset."""
    print(f"Loading dataset from {config.dataset_path}...")
    
    if config.dataset_name:
        dataset = load_dataset(config.dataset_path, config.dataset_name, split=config.dataset_split)
    else:
        dataset = load_dataset(config.dataset_path, split=config.dataset_split)
    
    # Convert to list
    data = list(dataset)
    
    # Limit samples if specified
    if config.max_samples is not None:
        data = data[:config.max_samples]
    
    # Shuffle and split
    random.shuffle(data)
    split_idx = int(len(data) * (1 - config.eval_ratio))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")
    return train_data, eval_data


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute classification metrics for tokens that propagate loss.
    
    Args:
        logits: Model output logits [B, L]
        labels: Ground truth labels (1=correct, 0=corrupted) [B, L]
        loss_mask: Mask for positions to compute metrics on [B, L]
    
    Returns:
        Dictionary with positive_ratio, pred_positive_ratio, pred_avg_prob, precision, recall
    """
    # Get probabilities and predictions
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    # Only consider masked positions
    masked_probs = probs[loss_mask]
    masked_preds = preds[loss_mask]
    masked_labels = labels[loss_mask]
    
    total_tokens = masked_labels.numel()
    if total_tokens == 0:
        return {"positive_ratio": 0.0, "pred_positive_ratio": 0.0, "pred_avg_prob": 0.0, "precision": 0.0, "recall": 0.0}
    
    # Positive class ratio in ground truth (ratio of correct/non-corrupted tokens)
    positive_ratio = masked_labels.sum().item() / total_tokens
    
    # Positive class ratio in predictions (ratio of tokens predicted as correct)
    pred_positive_ratio = masked_preds.sum().item() / total_tokens
    
    # Average probability output by classifier
    pred_avg_prob = masked_probs.mean().item()
    
    # True positives: predicted correct AND actually correct
    tp = ((masked_preds == 1) & (masked_labels == 1)).sum().item()
    # False positives: predicted correct BUT actually corrupted
    fp = ((masked_preds == 1) & (masked_labels == 0)).sum().item()
    # False negatives: predicted corrupted BUT actually correct
    fn = ((masked_preds == 0) & (masked_labels == 1)).sum().item()
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "positive_ratio": positive_ratio,
        "pred_positive_ratio": pred_positive_ratio,
        "pred_avg_prob": pred_avg_prob,
        "precision": precision,
        "recall": recall,
    }


def train_epoch(
    model,
    backbone,
    dataloader,
    optimizer,
    scheduler,
    config: RemaskerTrainingConfig,
    epoch: int,
    global_step: int,
    save_path: str,
    mask_token_id: Optional[int] = None,
    tokenizer = None,
) -> tuple[float, int]:
    """Train for one epoch."""
    model.train()
    backbone.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Accumulators for metrics over gradient accumulation steps
    accum_metrics = {"positive_ratio": 0.0, "pred_positive_ratio": 0.0, "pred_avg_prob": 0.0, "precision": 0.0, "recall": 0.0, "pos_weight": 0.0}
    accum_grad_norm = 0.0
    accum_count = 0
    
    # Get special token ids for corruption (denoising mode)
    special_token_ids = []
    if tokenizer is not None:
        if tokenizer.pad_token_id is not None:
            special_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            special_token_ids.append(tokenizer.eos_token_id)
        if tokenizer.bos_token_id is not None:
            special_token_ids.append(tokenizer.bos_token_id)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        loss_mask = batch["loss_mask"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        prompt_lens = batch["prompt_lens"].to(config.device)
        
        if config.use_denoising_training:
            # Denoising training mode: simulate inference process
            ground_truth_ids = batch["ground_truth_ids"].to(config.device)
            batch_size, seq_len = input_ids.shape
            
            # Sample timestep t uniformly from [t_off, t_on]
            t = random.uniform(config.denoising_t_off, config.denoising_t_on)
            
            # Compute alpha (fraction of tokens to keep unmasked)
            alpha = compute_alpha(
                t=t,
                schedule="linear",
                t_on=config.denoising_t_on,
                t_off=config.denoising_t_off,
                alpha_on=0.9,  # Not used for linear schedule
                eps=1e-3
            )
            
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
                    perm = torch.randperm(completion_len, device=config.device)
                    mask_indices = perm[:num_to_mask]
                    
                    # Apply masking
                    x_t[b, prompt_len + mask_indices] = mask_token_id
                    mask_positions[b, prompt_len + mask_indices] = True
            
            # Get hidden states and logits from backbone on x_t
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                    backbone_outputs = backbone(
                        input_ids=x_t,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        is_causal=False,  # Bidirectional attention for MDM
                    )
                    hidden_states = backbone_outputs.hidden_states[-1]
                    backbone_logits = backbone_outputs.logits
                    
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
                    
                    # Apply augmentations (random/repeat corruption) to completion tokens
                    # We need to do this per-sample since completion lengths vary
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
                            vocab_size=tokenizer.vocab_size if tokenizer else backbone.config.vocab_size,
                            random_ratio=config.random_corruption_ratio,
                            repeat_ratio=config.repeat_corruption_ratio,
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
            
            # Forward pass through remasker with x_0_full and hidden_states from x_t
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                logits = model(
                    x_0=x_0_full,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask.float(),
                )
                
                # Get masked logits and labels
                masked_logits = logits[loss_mask]
                masked_labels = labels[loss_mask]
                
                # Apply label smoothing if enabled
                if config.label_smoothing_alpha > 0:
                    masked_labels = masked_labels * (1 - 2 * config.label_smoothing_alpha) + config.label_smoothing_alpha
                
                # Compute class weights if enabled
                if config.use_class_reweighting and masked_labels.numel() > 0:
                    num_positive = masked_labels.sum()
                    num_negative = masked_labels.numel() - num_positive
                    
                    if num_positive > 0 and num_negative > 0:
                        pos_weight = num_negative / num_positive
                    else:
                        pos_weight = torch.tensor(1.0, device=config.device)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        pos_weight=pos_weight,
                        reduction="mean",
                    )
                    batch_pos_weight = pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        reduction="mean",
                    )
                    batch_pos_weight = 1.0
                loss = loss / config.gradient_accumulation_steps
        
        else:
            # Original corruption-based training mode
            # Get hidden states from backbone (no gradient)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                    backbone_outputs = backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = backbone_outputs.hidden_states[-1]  # Final layer
            
            # Forward pass through remasker
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
                logits = model(
                    x_0=input_ids,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask.float(),
                )
                
                # Get masked logits and labels
                masked_logits = logits[loss_mask]
                masked_labels = labels[loss_mask]
                
                # Apply label smoothing if enabled: 0 -> alpha, 1 -> 1-alpha
                if config.label_smoothing_alpha > 0:
                    masked_labels = masked_labels * (1 - 2 * config.label_smoothing_alpha) + config.label_smoothing_alpha
                
                # Compute class weights if enabled
                if config.use_class_reweighting and masked_labels.numel() > 0:
                    # Count positive (correct) and negative (corrupted) samples
                    num_positive = masked_labels.sum()
                    num_negative = masked_labels.numel() - num_positive
                    
                    # pos_weight: weight for positive class to balance with negative class
                    # If positive is majority, pos_weight < 1 to down-weight positives
                    # This is equivalent to up-weighting negatives
                    if num_positive > 0 and num_negative > 0:
                        pos_weight = num_negative / num_positive
                    else:
                        pos_weight = torch.tensor(1.0, device=config.device)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        pos_weight=pos_weight,
                        reduction="mean",
                    )
                    batch_pos_weight = pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight
                else:
                    # No reweighting
                    loss = F.binary_cross_entropy_with_logits(
                        masked_logits,
                        masked_labels,
                        reduction="mean",
                    )
                    batch_pos_weight = 1.0
                loss = loss / config.gradient_accumulation_steps
        
        # Compute classification metrics (no grad needed)
        with torch.no_grad():
            batch_metrics = compute_classification_metrics(logits, labels, loss_mask)
            for k in batch_metrics:
                accum_metrics[k] += batch_metrics[k]
            accum_metrics["pos_weight"] += batch_pos_weight
            accum_count += 1
        
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Clip gradients and get the total norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            accum_grad_norm += grad_norm.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Save checkpoint every N steps
            if config.save_every_n_steps > 0 and global_step % config.save_every_n_steps == 0:
                step_save_path = os.path.join(save_path, f"step_{global_step}")
                model.save_pretrained(step_save_path)
                print(f"\nSaved checkpoint to {step_save_path}")
            
            # Log to wandb with accumulated metrics
            if config.use_wandb and WANDB_AVAILABLE:
                avg_metrics = {k: v / accum_count for k, v in accum_metrics.items()}
                wandb.log({
                    "train/loss": loss.item() * config.gradient_accumulation_steps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": accum_grad_norm,
                    "train/positive_ratio": avg_metrics["positive_ratio"],
                    "train/pred_positive_ratio": avg_metrics["pred_positive_ratio"],
                    "train/pred_avg_prob": avg_metrics["pred_avg_prob"],
                    "train/precision": avg_metrics["precision"],
                    "train/recall": avg_metrics["recall"],
                    "train/pos_weight": avg_metrics["pos_weight"],
                    "global_step": global_step,
                })
            
            # Reset accumulators
            accum_metrics = {"positive_ratio": 0.0, "pred_positive_ratio": 0.0, "pred_avg_prob": 0.0, "precision": 0.0, "recall": 0.0, "pos_weight": 0.0}
            accum_grad_norm = 0.0
            accum_count = 0
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
    
    return total_loss / num_batches, global_step


@torch.no_grad()
def evaluate(
    model,
    backbone,
    dataloader,
    config: RemaskerTrainingConfig,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    backbone.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        loss_mask = batch["loss_mask"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        
        # Get hidden states from backbone
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
            backbone_outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = backbone_outputs.hidden_states[-1]
            
            # Forward pass through remasker
            logits = model(
                x_0=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask.float(),
            )
            
            # BCE loss
            loss = F.binary_cross_entropy_with_logits(
                logits[loss_mask],
                labels[loss_mask],
                reduction="mean",
            )
        
        total_loss += loss.item()
        
        # Accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds[loss_mask] == labels[loss_mask]).sum().item()
        total_correct += correct
        total_samples += loss_mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return {"eval_loss": avg_loss, "eval_accuracy": accuracy}


def main(config: RemaskerTrainingConfig):
    """Main training function."""
    set_seed(config.seed)
    
    # Create checkpoint directory (fail if already exists to prevent overwriting)
    save_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)
    if os.path.exists(save_path):
        raise FileExistsError(
            f"Checkpoint directory already exists: {save_path}\n"
            f"Please use a different --checkpoint_name or remove the existing directory."
        )
    os.makedirs(save_path)
    
    # Save config
    config_path = os.path.join(save_path, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)
    
    print(f"Loading backbone from {config.backbone_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_path)
    backbone = AutoModelForCausalLM.from_pretrained(
        config.backbone_path,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
    ).to(config.device)
    backbone.eval()
    
    # Get mask token id for denoising training
    mask_token_id = getattr(backbone.config, 'mask_token_id', None)
    if config.use_denoising_training and mask_token_id is None:
        # Try to get from tokenizer or use a default
        mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        if mask_token_id is None:
            # Use a common convention: vocab_size (out of vocabulary token)
            mask_token_id = backbone.config.vocab_size
            print(f"Warning: No mask_token_id found, using {mask_token_id}")
        else:
            print(f"Using tokenizer mask_token_id: {mask_token_id}")
    elif config.use_denoising_training:
        print(f"Using backbone mask_token_id: {mask_token_id}")
    
    # Get backbone config for remasker
    backbone_config = backbone.config
    
    # Import remasker
    from veomni.models.transformers.qwen2.remasker_model import Remasker, RemaskerConfig
    
    # Create remasker config
    remasker_config = RemaskerConfig(
        num_layers=config.remasker_num_layers,
        hidden_size=config.remasker_hidden_size or backbone_config.hidden_size,
        intermediate_size=config.remasker_intermediate_size or backbone_config.intermediate_size,
        num_attention_heads=config.remasker_num_attention_heads or backbone_config.num_attention_heads,
        num_key_value_heads=config.remasker_num_key_value_heads or backbone_config.num_key_value_heads,
        vocab_size=backbone_config.vocab_size,
        backbone_hidden_size=backbone_config.hidden_size,
    )
    
    print(f"Creating remasker with {config.remasker_num_layers} layers...")
    
    if config.init_from_backbone:
        # Calculate layer offset (default: use last N layers from backbone)
        if config.init_layer_offset < 0:
            backbone_num_layers = backbone_config.num_hidden_layers
            layer_offset = max(0, backbone_num_layers - config.remasker_num_layers)
        else:
            layer_offset = config.init_layer_offset
        
        print(f"Initializing remasker from backbone (layer_offset={layer_offset})...")
        model = Remasker.from_backbone(
            backbone_model=backbone,
            config=remasker_config,
            init_embedding=True,
            init_layers=True,
            layer_offset=layer_offset,
        ).to(config.device)
    else:
        model = Remasker(remasker_config).to(config.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Remasker parameters: {num_params:,} total, {num_trainable:,} trainable")
    
    # Load data
    train_data, eval_data = load_data(config)
    
    # Create datasets
    train_dataset = RemaskerDataset(train_data, tokenizer, backbone, config, is_eval=False, mask_token_id=mask_token_id)
    eval_dataset = RemaskerDataset(eval_data, tokenizer, backbone, config, is_eval=True, mask_token_id=mask_token_id)
    
    # Create dataloaders
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    
    if config.scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )
    else:
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps - warmup_steps,
        )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )
    
    # Initialize wandb
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or config.checkpoint_name,
            config=vars(config),
        )
    
    # Training loop
    global_step = 0
    best_eval_loss = float("inf")
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Save every: {config.save_every_n_steps} steps")
    print(f"Class reweighting: {'enabled' if config.use_class_reweighting else 'disabled'}")
    if config.label_smoothing_alpha > 0:
        print(f"Label smoothing: alpha={config.label_smoothing_alpha} (0->{config.label_smoothing_alpha:.3f}, 1->{1-config.label_smoothing_alpha:.3f})")
    if config.use_denoising_training:
        print(f"Denoising training: t_on={config.denoising_t_on}, t_off={config.denoising_t_off}, temperature={config.denoising_temperature}")
    
    for epoch in range(config.epochs):
        # Train
        train_loss, global_step = train_epoch(
            model, backbone, train_loader, optimizer, scheduler, config, epoch, global_step, save_path,
            mask_token_id=mask_token_id, tokenizer=tokenizer
        )
        print(f"\nEpoch {epoch + 1} - Train loss: {train_loss:.4f}")
        
        # Evaluate
        eval_metrics = evaluate(model, backbone, eval_loader, config)
        print(f"Epoch {epoch + 1} - Eval loss: {eval_metrics['eval_loss']:.4f}, "
              f"Eval accuracy: {eval_metrics['eval_accuracy']:.4f}")
        
        # Log to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                **{f"eval/{k}": v for k, v in eval_metrics.items()},
            })
        
        # Save best model
        if eval_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["eval_loss"]
            best_save_path = os.path.join(save_path, "best")
            model.save_pretrained(best_save_path)
            print(f"Saved best model to {best_save_path}")
    
    # Save final model
    final_save_path = os.path.join(save_path, "final")
    model.save_pretrained(final_save_path)
    print(f"\nTraining complete! Final model saved to {final_save_path}")
    
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Remasker model")
    
    # Model paths
    parser.add_argument("--backbone_path", type=str, default="./models/qwen2-0.5b")
    parser.add_argument("--checkpoint_name", type=str, default="remasker_v1")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    
    # Remasker architecture
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--init_from_backbone", action="store_true", help="Initialize remasker layers from backbone model")
    parser.add_argument("--init_layer_offset", type=int, default=-1, help="Which backbone layer to start copying from (-1 = auto: use last N layers)")
    
    # Corruption settings
    parser.add_argument("--random_corruption_ratio", type=float, default=0.1)
    parser.add_argument("--repeat_corruption_ratio", type=float, default=0.1)
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, default="nvidia/OpenCodeInstruct")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "linear"])
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="remasker-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Class reweighting and label smoothing
    parser.add_argument("--no_class_reweighting", action="store_true", help="Disable class reweighting for imbalanced classes")
    parser.add_argument("--label_smoothing_alpha", type=float, default=0.0, help="Label smoothing: 0->alpha, 1->1-alpha (default: 0.0, no smoothing)")
    
    # Denoising training mode
    parser.add_argument("--use_denoising_training", action="store_true", help="Use denoising-based training that matches inference")
    parser.add_argument("--denoising_t_on", type=float, default=0.1, help="Upper bound for timestep sampling in denoising mode")
    parser.add_argument("--denoising_t_off", type=float, default=0.1, help="Lower bound for timestep sampling in denoising mode")
    parser.add_argument("--denoising_temperature", type=float, default=0.0, help="Temperature for sampling x_0 from logits (0 = greedy)")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every_n_steps", type=int, default=1000)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--no_fp16", action="store_true")
    
    args = parser.parse_args()
    
    config = RemaskerTrainingConfig(
        backbone_path=args.backbone_path,
        checkpoint_name=args.checkpoint_name,
        checkpoint_dir=args.checkpoint_dir,
        remasker_num_layers=args.num_layers,
        remasker_hidden_size=args.hidden_size,
        remasker_intermediate_size=args.intermediate_size,
        remasker_num_attention_heads=args.num_attention_heads,
        remasker_num_key_value_heads=args.num_key_value_heads,
        init_from_backbone=args.init_from_backbone,
        init_layer_offset=args.init_layer_offset,
        random_corruption_ratio=args.random_corruption_ratio,
        repeat_corruption_ratio=args.repeat_corruption_ratio,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        scheduler_type=args.scheduler_type,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        num_workers=args.num_workers,
        use_class_reweighting=not args.no_class_reweighting,
        label_smoothing_alpha=args.label_smoothing_alpha,
        save_every_n_steps=args.save_every_n_steps,
        eval_ratio=args.eval_ratio,
        fp16=not args.no_fp16,
        use_denoising_training=args.use_denoising_training,
        denoising_t_on=args.denoising_t_on,
        denoising_t_off=args.denoising_t_off,
        denoising_temperature=args.denoising_temperature,
    )
    
    main(config)


