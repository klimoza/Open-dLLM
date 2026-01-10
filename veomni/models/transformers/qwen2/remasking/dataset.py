# veomni/models/transformers/qwen2/remasking/dataset.py

"""Dataset and data loading utilities for remasker training."""

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset

from .config import RemaskerTrainingConfig
from .corruption import corrupt_completion


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

