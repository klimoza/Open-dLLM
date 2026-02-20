# veomni/models/transformers/qwen2/remasking/training/evaluate.py

"""Evaluation loop for the remasker."""

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config import RemaskerTrainingConfig


@torch.no_grad()
def evaluate(
    model,
    backbone,
    dataloader,
    config: RemaskerTrainingConfig,
) -> Dict[str, float]:
    """
    Evaluate the remasker model.
    
    Args:
        model: The remasker model
        backbone: The backbone model
        dataloader: Evaluation dataloader
        config: Training configuration
    
    Returns:
        Dictionary with eval_loss and eval_accuracy
    """
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
        
        # Get hidden states from backbone if needed
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=config.fp16):
            if config.use_hidden_states:
                backbone_outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = backbone_outputs.hidden_states[-1]
            else:
                hidden_states = None
            
            # Create timestep tensor if time conditioning is enabled
            # Use t=0.5 as a default mid-range timestep for corruption-based eval
            batch_size = input_ids.shape[0]
            timestep_tensor = (
                torch.full((batch_size,), 0.5, device=config.device)
                if config.use_time_conditioning else None
            )
            
            # Create confidence tensor if confidence conditioning is enabled
            # Use 1.0 (fully confident) as default for corruption-based eval
            confidence_tensor = (
                torch.ones(input_ids.shape, device=config.device)
                if config.use_confidence_conditioning else None
            )
            
            # Forward pass through remasker
            logits = model(
                x_0=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask.float(),
                timestep=timestep_tensor,
                confidence=confidence_tensor,
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
