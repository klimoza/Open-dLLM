#!/usr/bin/env python3
# scripts/simulate_denoising.py

"""
Simulate single-step denoising for visualization.

This script demonstrates the denoising process used in remasker training:
1. Takes a prompt and completion (or generates from model)
2. Creates x_t by masking some completion tokens
3. Runs the backbone to get predictions (x_0)
4. Compares predictions with ground truth

Example usage:
    # Basic usage with default text
    python scripts/simulate_denoising.py \
        --backbone_path fredzzp/open-dcoder-0.5B

    # Custom text with specific timestep
    python scripts/simulate_denoising.py \
        --backbone_path fredzzp/open-dcoder-0.5B \
        --prompt "def fibonacci(n):" \
        --completion " return fibonacci(n-1) + fibonacci(n-2) if n > 1 else 0<|endoftext|>" \
        --t 0.9 \
        --num_steps 1

    # Use model to generate completion first
    python scripts/simulate_denoising.py \
        --backbone_path fredzzp/open-dcoder-0.5B \
        --prompt "def hello_world():" \
        --generate_completion \
        --t 0.3
"""

import argparse
import sys
from typing import List, Optional, Tuple

import torch

# Add project root to path
sys.path.insert(0, ".")

from transformers import AutoTokenizer, AutoModelForCausalLM


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def compute_alpha(t: float, schedule: str = "linear", t_on: float = 0.55, t_off: float = 0.05, 
                  alpha_on: float = 0.9, eps: float = 1e-3) -> float:
    """Compute alpha (ratio of unmasked tokens) based on schedule type."""
    if schedule == "linear":
        alpha = (1.0 - t) / (1.0 - eps)
        return min(max(alpha, 0.0), 1.0)
    elif schedule == "loop":
        if t >= t_on:
            alpha = alpha_on * (1.0 - t) / (1.0 - t_on)
            return min(max(alpha, 0.0), alpha_on)
        elif t >= t_off:
            return alpha_on
        else:
            alpha = alpha_on + (1.0 - alpha_on) * (t_off - t) / (t_off - eps)
            return min(max(alpha, alpha_on), 1.0)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def create_masked_sequence(
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    mask_token_id: int,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Create a masked sequence x_t by masking (1-alpha) fraction of completion tokens.
    
    Returns:
        x_t: Masked sequence [P + C]
        mask_positions: Boolean mask [P + C], True where completion tokens are masked
        masked_indices: List of indices (within completion) that were masked
    """
    device = completion_ids.device
    prompt_len = prompt_ids.shape[0]
    completion_len = completion_ids.shape[0]
    
    # Number of completion tokens to keep unmasked
    num_to_keep = int(completion_len * alpha)
    num_to_mask = completion_len - num_to_keep
    
    # Randomly select which positions to mask in completion
    perm = torch.randperm(completion_len, device=device)
    mask_indices = perm[:num_to_mask].tolist()
    
    # Create masked completion
    masked_completion = completion_ids.clone()
    for idx in mask_indices:
        masked_completion[idx] = mask_token_id
    
    # Combine prompt + masked completion
    x_t = torch.cat([prompt_ids, masked_completion])
    
    # Create mask indicating which positions are masked (in full sequence)
    mask_positions = torch.zeros(prompt_len + completion_len, dtype=torch.bool, device=device)
    for idx in mask_indices:
        mask_positions[prompt_len + idx] = True
    
    return x_t, mask_positions, sorted(mask_indices)


def multi_step_denoise(
    x_t: torch.Tensor,
    backbone,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    num_steps: int = 4,
    temperature: float = 0.0,
    fix_mask: torch.Tensor = None,
    fp16: bool = True,
    verbose: bool = False,
    tokenizer = None,
) -> Tuple[torch.Tensor, List[dict]]:
    """
    Multi-step entropy-based denoising from x_t to x_0.
    
    Returns:
        x_0: Denoised sequence [B, L]
        step_info: List of dicts with info about each step
    """
    device = x_t.device
    x = x_t.clone()
    step_info = []
    
    if fix_mask is None:
        fix_mask = (x_t != mask_token_id)
    
    for step in range(num_steps):
        mask_index = (x == mask_token_id)
        num_masked = mask_index.sum().item()
        
        if num_masked == 0:
            break
        
        # Forward pass through backbone
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            outputs = backbone(
                input_ids=x,
                attention_mask=attention_mask,
                is_causal=False,
            )
            logits = outputs.logits
        
        # Shift logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        
        # Get confidence and predictions
        probs = torch.softmax(logits.float(), dim=-1)
        x0_full = probs.argmax(dim=-1) if temperature == 0 else None
        if temperature > 0:
            flat_probs = (probs / temperature).softmax(dim=-1).view(-1, probs.shape[-1])
            x0_full = torch.multinomial(flat_probs, num_samples=1).view(probs.shape[:-1])
        
        log_probs = torch.log(probs.clamp(min=1e-10))
        confidence = (probs * log_probs).sum(dim=-1)  # Negative entropy
        
        # Only consider masked positions
        full_confidence = torch.full_like(x, float('-inf'), dtype=confidence.dtype)
        full_confidence[mask_index] = confidence[mask_index]
        
        # Calculate how many to transfer
        num_mask_tokens = mask_index.sum(dim=1)
        if step < num_steps - 1:
            transfer_ratio = 1.0 / (num_steps - step)
            num_to_transfer = (num_mask_tokens.float() * transfer_ratio).long()
        else:
            num_to_transfer = num_mask_tokens
        
        max_transfer = int(num_to_transfer.max().item())
        
        transferred_positions = []
        if max_transfer > 0:
            _, transfer_indices = torch.topk(full_confidence, max_transfer, dim=1)
            
            batch_size = x.size(0)
            valid_mask = torch.arange(max_transfer, device=device).unsqueeze(0) < num_to_transfer.unsqueeze(1)
            
            valid_transfer_indices = transfer_indices[valid_mask]
            valid_batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(transfer_indices)[valid_mask]
            
            x[valid_batch_indices, valid_transfer_indices] = x0_full[valid_batch_indices, valid_transfer_indices]
            transferred_positions = valid_transfer_indices.tolist()
        
        step_info.append({
            "step": step + 1,
            "masked_before": num_masked,
            "transferred": len(transferred_positions),
            "transferred_positions": transferred_positions,
            "x_state": x[0].clone() if x.dim() > 1 else x.clone(),
        })
        
        if verbose and tokenizer:
            print(f"\n  Step {step + 1}: {num_masked} masked -> transferred {len(transferred_positions)} tokens")
    
    return x, step_info


def sample_tokens_from_logits(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Sample tokens from logits."""
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(probs.shape[:-1])
    else:
        return logits.argmax(dim=-1)


def format_token(token: str, max_len: int = 15) -> str:
    """Format a token for display, handling special characters."""
    # Escape special characters for display
    display = token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    # Replace other non-printable characters
    display = ''.join(c if c.isprintable() or c == ' ' else f'\\x{ord(c):02x}' for c in display)
    if len(display) > max_len:
        display = display[:max_len-2] + ".."
    return display


def pad_with_color(text: str, width: int, color_start: str = "", color_end: str = "") -> str:
    """Pad text to width, then wrap with color codes."""
    # Pad the plain text first
    padded = f"{text:<{width}}"
    # Then wrap with colors
    if color_start:
        return f"{color_start}{padded}{color_end}"
    return padded


def print_comparison_table(
    prompt_tokens: List[str],
    completion_tokens: List[str],
    x_t_tokens: List[str],
    x_0_tokens: List[str],
    masked_indices: List[int],
    token_probs: List[float],
    prompt_len_for_probs: int,
    use_colors: bool = True,
):
    """Print a detailed comparison table of tokens."""
    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)
    
    # Calculate column widths
    col_width = 16
    idx_width = 5
    prob_width = 8
    
    # Header
    header = f"{'Idx':>{idx_width}} | {'Ground Truth':<{col_width}} | {'x_t (masked)':<{col_width}} | {'x_0 (predicted)':<{col_width}} | {'p':>{prob_width}} | {'Match':<5}"
    separator = "-" * len(header)
    
    print(f"\n{Colors.BOLD}=== Token Comparison Table ==={Colors.RESET}\n")
    print(f"{Colors.DIM}Prompt length: {prompt_len}, Completion length: {completion_len}{Colors.RESET}")
    print(f"{Colors.DIM}Masked {len(masked_indices)} tokens in completion at indices: {masked_indices}{Colors.RESET}\n")
    
    print(header)
    print(separator)
    
    # Print prompt tokens
    if prompt_len > 0:
        print(f"{Colors.CYAN}--- Prompt (fixed) ---{Colors.RESET}")
        for i, token in enumerate(prompt_tokens[:min(5, prompt_len)]):
            formatted = format_token(token, col_width - 2)
            gt_col = pad_with_color(formatted, col_width, Colors.DIM, Colors.RESET)
            fixed_col = pad_with_color("[fixed]", col_width, Colors.DIM, Colors.RESET)
            prob_col = pad_with_color("--", prob_width, Colors.DIM, Colors.RESET)
            match_col = pad_with_color("--", 5, Colors.DIM, Colors.RESET)
            print(f"{i:>{idx_width}} | {gt_col} | {fixed_col} | {fixed_col} | {prob_col} | {match_col}")
        if prompt_len > 5:
            print(f"{Colors.DIM}{'':>{idx_width}} | ... ({prompt_len - 5} more prompt tokens) ...{Colors.RESET}")
    
    print(f"{Colors.YELLOW}--- Completion (denoised) ---{Colors.RESET}")
    
    # Print completion tokens
    correct_count = 0
    for i in range(completion_len):
        gt_token = completion_tokens[i]
        is_masked = i in masked_indices
        
        x_t_token = x_t_tokens[prompt_len + i] if prompt_len + i < len(x_t_tokens) else "[OOB]"
        x_0_token = x_0_tokens[prompt_len + i] if prompt_len + i < len(x_0_tokens) else "[OOB]"
        
        # Get probability for this token
        prob_idx = prompt_len_for_probs + i
        prob = token_probs[prob_idx] if prob_idx < len(token_probs) else 0.0
        prob_str = f"{prob:.4f}"
        
        gt_formatted = format_token(gt_token, col_width - 2)
        x_t_formatted = format_token(x_t_token, col_width - 2)
        x_0_formatted = format_token(x_0_token, col_width - 2)
        
        is_correct = (gt_token == x_0_token)
        if is_correct:
            correct_count += 1
        
        idx = prompt_len + i
        
        if use_colors:
            if is_masked:
                # This position was masked
                if is_correct:
                    match_str = f"{Colors.GREEN}✓{Colors.RESET}"
                    x_0_display = pad_with_color(x_0_formatted, col_width, Colors.GREEN, Colors.RESET)
                else:
                    match_str = f"{Colors.RED}✗{Colors.RESET}"
                    x_0_display = pad_with_color(x_0_formatted, col_width, Colors.RED, Colors.RESET)
                x_t_display = pad_with_color("[MASK]", col_width, Colors.MAGENTA, Colors.RESET)
                gt_display = pad_with_color(gt_formatted, col_width, Colors.YELLOW, Colors.RESET)
                # Color probability based on value (green if high, red if low)
                if prob >= 0.8:
                    prob_display = pad_with_color(prob_str, prob_width, Colors.GREEN, Colors.RESET)
                elif prob >= 0.5:
                    prob_display = pad_with_color(prob_str, prob_width, Colors.YELLOW, Colors.RESET)
                else:
                    prob_display = pad_with_color(prob_str, prob_width, Colors.RED, Colors.RESET)
            else:
                # This position was not masked (kept from ground truth)
                match_str = f"{Colors.DIM}={Colors.RESET}"
                x_t_display = pad_with_color(x_t_formatted, col_width, Colors.DIM, Colors.RESET)
                x_0_display = pad_with_color(x_0_formatted, col_width, Colors.DIM, Colors.RESET)
                gt_display = pad_with_color(gt_formatted, col_width, Colors.DIM, Colors.RESET)
                prob_display = pad_with_color(prob_str, prob_width, Colors.DIM, Colors.RESET)
        else:
            match_str = "✓" if is_correct else "✗"
            x_t_display = f"{'[MASK]' if is_masked else x_t_formatted:<{col_width}}"
            x_0_display = f"{x_0_formatted:<{col_width}}"
            gt_display = f"{gt_formatted:<{col_width}}"
            prob_display = f"{prob_str:>{prob_width}}"
        
        print(f"{idx:>{idx_width}} | {gt_display} | {x_t_display} | {x_0_display} | {prob_display} | {match_str}")
    
    print(separator)
    
    # Summary
    masked_correct = sum(1 for i in masked_indices if completion_tokens[i] == x_0_tokens[prompt_len + i])
    print(f"\n{Colors.BOLD}=== Summary ==={Colors.RESET}")
    print(f"  Total completion tokens: {completion_len}")
    print(f"  Masked tokens: {len(masked_indices)}")
    print(f"  Correctly predicted (masked): {Colors.GREEN}{masked_correct}/{len(masked_indices)}{Colors.RESET} ({100*masked_correct/max(1,len(masked_indices)):.1f}%)")
    print(f"  Total correct: {correct_count}/{completion_len} ({100*correct_count/completion_len:.1f}%)")


def print_sequences_inline(
    tokenizer,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    masked_indices: List[int],
):
    """Print sequences in a more compact inline format."""
    prompt_len = len(prompt_ids)
    
    print(f"\n{Colors.BOLD}=== Inline Sequence View ==={Colors.RESET}\n")
    
    # Ground truth
    gt_text = tokenizer.decode(torch.cat([prompt_ids, completion_ids]))
    print(f"{Colors.CYAN}Ground Truth:{Colors.RESET}")
    print(f"  {gt_text[:200]}{'...' if len(gt_text) > 200 else ''}\n")
    
    # x_t (with masks shown)
    x_t_tokens = [tokenizer.decode([t]) for t in x_t]
    x_t_display = ""
    for i, tok in enumerate(x_t_tokens):
        if i >= prompt_len and (i - prompt_len) in masked_indices:
            x_t_display += f"{Colors.MAGENTA}[M]{Colors.RESET}"
        else:
            x_t_display += tok
    print(f"{Colors.YELLOW}x_t (masked):{Colors.RESET}")
    print(f"  {x_t_display[:300]}{'...' if len(x_t_display) > 300 else ''}\n")
    
    # x_0 (predictions)
    x_0_text = tokenizer.decode(x_0)
    print(f"{Colors.GREEN}x_0 (predicted):{Colors.RESET}")
    print(f"  {x_0_text[:200]}{'...' if len(x_0_text) > 200 else ''}")


def main():
    parser = argparse.ArgumentParser(description="Simulate single-step denoising")
    
    # Model
    parser.add_argument("--backbone_path", type=str, default="fredzzp/open-dcoder-0.5B",
                        help="Path to backbone model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_true")
    
    # Input text
    parser.add_argument("--prompt", type=str, default="def fibonacci(n):",
                        help="Prompt text")
    parser.add_argument("--completion", type=str, default=None,
                        help="Completion text (if not provided, will generate)")
    parser.add_argument("--generate_completion", action="store_true",
                        help="Generate completion from model instead of using provided text")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens to generate for completion")
    
    # Denoising parameters
    parser.add_argument("--t", type=float, default=0.5,
                        help="Timestep value (0=fully unmasked, 1=fully masked)")
    parser.add_argument("--t_on", type=float, default=0.55,
                        help="Upper bound of remasking interval")
    parser.add_argument("--t_off", type=float, default=0.05,
                        help="Lower bound of remasking interval")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "loop"],
                        help="Alpha schedule type")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Number of denoising steps (1=single step, >1=multi-step)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    
    # Output options
    parser.add_argument("--no_color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("--inline", action="store_true",
                        help="Show inline sequence view")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output for multi-step denoising")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.no_fp16:
        args.fp16 = False
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    use_colors = not args.no_color and sys.stdout.isatty()
    
    # Disable colors if not supported
    if not use_colors:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")
    
    print(f"{Colors.BOLD}=== Denoising Simulation ==={Colors.RESET}\n")
    print(f"Backbone: {args.backbone_path}")
    print(f"Device: {args.device}")
    print(f"Timestep t: {args.t}")
    print(f"Schedule: {args.schedule}")
    print(f"Num steps: {args.num_steps}")
    print(f"Temperature: {args.temperature}")
    
    # Load model and tokenizer
    print(f"\n{Colors.DIM}Loading model...{Colors.RESET}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone_path,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    ).to(args.device)
    backbone.eval()
    
    # Get mask token id
    mask_token_id = getattr(backbone.config, 'mask_token_id', None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        if mask_token_id is None:
            mask_token_id = backbone.config.vocab_size
            print(f"{Colors.YELLOW}Warning: No mask_token_id found, using {mask_token_id}{Colors.RESET}")
    
    # Tokenize prompt
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt, add_special_tokens=False), device=args.device)
    
    # Get or generate completion
    if args.completion is None or args.generate_completion:
        print(f"{Colors.DIM}Generating completion...{Colors.RESET}")
        with torch.no_grad():
            input_ids = prompt_ids.unsqueeze(0)
            outputs = backbone.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            completion_ids = outputs[0, len(prompt_ids):]
    else:
        completion_ids = torch.tensor(tokenizer.encode(args.completion, add_special_tokens=False), device=args.device)
    
    if len(completion_ids) == 0:
        print(f"{Colors.RED}Error: Empty completion{Colors.RESET}")
        return
    
    # Compute alpha from timestep
    alpha = compute_alpha(
        t=args.t,
        schedule=args.schedule,
        t_on=args.t_on,
        t_off=args.t_off,
    )
    
    # Calculate tokens to mask (using same logic as create_masked_sequence)
    num_to_keep = int(len(completion_ids) * alpha)
    num_to_mask = len(completion_ids) - num_to_keep
    
    print(f"\n{Colors.BOLD}Alpha (fraction unmasked):{Colors.RESET} {alpha:.3f}")
    print(f"{Colors.BOLD}Completion length:{Colors.RESET} {len(completion_ids)} tokens")
    print(f"{Colors.BOLD}Tokens to mask:{Colors.RESET} {num_to_mask} ({100*(1-alpha):.1f}%)")
    
    # Create masked sequence
    x_t, mask_positions, masked_indices = create_masked_sequence(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        mask_token_id=mask_token_id,
        alpha=alpha,
    )
    
    # Create attention mask
    x_t_batch = x_t.unsqueeze(0)
    attention_mask = torch.ones_like(x_t_batch)
    
    # Create fix_mask (prompt positions are fixed)
    fix_mask = torch.zeros_like(x_t_batch, dtype=torch.bool)
    fix_mask[0, :len(prompt_ids)] = True
    
    # Run denoising
    print(f"\n{Colors.DIM}Running denoising...{Colors.RESET}")
    
    if args.num_steps > 1:
        # Multi-step denoising
        x_0, step_info = multi_step_denoise(
            x_t=x_t_batch,
            backbone=backbone,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            num_steps=args.num_steps,
            temperature=args.temperature,
            fix_mask=fix_mask,
            fp16=args.fp16,
            verbose=args.verbose,
            tokenizer=tokenizer,
        )
        x_0 = x_0[0]  # Remove batch dimension
        
        if args.verbose:
            print(f"\n{Colors.BOLD}Multi-step denoising progress:{Colors.RESET}")
            for info in step_info:
                print(f"  Step {info['step']}: {info['masked_before']} masked -> transferred {info['transferred']} tokens")
        # For multi-step, we need to get final probabilities
        # Run backbone on final x_0 to get probabilities
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                final_outputs = backbone(
                    input_ids=x_0.unsqueeze(0),
                    attention_mask=attention_mask,
                    is_causal=False,
                )
                final_logits = final_outputs.logits
                final_logits = torch.cat([final_logits[:, :1], final_logits[:, :-1]], dim=1)
                probs = torch.softmax(final_logits.float(), dim=-1)
                # Get probability of each token in x_0
                token_probs = torch.gather(probs, -1, x_0.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0)
    else:
        # Single-step denoising
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                outputs = backbone(
                    input_ids=x_t_batch,
                    attention_mask=attention_mask,
                    is_causal=False,
                )
                logits = outputs.logits
                
                # Shift logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                
                # Compute probabilities
                probs = torch.softmax(logits.float(), dim=-1)
                
                # Sample predictions
                x_0_pred = sample_tokens_from_logits(logits, temperature=args.temperature)
                
                # Build x_0: predictions for masked positions, ground truth for unmasked
                x_0 = torch.cat([prompt_ids, completion_ids])
                x_0[mask_positions] = x_0_pred[0, mask_positions]
                
                # Get probability of each token in x_0
                # For masked positions: prob of predicted token
                # For unmasked positions: prob of ground truth token  
                token_probs = torch.gather(probs, -1, x_0.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0)
    
    # Ensure prompt tokens are ground truth
    x_0[:len(prompt_ids)] = prompt_ids
    
    # Convert to token strings for display
    prompt_tokens = [tokenizer.decode([t]) for t in prompt_ids]
    completion_tokens = [tokenizer.decode([t]) for t in completion_ids]
    x_t_tokens = [tokenizer.decode([t]) if t != mask_token_id else "[MASK]" for t in x_t]
    x_0_tokens = [tokenizer.decode([t]) for t in x_0]
    
    # Print comparison
    print_comparison_table(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        x_t_tokens=x_t_tokens,
        x_0_tokens=x_0_tokens,
        masked_indices=masked_indices,
        token_probs=token_probs.tolist(),
        prompt_len_for_probs=len(prompt_ids),
        use_colors=use_colors,
    )
    
    if args.inline:
        print_sequences_inline(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            x_t=x_t,
            x_0=x_0,
            masked_indices=masked_indices,
        )
    
    print(f"\n{Colors.BOLD}=== Legend ==={Colors.RESET}")
    print(f"  {Colors.MAGENTA}[MASK]{Colors.RESET} = Masked position in x_t")
    print(f"  {Colors.GREEN}✓{Colors.RESET} = Correctly predicted (masked position)")
    print(f"  {Colors.RED}✗{Colors.RESET} = Incorrectly predicted (masked position)")
    print(f"  {Colors.DIM}={Colors.RESET} = Unmasked position (copied from ground truth)")


if __name__ == "__main__":
    main()
