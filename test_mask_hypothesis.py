# test_mask_hypothesis.py
# Test hypothesis: output at mask position does not depend on input token at that position

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM

# 1. Setup
model_path = "fredzzp/open-dcoder-0.5B"
tokenizer_path = model_path
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = model.to(device).eval()

if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    model.resize_token_embeddings(len(tokenizer))
    print("Added new [MASK] token.")

mask_token_id = tokenizer.mask_token_id
print(f"Mask token ID: {mask_token_id}")

# 2. Create a test sequence: "t1 t2 t3 M t5 t6 t7"
# Using actual tokens from a simple prompt
base_prompt = "def hello_world():\n    return 'Hello, World!'"
base_ids = tokenizer(base_prompt, return_tensors="pt").input_ids.to(device)
print(f"Base tokens: {tokenizer.convert_ids_to_tokens(base_ids[0].tolist())}")
print(f"Base IDs: {base_ids[0].tolist()}")

# Insert mask at position 3 (0-indexed)
mask_position = 5
seq_with_mask = base_ids.clone()
original_token_at_mask_pos = seq_with_mask[0, mask_position].item()
seq_with_mask[0, mask_position] = mask_token_id

print(f"\nSequence with mask at position {mask_position}:")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(seq_with_mask[0].tolist())}")

# Position to change for comparison (a non-mask position)
change_position = 2  # t3 in our notation

# 3. Helper function to get logits
# CRITICAL: Use is_causal=False for bidirectional attention (as in training/generation)
# and apply the logit shift to align predictions with positions
def get_logits(input_ids, apply_shift=True):
    with torch.no_grad():
        outputs = model(input_ids, is_causal=False)
        logits = outputs.logits
        if apply_shift:
            # CRITICAL: Shift logits to align with token positions (from generation_utils.py)
            # This makes logits[i] predict token[i] instead of token[i+1]
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits

# Test: systematically check all positions - compare WITH and WITHOUT shift
print("\n" + "="*70)
print("PRELIMINARY: Test if output[i] depends on input[i]")
print("="*70)

# Use fixed test token
test_token_id = 1000  # A common token

print("\n--- WITHOUT shift (original logits) ---")
print("This tests: does original_logits[i] depend on input[i]?")
for test_pos in range(1, len(base_ids[0])):
    baseline = get_logits(base_ids, apply_shift=False)
    baseline_at_pos = baseline[0, test_pos].float()
    
    modified_ids = base_ids.clone()
    modified_ids[0, test_pos] = test_token_id
    modified = get_logits(modified_ids, apply_shift=False)
    modified_at_pos = modified[0, test_pos].float()
    
    diff = (modified_at_pos - baseline_at_pos).abs().mean().item()
    orig_token = tokenizer.convert_ids_to_tokens([base_ids[0, test_pos].item()])[0]
    print(f"Pos {test_pos:2d} ({orig_token:15}): Change input[{test_pos}] -> Mean Δ at original[{test_pos}]: {diff:.4f}")

print("\n--- WITH shift (shifted logits) ---")
print("This tests: does shifted_logits[i] depend on input[i]?")
print("Since shifted[i] = original[i-1], we check if original[i-1] attends to input[i]")
for test_pos in range(1, len(base_ids[0])):
    baseline = get_logits(base_ids, apply_shift=True)
    baseline_at_pos = baseline[0, test_pos].float()
    
    modified_ids = base_ids.clone()
    modified_ids[0, test_pos] = test_token_id
    modified = get_logits(modified_ids, apply_shift=True)
    modified_at_pos = modified[0, test_pos].float()
    
    diff = (modified_at_pos - baseline_at_pos).abs().mean().item()
    orig_token = tokenizer.convert_ids_to_tokens([base_ids[0, test_pos].item()])[0]
    print(f"Pos {test_pos:2d} ({orig_token:15}): Change input[{test_pos}] -> Mean Δ at shifted[{test_pos}]: {diff:.4f}")

print("\n--- CROSS-CHECK: Does position i-1 attend to position i? ---")
print("If bidirectional: change input[i] should affect original[i-1]")
for test_pos in range(2, len(base_ids[0])):
    # Get original[test_pos-1] from baseline
    baseline = get_logits(base_ids, apply_shift=False)
    baseline_at_prev = baseline[0, test_pos-1].float()
    
    # Change input[test_pos] and check original[test_pos-1]
    modified_ids = base_ids.clone()
    modified_ids[0, test_pos] = test_token_id
    modified = get_logits(modified_ids, apply_shift=False)
    modified_at_prev = modified[0, test_pos-1].float()
    
    diff = (modified_at_prev - baseline_at_prev).abs().mean().item()
    print(f"Change input[{test_pos}] -> Mean Δ at original[{test_pos-1}]: {diff:.4f}")

print("\n--- CROSS-CHECK: Does position i+1 attend to position i? (FUTURE attending to PAST) ---")
print("This should ALWAYS work in both causal and bidirectional")
for test_pos in range(1, len(base_ids[0])-1):
    baseline = get_logits(base_ids, apply_shift=False)
    baseline_at_next = baseline[0, test_pos+1].float()
    
    modified_ids = base_ids.clone()
    modified_ids[0, test_pos] = test_token_id
    modified = get_logits(modified_ids, apply_shift=False)
    modified_at_next = modified[0, test_pos+1].float()
    
    diff = (modified_at_next - baseline_at_next).abs().mean().item()
    print(f"Change input[{test_pos}] -> Mean Δ at original[{test_pos+1}]: {diff:.4f}")

print()

# 4. Get baseline logits with mask
baseline_logits = get_logits(seq_with_mask)
baseline_logits_at_mask = baseline_logits[0, mask_position].float()  # Convert to float32 for comparison
baseline_probs_at_mask = F.softmax(baseline_logits_at_mask, dim=-1)

print(f"\nBaseline logits shape: {baseline_logits.shape}")
print(f"Top 5 tokens at mask position (baseline):")
top_k_baseline = torch.topk(baseline_probs_at_mask, 5)
for idx, (prob, tok_id) in enumerate(zip(top_k_baseline.values, top_k_baseline.indices)):
    print(f"  {idx+1}. {tokenizer.convert_ids_to_tokens([tok_id.item()])[0]!r} (id={tok_id.item()}): {prob.item():.4f}")

# 5. Test: Change the token AT the mask position to different tokens
# (but keep it marked as mask in our mental model - we're testing if input matters)
print("\n" + "="*60)
print("EXPERIMENT 1: Change input token at MASK position")
print("="*60)

# Sample random tokens to substitute at mask position
vocab_size = tokenizer.vocab_size
random_token_ids = torch.randint(100, vocab_size - 100, (5,)).tolist()

mask_position_changes = []
for new_token_id in random_token_ids:
    seq_modified = seq_with_mask.clone()
    seq_modified[0, mask_position] = new_token_id
    
    new_logits = get_logits(seq_modified)
    new_logits_at_mask = new_logits[0, mask_position].float()
    
    # Compute difference metrics
    logit_diff = (new_logits_at_mask - baseline_logits_at_mask).abs()
    mean_diff = logit_diff.mean().item()
    max_diff = logit_diff.max().item()
    
    # KL divergence
    new_probs = F.softmax(new_logits_at_mask, dim=-1)
    kl_div = F.kl_div(new_probs.log(), baseline_probs_at_mask, reduction='sum').item()
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(new_logits_at_mask.unsqueeze(0), 
                                   baseline_logits_at_mask.unsqueeze(0)).item()
    
    mask_position_changes.append({
        'token': tokenizer.convert_ids_to_tokens([new_token_id])[0],
        'token_id': new_token_id,
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'kl_div': kl_div,
        'cos_sim': cos_sim
    })
    
    print(f"Token: {new_token_id:6d} ({tokenizer.convert_ids_to_tokens([new_token_id])[0]!r:15}) | "
          f"Mean Δ: {mean_diff:.4f} | Max Δ: {max_diff:.4f} | "
          f"KL: {kl_div:.4f} | Cos: {cos_sim:.6f}")

# 6. Test: Change the token at a DIFFERENT position (t3)
print("\n" + "="*60)
print(f"EXPERIMENT 2: Change input token at position {change_position} (non-mask)")
print("="*60)

other_position_changes = []
original_token_at_change = seq_with_mask[0, change_position].item()

for new_token_id in random_token_ids:
    seq_modified = seq_with_mask.clone()
    seq_modified[0, change_position] = new_token_id
    
    new_logits = get_logits(seq_modified)
    new_logits_at_mask = new_logits[0, mask_position].float()
    
    # Compute difference metrics
    logit_diff = (new_logits_at_mask - baseline_logits_at_mask).abs()
    mean_diff = logit_diff.mean().item()
    max_diff = logit_diff.max().item()
    
    # KL divergence
    new_probs = F.softmax(new_logits_at_mask, dim=-1)
    kl_div = F.kl_div(new_probs.log(), baseline_probs_at_mask, reduction='sum').item()
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(new_logits_at_mask.unsqueeze(0), 
                                   baseline_logits_at_mask.unsqueeze(0)).item()
    
    other_position_changes.append({
        'token': tokenizer.convert_ids_to_tokens([new_token_id])[0],
        'token_id': new_token_id,
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'kl_div': kl_div,
        'cos_sim': cos_sim
    })
    
    print(f"Token: {new_token_id:6d} ({tokenizer.convert_ids_to_tokens([new_token_id])[0]!r:15}) | "
          f"Mean Δ: {mean_diff:.4f} | Max Δ: {max_diff:.4f} | "
          f"KL: {kl_div:.4f} | Cos: {cos_sim:.6f}")

# 7. Summary comparison
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

avg_mask_mean_diff = sum(c['mean_diff'] for c in mask_position_changes) / len(mask_position_changes)
avg_mask_kl = sum(c['kl_div'] for c in mask_position_changes) / len(mask_position_changes)
avg_mask_cos = sum(c['cos_sim'] for c in mask_position_changes) / len(mask_position_changes)

avg_other_mean_diff = sum(c['mean_diff'] for c in other_position_changes) / len(other_position_changes)
avg_other_kl = sum(c['kl_div'] for c in other_position_changes) / len(other_position_changes)
avg_other_cos = sum(c['cos_sim'] for c in other_position_changes) / len(other_position_changes)

print(f"\nChanging token AT mask position:")
print(f"  Avg Mean Logit Diff: {avg_mask_mean_diff:.6f}")
print(f"  Avg KL Divergence:   {avg_mask_kl:.6f}")
print(f"  Avg Cosine Sim:      {avg_mask_cos:.6f}")

print(f"\nChanging token at position {change_position} (non-mask):")
print(f"  Avg Mean Logit Diff: {avg_other_mean_diff:.6f}")
print(f"  Avg KL Divergence:   {avg_other_kl:.6f}")
print(f"  Avg Cosine Sim:      {avg_other_cos:.6f}")

print(f"\nRatio (other/mask):")
print(f"  Mean Diff Ratio: {avg_other_mean_diff / (avg_mask_mean_diff + 1e-10):.2f}x")
print(f"  KL Div Ratio:    {avg_other_kl / (avg_mask_kl + 1e-10):.2f}x")

if avg_mask_mean_diff < avg_other_mean_diff * 0.1:
    print("\n✓ HYPOTHESIS SUPPORTED: Logits at mask position are largely independent of input token there")
elif avg_mask_mean_diff < avg_other_mean_diff:
    print("\n~ PARTIAL SUPPORT: Mask position is less sensitive but not fully independent")
else:
    print("\n✗ HYPOTHESIS NOT SUPPORTED: Mask position depends on input token")

# 8. Additional experiment: Compare mask token embedding vs other token embeddings
print("\n" + "="*60)
print("EXPERIMENT 3: What if we keep MASK token vs use original token?")
print("="*60)

# With mask token
seq_with_mask_orig = base_ids.clone()
seq_with_mask_orig[0, mask_position] = mask_token_id
logits_with_mask = get_logits(seq_with_mask_orig)
logits_with_mask_at_pos = logits_with_mask[0, mask_position].float()

# With original token (no mask)
seq_no_mask = base_ids.clone()  # Keep original token
logits_no_mask = get_logits(seq_no_mask)
logits_no_mask_at_pos = logits_no_mask[0, mask_position].float()

# Compare
diff_mask_vs_orig = (logits_with_mask_at_pos - logits_no_mask_at_pos).abs()
print(f"Comparing output at position {mask_position}:")
print(f"  With MASK token vs original token ({tokenizer.convert_ids_to_tokens([original_token_at_mask_pos])[0]}):")
print(f"  Mean logit diff: {diff_mask_vs_orig.mean().item():.4f}")
print(f"  Max logit diff:  {diff_mask_vs_orig.max().item():.4f}")

probs_with_mask = F.softmax(logits_with_mask_at_pos, dim=-1)
probs_no_mask = F.softmax(logits_no_mask_at_pos, dim=-1)
kl_mask_vs_orig = F.kl_div(probs_with_mask.log(), probs_no_mask, reduction='sum').item()
cos_mask_vs_orig = F.cosine_similarity(logits_with_mask_at_pos.unsqueeze(0), 
                                        logits_no_mask_at_pos.unsqueeze(0)).item()
print(f"  KL Divergence:   {kl_mask_vs_orig:.4f}")
print(f"  Cosine Sim:      {cos_mask_vs_orig:.6f}")

print("\nTop 5 predictions with MASK token:")
top_k_mask = torch.topk(probs_with_mask, 5)
for idx, (prob, tok_id) in enumerate(zip(top_k_mask.values, top_k_mask.indices)):
    print(f"  {idx+1}. {tokenizer.convert_ids_to_tokens([tok_id.item()])[0]!r}: {prob.item():.4f}")

print("\nTop 5 predictions with original token (no mask):")
top_k_orig = torch.topk(probs_no_mask, 5)
for idx, (prob, tok_id) in enumerate(zip(top_k_orig.values, top_k_orig.indices)):
    print(f"  {idx+1}. {tokenizer.convert_ids_to_tokens([tok_id.item()])[0]!r}: {prob.item():.4f}")

# 9. Look at attention pattern (if accessible)
print("\n" + "="*60)
print("EXPERIMENT 4: Test with multiple mask positions")
print("="*60)

# Create sequence with two masks
seq_two_masks = base_ids.clone()
seq_two_masks[0, 2] = mask_token_id  # Position 2
seq_two_masks[0, 4] = mask_token_id  # Position 4
print(f"Sequence with two masks: {tokenizer.convert_ids_to_tokens(seq_two_masks[0].tolist())}")

logits_two_masks = get_logits(seq_two_masks)

# Compare logits at position 2 and position 4
logits_at_pos2 = logits_two_masks[0, 2].float()
logits_at_pos4 = logits_two_masks[0, 4].float()

probs_at_pos2 = F.softmax(logits_at_pos2, dim=-1)
probs_at_pos4 = F.softmax(logits_at_pos4, dim=-1)

print("\nTop 5 predictions at mask position 2:")
top_k_p2 = torch.topk(probs_at_pos2, 5)
for idx, (prob, tok_id) in enumerate(zip(top_k_p2.values, top_k_p2.indices)):
    print(f"  {idx+1}. {tokenizer.convert_ids_to_tokens([tok_id.item()])[0]!r}: {prob.item():.4f}")

print("\nTop 5 predictions at mask position 4:")
top_k_p4 = torch.topk(probs_at_pos4, 5)
for idx, (prob, tok_id) in enumerate(zip(top_k_p4.values, top_k_p4.indices)):
    print(f"  {idx+1}. {tokenizer.convert_ids_to_tokens([tok_id.item()])[0]!r}: {prob.item():.4f}")

# 10. Check if output depends on OTHER mask's input
print("\n" + "="*60)
print("EXPERIMENT 5: Does changing one MASK affect other MASK's output?")
print("="*60)

# Baseline with two masks
baseline_two_masks = seq_two_masks.clone()
baseline_logits_2m = get_logits(baseline_two_masks)
baseline_at_pos4 = baseline_logits_2m[0, 4].float()

# Change the token at position 2 (which is a mask)
changes_at_other_mask = []
for new_token_id in random_token_ids[:3]:
    seq_mod = seq_two_masks.clone()
    seq_mod[0, 2] = new_token_id  # Change mask at position 2
    
    new_logits = get_logits(seq_mod)
    new_at_pos4 = new_logits[0, 4].float()
    
    diff = (new_at_pos4 - baseline_at_pos4).abs()
    mean_diff = diff.mean().item()
    
    print(f"Changing pos 2 to {new_token_id:6d} ({tokenizer.convert_ids_to_tokens([new_token_id])[0]!r:15}) | "
          f"Mean Δ at pos 4: {mean_diff:.4f}")

