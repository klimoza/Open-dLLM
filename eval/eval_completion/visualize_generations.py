#!/usr/bin/env python3
"""
Visualize generations from evaluation results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from transformers import AutoTokenizer


def load_samples(jsonl_path: str) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            # if line.strip():
            samples.append(json.loads(line))
    return samples



def print_section(title: str, content: str, width: int = 100):
    """Print a formatted section."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(content)
    print("=" * width)


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the tokenizer."""
    if not text:
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def visualize_sample(sample: Dict, index: int = None, show_all_responses: bool = True, tokenizer=None):
    """Visualize a single sample."""
    doc = sample['doc']
    task_id = doc.get('task_id', f'Task {sample.get("doc_id", index)}')
    
    print("\n" + "🔷" * 50)
    print(f"  TASK: {task_id}")
    if index is not None:
        print(f"  INDEX: {index}")
    print("🔷" * 50)
    
    # Prompt
    prompt = doc.get('prompt', '')
    prompt_tokens = count_tokens(prompt, tokenizer) if tokenizer else None
    print_section("📝 PROMPT", prompt)
    if prompt_tokens is not None:
        print(f"  📊 Tokens: {prompt_tokens}")
    
    # Canonical Solution
    canonical = doc.get('canonical_solution', '')
    if canonical:
        canonical_tokens = count_tokens(canonical, tokenizer) if tokenizer else None
        print_section("✅ CANONICAL SOLUTION", canonical)
        if canonical_tokens is not None:
            print(f"  📊 Tokens: {canonical_tokens}")
    
    # Metrics
    pass_at_1 = sample.get('pass@1', 0.0)
    pass_at_10 = sample.get('pass@10', 0.0)
    print(f"\n📊 METRICS:")
    print(f"   Pass@1:  {pass_at_1:.2%}")
    print(f"   Pass@10: {pass_at_10:.2%}")
    
    # Generated Responses
    resps = sample.get('resps', [])
    filtered_resps = sample.get('filtered_resps', [])
    
    if resps:
        num_resps = len(resps[0]) if isinstance(resps[0], list) else len(resps)
        print(f"\n🤖 GENERATED RESPONSES ({num_resps} total):")
        print("-" * 100)
        
        responses_to_show = resps[0] if isinstance(resps[0], list) else resps
        filtered_to_show = filtered_resps[0] if filtered_resps and isinstance(filtered_resps[0], list) else filtered_resps
        
        for i, resp in enumerate(responses_to_show):
            status = "✅" if pass_at_10 > 0 else "❌"
            resp_tokens = count_tokens(resp, tokenizer) if tokenizer else None
            print(f"\n{status} Response {i+1}:")
            if resp_tokens is not None:
                print(f"  📊 Tokens: {resp_tokens}")
            print(f"{'─' * 98}")
            
            # Show full response
            print(f"  {resp}")
            
            # Show filtered response if available and different
            # if filtered_to_show and i < len(filtered_to_show):
            #     filtered = filtered_to_show[i]
            #     if filtered != resp and filtered.strip():
                    # filtered_tokens = count_tokens(filtered, tokenizer) if tokenizer else None
                    # print(f"\n  [FILTERED]: {format_code(filtered)}")
                    # if filtered_tokens is not None:
                    #     print(f"  📊 Filtered Tokens: {filtered_tokens}")
    
    # Target (test code)
    target = sample.get('target', '')
    if target:
        print_section("🎯 TARGET (Test Code)", target)


def print_summary(samples: List[Dict]):
    """Print summary statistics."""
    total = len(samples)
    pass_at_1_counts = [s.get('pass@1', 0.0) for s in samples]
    pass_at_10_counts = [s.get('pass@10', 0.0) for s in samples]
    
    avg_pass_at_1 = sum(pass_at_1_counts) / len(pass_at_1_counts) if pass_at_1_counts else 0
    avg_pass_at_10 = sum(pass_at_10_counts) / len(pass_at_10_counts) if pass_at_10_counts else 0
    
    print("\n" + "=" * 100)
    print("  📈 SUMMARY")
    print("=" * 100)
    print(f"  Total samples: {total}")
    print(f"  Average Pass@1:  {avg_pass_at_1:.2%}")
    print(f"  Average Pass@10: {avg_pass_at_10:.2%}")
    print(f"  Samples with Pass@1 > 0:  {sum(1 for p in pass_at_1_counts if p > 0)}")
    print(f"  Samples with Pass@10 > 0: {sum(1 for p in pass_at_10_counts if p > 0)}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation generations")
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to the JSONL samples file"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Show only a specific sample index (0-based)"
    )
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="Show a range of samples (e.g., '0:5' for first 5)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary statistics"
    )
    parser.add_argument(
        "--max-responses",
        type=int,
        default=None,
        help="Maximum number of responses to show per sample"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all responses (not truncated)"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer from HuggingFace (fredzzp/open-dcoder-0.5B)...", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained("fredzzp/open-dcoder-0.5B")
        print("Tokenizer loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer: {e}", file=sys.stderr)
        print("Continuing without token counting...", file=sys.stderr)
        tokenizer = None
    
    # Load samples
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)
    
    samples = load_samples(str(jsonl_path))
    
    if args.summary:
        print_summary(samples)
        return
    
    # Filter samples
    if args.index is not None:
        if 0 <= args.index < len(samples):
            visualize_sample(samples[args.index], index=args.index, show_all_responses=args.show_all, tokenizer=tokenizer)
        else:
            print(f"Error: Index {args.index} out of range (0-{len(samples)-1})", file=sys.stderr)
            sys.exit(1)
    elif args.range:
        try:
            start, end = map(int, args.range.split(':'))
            for i in range(start, min(end, len(samples))):
                visualize_sample(samples[i], index=i, show_all_responses=args.show_all, tokenizer=tokenizer)
        except ValueError:
            print("Error: Invalid range format. Use 'start:end' (e.g., '0:5')", file=sys.stderr)
            sys.exit(1)
    else:
        # Show first 5 by default
        print_summary(samples)
        print("\n\nShowing first 5 samples:")
        for i in range(min(5, len(samples))):
            visualize_sample(samples[i], index=i, show_all_responses=args.show_all, tokenizer=tokenizer)
        
        if len(samples) > 5:
            print(f"\n\n... ({len(samples) - 5} more samples not shown)")
            print("Use --index N to view a specific sample, or --range start:end for a range")


if __name__ == "__main__":
    main()

