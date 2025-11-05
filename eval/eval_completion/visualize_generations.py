#!/usr/bin/env python3
"""
Visualize generations from evaluation results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_samples(jsonl_path: str) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def format_code(text: str, max_lines: int = None) -> str:
    """Format code with proper indentation."""
    lines = text.split('\n')
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return '\n'.join(lines)


def print_section(title: str, content: str, width: int = 100):
    """Print a formatted section."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(content)
    print("=" * width)


def visualize_sample(sample: Dict, index: int = None, show_all_responses: bool = True):
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
    print_section("📝 PROMPT", format_code(prompt, max_lines=30))
    
    # Canonical Solution
    canonical = doc.get('canonical_solution', '')
    if canonical:
        print_section("✅ CANONICAL SOLUTION", format_code(canonical))
    
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
            print(f"\n{status} Response {i+1}:")
            print(f"{'─' * 98}")
            
            # Show filtered response if available and different
            if filtered_to_show and i < len(filtered_to_show):
                filtered = filtered_to_show[i]
                if filtered != resp and filtered.strip():
                    print(f"  [FILTERED]: {filtered[:200]}")
                    print()
            
            # Show full response
            if show_all_responses or i < 3:  # Always show first 3
                resp_formatted = format_code(resp, max_lines=20)
                print(f"  {resp_formatted}")
            else:
                print(f"  {resp[:200]}...")
    
    # Target (test code)
    target = sample.get('target', '')
    if target:
        print_section("🎯 TARGET (Test Code)", format_code(target, max_lines=20))


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
            visualize_sample(samples[args.index], index=args.index, show_all_responses=args.show_all)
        else:
            print(f"Error: Index {args.index} out of range (0-{len(samples)-1})", file=sys.stderr)
            sys.exit(1)
    elif args.range:
        try:
            start, end = map(int, args.range.split(':'))
            for i in range(start, min(end, len(samples))):
                visualize_sample(samples[i], index=i, show_all_responses=args.show_all)
        except ValueError:
            print("Error: Invalid range format. Use 'start:end' (e.g., '0:5')", file=sys.stderr)
            sys.exit(1)
    else:
        # Show first 5 by default
        print_summary(samples)
        print("\n\nShowing first 5 samples:")
        for i in range(min(5, len(samples))):
            visualize_sample(samples[i], index=i, show_all_responses=args.show_all)
        
        if len(samples) > 5:
            print(f"\n\n... ({len(samples) - 5} more samples not shown)")
            print("Use --index N to view a specific sample, or --range start:end for a range")


if __name__ == "__main__":
    main()

