#!/usr/bin/env python3
"""
Analyze token lengths in a parquet dataset.
"""

import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths in a parquet dataset")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the parquet dataset file"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="fredzzp/open-dcoder-0.5B",
        help="Tokenizer model name or path"
    )
    parser.add_argument(
        "--instruction_col",
        type=str,
        default="instruction",
        help="Name of the instruction column"
    )
    parser.add_argument(
        "--output_col",
        type=str,
        default="output",
        help="Name of the output column"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for tokenization"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("Tokenizer loaded successfully.")
    
    # Load parquet dataset
    print(f"Loading dataset from {args.dataset_path}...")
    df = pd.read_parquet(args.dataset_path)
    print(f"Dataset loaded: {len(df)} rows")
    
    # Check columns
    if args.instruction_col not in df.columns:
        raise ValueError(f"Column '{args.instruction_col}' not found in dataset. Available columns: {df.columns.tolist()}")
    if args.output_col not in df.columns:
        raise ValueError(f"Column '{args.output_col}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    # Initialize lists to store lengths
    instruction_lengths = []
    output_lengths = []
    concatenated_lengths = []
    
    # Process in batches
    print("Tokenizing and computing lengths...")
    for i in tqdm(range(0, len(df), args.batch_size)):
        batch = df.iloc[i:i + args.batch_size]
        
        # Get instruction and output texts
        instructions = batch[args.instruction_col].astype(str).tolist()
        outputs = batch[args.output_col].astype(str).tolist()
        
        # Tokenize instructions
        instruction_tokens = tokenizer(
            instructions,
            add_special_tokens=False
        )
        batch_instruction_lengths = [len(ids) for ids in instruction_tokens['input_ids']]
        instruction_lengths.extend(batch_instruction_lengths)
        
        # Tokenize outputs
        output_tokens = tokenizer(
            outputs,
            add_special_tokens=False
        )
        batch_output_lengths = [len(ids) for ids in output_tokens['input_ids']]
        output_lengths.extend(batch_output_lengths)
        
        # Concatenate and tokenize
        concatenated = [inst + out for inst, out in zip(instructions, outputs)]
        concat_tokens = tokenizer(
            concatenated,
            add_special_tokens=False
        )
        batch_concat_lengths = [len(ids) for ids in concat_tokens['input_ids']]
        concatenated_lengths.extend(batch_concat_lengths)
    
    # Convert to numpy arrays for statistics
    instruction_lengths = np.array(instruction_lengths)
    output_lengths = np.array(output_lengths)
    concatenated_lengths = np.array(concatenated_lengths)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("TOKEN LENGTH STATISTICS")
    print("=" * 80)
    
    print(f"\n📊 INSTRUCTION COLUMN ('{args.instruction_col}'):")
    print(f"  Total samples: {len(instruction_lengths):,}")
    print(f"  Min length: {instruction_lengths.min():,} tokens")
    print(f"  Max length: {instruction_lengths.max():,} tokens")
    print(f"  Mean length: {instruction_lengths.mean():.2f} tokens")
    print(f"  Median length: {np.median(instruction_lengths):.2f} tokens")
    print(f"  Std deviation: {instruction_lengths.std():.2f} tokens")
    print(f"  25th percentile: {np.percentile(instruction_lengths, 25):.2f} tokens")
    print(f"  75th percentile: {np.percentile(instruction_lengths, 75):.2f} tokens")
    print(f"  90th percentile: {np.percentile(instruction_lengths, 90):.2f} tokens")
    print(f"  95th percentile: {np.percentile(instruction_lengths, 95):.2f} tokens")
    print(f"  99th percentile: {np.percentile(instruction_lengths, 99):.2f} tokens")
    
    print(f"\n📊 OUTPUT COLUMN ('{args.output_col}'):")
    print(f"  Total samples: {len(output_lengths):,}")
    print(f"  Min length: {output_lengths.min():,} tokens")
    print(f"  Max length: {output_lengths.max():,} tokens")
    print(f"  Mean length: {output_lengths.mean():.2f} tokens")
    print(f"  Median length: {np.median(output_lengths):.2f} tokens")
    print(f"  Std deviation: {output_lengths.std():.2f} tokens")
    print(f"  25th percentile: {np.percentile(output_lengths, 25):.2f} tokens")
    print(f"  75th percentile: {np.percentile(output_lengths, 75):.2f} tokens")
    print(f"  90th percentile: {np.percentile(output_lengths, 90):.2f} tokens")
    print(f"  95th percentile: {np.percentile(output_lengths, 95):.2f} tokens")
    print(f"  99th percentile: {np.percentile(output_lengths, 99):.2f} tokens")
    
    print(f"\n📊 CONCATENATED (instruction + output):")
    print(f"  Total samples: {len(concatenated_lengths):,}")
    print(f"  Min length: {concatenated_lengths.min():,} tokens")
    print(f"  Max length: {concatenated_lengths.max():,} tokens")
    print(f"  Mean length: {concatenated_lengths.mean():.2f} tokens")
    print(f"  Median length: {np.median(concatenated_lengths):.2f} tokens")
    print(f"  Std deviation: {concatenated_lengths.std():.2f} tokens")
    print(f"  25th percentile: {np.percentile(concatenated_lengths, 25):.2f} tokens")
    print(f"  75th percentile: {np.percentile(concatenated_lengths, 75):.2f} tokens")
    print(f"  90th percentile: {np.percentile(concatenated_lengths, 90):.2f} tokens")
    print(f"  95th percentile: {np.percentile(concatenated_lengths, 95):.2f} tokens")
    print(f"  99th percentile: {np.percentile(concatenated_lengths, 99):.2f} tokens")
    print(f"  Total tokens (sum): {concatenated_lengths.sum():,} tokens")
    
    # Additional statistics
    print(f"\n📊 ADDITIONAL STATISTICS:")
    print(f"  Sum of instruction lengths: {instruction_lengths.sum():,} tokens")
    print(f"  Sum of output lengths: {output_lengths.sum():,} tokens")
    print(f"  Sum of concatenated lengths: {concatenated_lengths.sum():,} tokens")
    print(f"  Average ratio (output/instruction): {(output_lengths / instruction_lengths).mean():.2f}")
    print(f"  Average ratio (instruction/output): {(instruction_lengths / output_lengths).mean():.2f}")
    
    # Length distribution bins
    print(f"\n📊 CONCATENATED LENGTH DISTRIBUTION:")
    bins = [0, 100, 500, 1000, 2000, 4000, 8000, 16000, float('inf')]
    bin_labels = ['0-100', '100-500', '500-1K', '1K-2K', '2K-4K', '4K-8K', '8K-16K', '16K+']
    hist, _ = np.histogram(concatenated_lengths, bins=bins)
    for label, count in zip(bin_labels, hist):
        percentage = (count / len(concatenated_lengths)) * 100
        print(f"  {label:>8} tokens: {count:>8,} samples ({percentage:>5.2f}%)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

