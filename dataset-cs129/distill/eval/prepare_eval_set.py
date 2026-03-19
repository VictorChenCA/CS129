"""
Prepare evaluation dataset from test splits of the subsets used for training.

Loads test splits from the same subsets used for training (to avoid data leakage),
tokenizes them for inference with the fine-tuned model, and saves to GCS for evaluation.
"""

import argparse
import tempfile
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pseudo_rationales.tokenizer import load_math_dataset
from train.build_train_set import (
    ensure_answer_column,
    persist_dataset,
)


def tokenize_for_eval(
    ds: Dataset,
    model_name: str,
    max_length: int = 2048,
) -> Dataset:
    """
    Tokenize dataset for evaluation (inference only).
    
    Creates `input_ids` and `attention_mask` for the problem prompt only.
    No labels needed since we'll generate answers.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize_example(ex):
        """Tokenize just the problem for inference."""
        problem_text = ex.get("problem")
        if problem_text is None:
            raise KeyError(f"Expected 'problem' field. Available: {list(ex.keys())}")
        
        # Build user message (same format as training)
        user_content = f"Solve the following math problem.\n\nProblem:\n{problem_text}\n"
        messages = [{"role": "user", "content": user_content}]
        
        # Apply chat template to get the prompt text
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize the prompt
        tokenized = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors=None,  # Return as lists, not tensors
        )
        
        return {
            "messages": messages,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt_text": prompt_text,  # Keep for debugging
        }

    return ds.map(_tokenize_example)


def build_eval_dataset(
    model_name: str,
    dataset_name: str = "EleutherAI/hendrycks_math",
    subsets: list[str] = ["algebra", "geometry", "intermediate_algebra", "number_theory", "precalculus"],
    split: str = "test",
    max_length: int = 2048,
) -> Dataset:
    """
    Build evaluation dataset from test splits of specified subsets.
    
    Args:
        model_name: Model name for tokenization
        dataset_name: HuggingFace dataset name
        subsets: List of subset names to load (should match training subsets)
        split: Dataset split (should be "test" to avoid data leakage)
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset ready for evaluation
    """
    # Load and concatenate subsets
    parts = []
    for subset in subsets:
        ds = load_math_dataset(dataset_name=dataset_name, subset=subset, split=split)
        ds = ds.map(lambda ex, _s=subset: {"subset": _s})
        parts.append(ds)
    
    base = concatenate_datasets(parts)
    
    # Ensure answer column exists (for accuracy calculation)
    base = ensure_answer_column(base)
    
    # Tokenize for inference
    tokenized = tokenize_for_eval(base, model_name=model_name, max_length=max_length)
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(
        description="Prepare evaluation dataset from test splits of training subsets."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name for tokenization (should match fine-tuned model base)",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="algebra,geometry,intermediate_algebra,number_theory,precalculus",
        help='Comma-separated list of subsets to use for eval (default: training subsets)',
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="EleutherAI/hendrycks_math",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (should be 'test' to avoid data leakage from training)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--gcp_project",
        type=str,
        default="cs224n-dapo-distill",
    )
    parser.add_argument(
        "--out_dataset",
        type=str,
        required=True,
        help="Output path (local dir or gs:// prefix) for eval dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output path",
    )
    parser.add_argument(
        "--print_n",
        type=int,
        default=3,
        help="Number of examples to print for verification",
    )
    
    args = parser.parse_args()
    
    # Parse subsets
    subset_list = [s.strip() for s in args.subsets.split(",") if s.strip()]
    if len(subset_list) < 1:
        raise ValueError("Must specify at least one subset")
    
    print(f"Loading subsets: {subset_list}")
    print(f"Using model: {args.model_name}")
    
    # Build eval dataset
    eval_ds = build_eval_dataset(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        subsets=subset_list,
        split=args.split,
        max_length=args.max_length,
    )
    
    print(f"Built eval dataset with {len(eval_ds)} examples")
    print(f"Columns: {eval_ds.column_names}")
    
    # Print sample examples
    for i in range(min(args.print_n, len(eval_ds))):
        ex = eval_ds[i]
        print(f"\nExample {i+1}:")
        print(f"  Subset: {ex.get('subset', 'N/A')}")
        print(f"  Problem (first 150 chars): {ex.get('problem', '')[:150]}...")
        print(f"  Answer: {ex.get('answer', 'N/A')}")
        print(f"  Input IDs length: {len(ex.get('input_ids', []))}")
    
    # Save to output location
    output_loc = persist_dataset(
        eval_ds, 
        args.out_dataset, 
        gcp_project=args.gcp_project, 
        overwrite=args.overwrite
    )
    print(f"\nSaved eval dataset to: {output_loc}")


if __name__ == "__main__":
    main()
