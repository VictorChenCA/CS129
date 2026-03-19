"""
Separate script to calculate perplexity for saved predictions.

This script loads saved predictions and calculates perplexity separately
to avoid OOM errors during the main evaluation.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from distilling_step_by_step.train.train_model import (
    _download_gcs_prefix_to_dir,
    _load_dataset_with_fallback,
)


def calculate_perplexity(
    model,
    tokenizer,
    eval_ds,
    device,
    batch_size=32,
):
    """
    Calculate perplexity on the dataset.
    
    Uses a smaller batch size to avoid OOM errors.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    def collate_fn(batch):
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
        
        # Pad sequences (left padding for decoder-only models)
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(torch.cat([torch.full((pad_len,), tokenizer.pad_token_id), ids]))
            padded_attention_mask.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), mask]))
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
        }
    
    dataloader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            # For perplexity, we need labels (shifted input_ids)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def load_model(model_path: str, base_model_name: str, gcp_project: str):
    """Load model from local path or GCS."""
    from peft import PeftModel
    
    if model_path.startswith("gs://"):
        temp_dir = Path(tempfile.mkdtemp(prefix="model_dl_"))
        _download_gcs_prefix_to_dir(model_path, temp_dir, project=gcp_project)
        model_path = str(temp_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Check if it's a LoRA adapter
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        print(f"Loading LoRA adapter from {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading full model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity for saved predictions.")
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to saved predictions JSON file",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        required=True,
        help="Path to evaluation dataset (local or gs:// prefix)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (local or gs:// prefix, or model name for baseline)",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name (for loading LoRA adapters)",
    )
    parser.add_argument(
        "--gcp_project",
        type=str,
        default="cs224n-dapo-distill",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for perplexity calculation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save perplexity results (default: predictions_file with _perplexity.json)",
    )
    
    args = parser.parse_args()
    
    # Load saved predictions
    with open(args.predictions_file, "r") as f:
        predictions_data = json.load(f)
    
    print(f"Loaded predictions for model: {predictions_data['model_name']}")
    print(f"Accuracy: {predictions_data['accuracy']:.4f}")
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model_name, args.gcp_project)
    device = next(model.parameters()).device
    
    # Load eval dataset
    if args.eval_dataset_path.startswith("gs://"):
        temp_dir = Path(tempfile.mkdtemp(prefix="eval_ds_dl_"))
        _download_gcs_prefix_to_dir(args.eval_dataset_path, temp_dir, project=args.gcp_project)
        eval_dataset_path = str(temp_dir)
    else:
        eval_dataset_path = args.eval_dataset_path
    
    eval_ds = _load_dataset_with_fallback(eval_dataset_path)
    if isinstance(eval_ds, dict):
        # Prefer "test" split since we're evaluating on test data
        eval_ds = eval_ds.get("test", eval_ds.get("train", list(eval_ds.values())[0]))
    
    print(f"Loaded eval dataset with {len(eval_ds)} examples")
    
    # Calculate perplexity
    print(f"\nComputing perplexity with batch_size={args.batch_size}...")
    perplexity, avg_loss = calculate_perplexity(model, tokenizer, eval_ds, device, args.batch_size)
    
    print(f"\nResults:")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    
    # Update predictions file with perplexity
    predictions_data["perplexity"] = perplexity
    predictions_data["avg_loss"] = avg_loss
    
    output_file = args.output_file or args.predictions_file.replace(".json", "_perplexity.json")
    with open(output_file, "w") as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
