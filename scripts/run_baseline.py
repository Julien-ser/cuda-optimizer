#!/usr/bin/env python3
"""
Baseline Benchmark Script

Runs baseline performance benchmarks for standard models to establish
performance baseline before applying CUDA optimizations.

Task 1.3: Full implementation.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torchvision import models

from cuda_optimizer.profiling.base_profiler import BaseProfiler


def get_resnet50(input_shape: tuple) -> nn.Module:
    """Create ResNet50 model with adjusted input layer."""
    model = models.resnet50(weights=None)
    # Adjust first conv layer for custom input shape
    if input_shape[1] != 3:
        model.conv1 = nn.Conv2d(
            input_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    # Adjust final fc layer for classification
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


class BertSmallTransformer(nn.Module):
    """Small BERT-like transformer model for benchmarking."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        max_seq_length: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Token + position embeddings
        token_embeds = self.embedding(input_ids)
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_embeds = self.position_embedding(position_ids)

        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Pass through encoder layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        # Classification (use [CLS] token equivalent - first token)
        pooled = hidden_states[:, 0, :]
        logits = self.classifier(pooled)
        return logits


def get_bert_small(input_shape: tuple) -> nn.Module:
    """Create BERT-small model."""
    # input_shape should be (batch_size, seq_len) for token IDs
    model = BertSmallTransformer(
        vocab_size=30522,
        hidden_size=768,
        num_heads=12,
        num_layers=6,
        max_seq_length=input_shape[1] if len(input_shape) > 1 else 128,
    )
    return model


def run_benchmark(
    model_name: str,
    model_fn,
    input_shape: tuple,
    batch_size: int,
    iterations: int,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run benchmark for a single model.

    Args:
        model_name: Name of the model
        model_fn: Function that returns the model
        input_shape: Input tensor shape
        batch_size: Batch size for training
        iterations: Number of iterations to profile
        output_dir: Directory to save results

    Returns:
        Dictionary with profiling results
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {model_name}")
    print(f"  Input shape: {input_shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iterations}")
    print(f"{'=' * 60}\n")

    # Create model
    model = model_fn()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Initialize profiler
    profiler = BaseProfiler(
        model=model,
        input_shape=input_shape,
        device="cuda",
        use_nsight=False,  # Disable Nsight by default (requires system config)
    )

    # Run training profile
    print(f"  Profiling training...")
    training_results = profiler.profile_training(iterations=iterations)

    # Run inference profile
    print(f"  Profiling inference...")
    inference_results = profiler.profile_inference(iterations=iterations)

    # Combine results
    results = {
        "model": model_name,
        "batch_size": batch_size,
        "num_parameters": num_params,
        "training": training_results,
        "inference": inference_results,
    }

    # Export to JSON
    output_path = Path(output_dir) / f"{model_name}_baseline.json"
    profiler.export_results(str(output_path))

    # Print summary
    print(f"\n  Results:")
    print(f"    Training FPS: {training_results['fps']:.2f}")
    print(f"    Training memory peak: {training_results['memory_peak_mb']:.2f} MB")
    print(f"    Inference FPS: {inference_results['fps']:.2f}")
    print(f"    Inference memory peak: {inference_results['memory_peak_mb']:.2f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline benchmarking for PyTorch models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["resnet50", "bert-small"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for aggregated results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./logs",
        help="Directory to save per-model results",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. This benchmark requires a CUDA-enabled GPU."
        )
        sys.exit(1)

    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Model configurations
    model_configs = {
        "resnet50": {
            "fn": lambda: get_resnet50((args.batch_size, 3, 224, 224)),
            "input_shape": (args.batch_size, 3, 224, 224),
        },
        "bert-small": {
            "fn": lambda: get_bert_small((args.batch_size, 128)),
            "input_shape": (args.batch_size, 128),  # Token IDs
        },
    }

    # Run benchmarks
    all_results = {}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        if model_name not in model_configs:
            print(f"Warning: Unknown model '{model_name}'. Skipping.")
            print(f"Available models: {list(model_configs.keys())}")
            continue

        config = model_configs[model_name]
        try:
            results = run_benchmark(
                model_name=model_name,
                model_fn=config["fn"],
                input_shape=config["input_shape"],
                batch_size=args.batch_size,
                iterations=args.iterations,
                output_dir=args.output_dir,
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"ERROR: Benchmark failed for {model_name}: {e}")
            import traceback

            traceback.print_exc()

    # Save aggregated results
    aggregated_path = Path(args.output_dir) / args.output
    with open(aggregated_path, "w") as f:
        json.dump(
            {
                "config": {
                    "batch_size": args.batch_size,
                    "iterations": args.iterations,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print(f"All benchmarks complete!")
    print(f"Aggregated results saved to: {aggregated_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
