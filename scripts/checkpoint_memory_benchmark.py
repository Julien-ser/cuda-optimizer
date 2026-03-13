#!/usr/bin/env python3
"""
Checkpoint Memory Benchmark

Measures memory savings achieved by gradient checkpointing.
Demonstrates >50% memory reduction on standard models.

Run: python scripts/checkpoint_memory_benchmark.py --model resnet50
"""

import argparse
import json
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

# Ensure we can import cuda_optimizer when run as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cuda_optimizer.checkpoint import SelectiveCheckpoint, CheckpointCompiler


def get_resnet50():
    """Create ResNet50 model."""
    from torchvision import models

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


class BertSmallTransformer(nn.Module):
    """BERT-small from baseline script."""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_heads=12,
        num_layers=6,
        max_seq_length=128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
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
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        token_embeds = self.embedding(input_ids)
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        x = self.layer_norm(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        pooled = x[:, 0, :]
        return self.classifier(pooled)


def get_bert_small():
    return BertSmallTransformer()


def profile_peak_memory(model, input_tensor, target, num_iters=10):
    """Profile peak memory during training."""
    torch.cuda.reset_peak_memory_stats()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warm-up
    for _ in range(3):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()

    # Actual profiling
    for _ in range(num_iters):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    return peak / (1024**2)  # MB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["resnet50", "bert-small"], default="resnet50"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--output", type=str, default="./logs/checkpoint_benchmark.json"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    # Model setup
    if args.model == "resnet50":
        model_fn = get_resnet50
        input_shape = (args.batch_size, 3, 224, 224)
    else:
        model_fn = get_bert_small
        input_shape = (args.batch_size, 128)

    # Create input and target
    if len(input_shape) == 4:
        input_tensor = torch.randn(input_shape, device=device)
        target = torch.randint(0, 10, (args.batch_size,), device=device)
    else:
        input_tensor = torch.randint(0, 30522, input_shape, device=device)
        target = torch.randint(0, 10, (args.batch_size,), device=device)

    # Baseline
    print(f"\n=== Baseline (no checkpoint) ===")
    model_baseline = model_fn()
    baseline_peak = profile_peak_memory(
        model_baseline, input_tensor, target, args.iterations
    )
    print(f"Baseline peak memory: {baseline_peak:.2f} MB")

    # Checkpointed
    print(f"\n=== With Gradient Checkpointing ===")
    model_ckpt = model_fn()

    selector = SelectiveCheckpoint()
    compiler = CheckpointCompiler(selector)

    # Select layers based on model type
    if args.model == "resnet50":
        # Checkpoint all conv and linear layers except the final fc
        for name, module in model_ckpt.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if "fc" not in name:  # exclude final fully connected
                    selector.select_layers([module])
    else:
        # BERT: checkpoint all transformer encoder layers
        for layer in model_ckpt.encoder_layers:
            selector.select_layers([layer])

    compiler.compile(model_ckpt)
    checkpointed_peak = profile_peak_memory(
        model_ckpt, input_tensor, target, args.iterations
    )
    print(f"Checkpointed peak memory: {checkpointed_peak:.2f} MB")

    # Results
    reduction = baseline_peak - checkpointed_peak
    reduction_pct = (reduction / baseline_peak) * 100 if baseline_peak > 0 else 0

    print(f"\n=== Results ===")
    print(f"Memory reduction: {reduction:.2f} MB ({reduction_pct:.1f}%)")

    results = {
        "model": args.model,
        "batch_size": args.batch_size,
        "baseline_memory_mb": baseline_peak,
        "checkpointed_memory_mb": checkpointed_peak,
        "reduction_mb": reduction,
        "reduction_percent": reduction_pct,
        "target_achieved": reduction_pct >= 50,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    if reduction_pct >= 50:
        print("✓ Target achieved: >= 50% memory reduction")
        sys.exit(0)
    else:
        print("✗ Target not achieved: < 50% memory reduction")
        sys.exit(1)


if __name__ == "__main__":
    main()
