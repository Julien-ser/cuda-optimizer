#!/usr/bin/env python3
"""
Benchmark for custom CUDA kernels vs native PyTorch operations.
Demonstrates the performance improvement of fused LayerNorm + Activation.
"""

import torch
import torch.nn as nn
import time
import argparse
from typing import Callable, Tuple
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

try:
    from cuda_optimizer.kernels import CustomOps

    CUSTOM_OPS_AVAILABLE = CustomOps.is_available()
except ImportError:
    CUSTOM_OPS_AVAILABLE = False
    print("Warning: CustomOps not available. Only running native benchmark.")


def benchmark_operation(
    fn: Callable, *args, num_warmup: int = 10, num_iters: int = 100, **kwargs
) -> float:
    """
    Benchmark a function and return average execution time in milliseconds.
    """
    # Warmup
    for _ in range(num_warmup):
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) * 1000 / num_iters
    return avg_time_ms


def create_test_data(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test data for LayerNorm benchmark.
    Returns: (input, weight, bias)
    """
    shape = (batch_size, seq_length, hidden_size)
    input = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    bias = torch.zeros(hidden_size, dtype=dtype, device="cuda")
    return input, weight, bias


def native_layernorm_gelu(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Native PyTorch implementation: LayerNorm + GELU
    """
    # LayerNorm over last dimension
    normalized = nn.functional.layer_norm(
        input, input.shape[-1:], weight=weight, bias=bias, eps=eps
    )
    # GELU activation
    return nn.functional.gelu(normalized)


def native_layernorm_relu(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Native PyTorch implementation: LayerNorm + ReLU
    """
    normalized = nn.functional.layer_norm(
        input, input.shape[-1:], weight=weight, bias=bias, eps=eps
    )
    return torch.relu(normalized)


def custom_layernorm_gelu(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Custom fused kernel"""
    return CustomOps.fused_layernorm_gelu(input, weight, bias, eps)


def custom_layernorm_relu(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Custom fused kernel"""
    return CustomOps.fused_layernorm_relu(input, weight, bias, eps)


def run_benchmark(
    batch_sizes: list = [8, 16, 32],
    seq_lengths: list = [128, 256, 512, 1024],
    hidden_sizes: list = [768, 1024, 2048],
    dtype: torch.dtype = torch.float16,
):
    """
    Run comprehensive benchmark comparing native vs fused operations.
    """
    print("=" * 80)
    print("CUDA Custom Ops Benchmark")
    print("=" * 80)
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    if not CUSTOM_OPS_AVAILABLE:
        print("WARNING: Custom ops not available. Skipping fused benchmarks.")
        print(
            "Build the CUDA extension first by running: python setup.py build_ext --inplace"
        )
        return

    print(
        f"{'Batch':<6} {'Seq':<6} {'Hidden':<8} {'Native (ms)':<12} {'Fused (ms)':<12} {'Speedup':<8}"
    )
    print("-" * 80)

    for batch in batch_sizes:
        for seq in seq_lengths:
            for hidden in hidden_sizes:
                input, weight, bias = create_test_data(batch, seq, hidden, dtype)

                # Test GELU
                native_time = benchmark_operation(
                    native_layernorm_gelu, input, weight, bias
                )
                fused_time = benchmark_operation(
                    custom_layernorm_gelu, input, weight, bias
                )

                speedup = native_time / fused_time if fused_time > 0 else float("inf")

                marker = " ***" if speedup >= 1.2 else ""
                print(
                    f"{batch:<6} {seq:<6} {hidden:<8} {native_time:<12.4f} {fused_time:<12.4f} {speedup:<8.2f}x{marker}"
                )

    print()
    print("Target: 20%+ speedup (1.20x) indicated by ***")
    print("=" * 80)


def verify_correctness():
    """
    Verify that custom ops produce identical results to native operations.
    """
    if not CUSTOM_OPS_AVAILABLE:
        print("Custom ops not available. Skipping correctness check.")
        return

    print("\nCorrectness Verification")
    print("-" * 80)

    torch.manual_seed(42)
    batch, seq, hidden = 4, 256, 1024
    input, weight, bias = create_test_data(batch, seq, hidden, torch.float32)

    # Test GELU
    native_out = native_layernorm_gelu(input, weight, bias)
    custom_out = custom_layernorm_gelu(input, weight, bias)
    diff_gelu = torch.max(torch.abs(native_out - custom_out)).item()
    print(f"GELU max difference: {diff_gelu:.2e}")
    assert diff_gelu < 1e-3, f"GELU outputs differ too much: {diff_gelu}"

    # Test ReLU
    native_relu = native_layernorm_relu(input, weight, bias)
    custom_relu = custom_layernorm_relu(input, weight, bias)
    diff_relu = torch.max(torch.abs(native_relu - custom_relu)).item()
    print(f"ReLU max difference: {diff_relu:.2e}")
    assert diff_relu < 1e-3, f"ReLU outputs differ too much: {diff_relu}"

    print("All correctness checks passed!")
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark custom CUDA kernels")
    parser.add_argument(
        "--verify", action="store_true", help="Run correctness verification"
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if args.verify:
        verify_correctness()
    else:
        run_benchmark(dtype=dtype)
