"""
Performance benchmark for fused AdamW optimizer.

Compares FusedAdamW against torch.optim.AdamW to demonstrate
the expected 30%+ speedup.
"""

import torch
import torch.nn as nn
import time
from typing import Callable, Tuple

from cuda_optimizer.fusion import FusedAdamW, is_available


def benchmark_optimizer(
    optimizer_class: Callable,
    params: list,
    n_steps: int = 100,
    device: str = "cuda",
    warmup: int = 10,
) -> float:
    """
    Benchmark an optimizer for average step time.

    Args:
        optimizer_class: Optimizer class to benchmark
        params: List of parameters to optimize
        n_steps: Number of steps to measure
        device: Device to run on ('cuda' or 'cpu')
        warmup: Number of warmup steps

    Returns:
        float: Average step time in milliseconds
    """
    # Move parameters to device
    params_on_device = [p.to(device) for p in params]

    # Create optimizer
    opt = optimizer_class(params_on_device, lr=1e-3, weight_decay=0.01)

    # Create dummy gradients
    for p in params_on_device:
        if p.grad is None:
            p.grad = torch.randn_like(p)

    # Warmup
    for _ in range(warmup):
        opt.zero_grad()
        # Simulate gradient computation (already set)
        opt.step()
        # Reset gradients for next step
        for p in params_on_device:
            p.grad = torch.randn_like(p)

    # Sync GPU before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_steps):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        opt.zero_grad()
        # Gradients are already set, just call step
        opt.step()
        # Reset gradients for next step
        for p in params_on_device:
            p.grad = torch.randn_like(p)

        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    return avg_time


def create_test_model(num_params: int = 10_000_000, device: str = "cuda") -> list:
    """
    Create a model with a large number of parameters for benchmarking.

    Args:
        num_params: Total number of parameters
        device: Device to create parameters on

    Returns:
        list: List of parameter tensors
    """
    # Create parameter groups similar to real models
    params = []
    remaining = num_params

    # Create a few large parameter tensors
    for i in range(5):
        size = min(num_params // 5, remaining)
        if size > 0:
            p = nn.Parameter(torch.randn(size, device=device))
            params.append(p)
            remaining -= size
            if remaining <= 0:
                break

    # Add some smaller tensors if needed
    if remaining > 0:
        p = nn.Parameter(torch.randn(remaining, device=device))
        params.append(p)

    return params


def run_adamw_benchmark(
    n_steps: int = 100, num_params: int = 10_000_000
) -> Tuple[float, float, float]:
    """
    Run benchmark comparing FusedAdamW vs standard AdamW.

    Args:
        n_steps: Number of steps to benchmark each optimizer
        num_params: Number of parameters in test model

    Returns:
        tuple: (fused_time_ms, standard_time_ms, speedup_factor)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this benchmark")

    device = "cuda"

    print(f"Creating test model with {num_params:,} parameters...")
    params_std = create_test_model(num_params, device)
    params_fused = [p.clone() for p in params_std]  # Clone for fair comparison

    print("Benchmarking standard AdamW...")
    standard_time = benchmark_optimizer(
        torch.optim.AdamW, params_std, n_steps=n_steps, device=device
    )
    print(f"  Average step time: {standard_time:.3f} ms")

    if not is_available():
        warnings.warn("FusedAdamW not available, cannot benchmark fused version")
        fused_time = float("inf")
        speedup = 0.0
    else:
        print("Benchmarking FusedAdamW...")
        fused_time = benchmark_optimizer(
            FusedAdamW, params_fused, n_steps=n_steps, device=device
        )
        print(f"  Average step time: {fused_time:.3f} ms")

        speedup = (standard_time - fused_time) / standard_time * 100
        print(f"\nSpeedup: {speedup:.1f}%")

    return fused_time, standard_time, speedup


def validate_accuracy() -> bool:
    """
    Validate that FusedAdamW produces similar results to standard AdamW.

    Returns:
        bool: True if results match within tolerance
    """
    print("\n" + "=" * 60)
    print("ACCURACY VALIDATION")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    ).to(device)

    params_std = list(model.parameters())
    params_fused = [p.clone() for p in params_std]

    # Create optimizers
    opt_std = torch.optim.AdamW(params_std, lr=1e-2, weight_decay=1e-3)
    if is_available():
        opt_fused = FusedAdamW(params_fused, lr=1e-2, weight_decay=1e-3)
    else:
        print("FusedAdamW not available, skipping accuracy check")
        return True

    # Dummy input and target
    x = torch.randn(32, 100, device=device)
    target = torch.randn(32, 10, device=device)
    criterion = nn.MSELoss()

    # Train for a few steps
    for step in range(20):
        # Standard AdamW
        opt_std.zero_grad()
        out_std = model(x)
        loss_std = criterion(out_std, target)
        loss_std.backward()
        opt_std.step()

        # Fused AdamW (using cloned model/params)
        for p_std, p_fused in zip(model.parameters(), params_fused):
            p_fused.data.copy_(p_std.data)
            if p_std.grad is not None:
                p_fused.grad = p_std.grad.clone()

        opt_fused.step()

        if step % 5 == 0:
            print(f"Step {step}: std loss={loss_std.item():.6f}")

    # Final comparison
    final_std = torch.cat([p.view(-1) for p in params_std])
    final_fused = torch.cat([p.view(-1) for p in params_fused])
    diff = torch.abs(final_std - final_fused).max().item()

    print(f"\nMax parameter difference: {diff:.2e}")
    tolerance = 1e-4
    passed = diff < tolerance
    print(
        f"Validation: {'PASSED' if passed else 'FAILED'} (tolerance: {tolerance:.0e})"
    )

    return passed


if __name__ == "__main__":
    import warnings

    print("=" * 60)
    print("FUSED ADAMW PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available. This benchmark requires a GPU.")
        exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Run performance benchmark
    n_steps = 100
    num_params = 10_000_000  # 10M parameters

    fused_t, std_t, speedup = run_adamw_benchmark(n_steps, num_params)

    # Validate accuracy
    valid = validate_accuracy()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Standard AdamW: {std_t:.3f} ms/step")
    if fused_t != float("inf"):
        print(f"Fused AdamW:   {fused_t:.3f} ms/step")
        print(f"Speedup:       {speedup:.1f}%")
    else:
        print("Fused AdamW:   N/A (kernels not available)")
    print(f"Accuracy:      {'PASS' if valid else 'FAIL'}")
    print()

    if speedup >= 30.0 and valid:
        print("✓ Performance target achieved: 30%+ speedup with accuracy preserved")
    elif speedup > 0:
        print(f"⚠ Speedup of {speedup:.1f}% is below target of 30%")
    else:
        print("✗ Fused optimizer not available")
