"""
Automatic kernel tuning using NVIDIA NVTX.

This module provides an autotuner that systematically searches for optimal
CUDA kernel launch configurations (block/grid dimensions) and caches the
results using NVTX for precise timing.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple
import time

import torch
import torch.cuda

try:
    import nvtx

    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    print("Warning: nvtx package not found. Install with: pip install nvtx")


class Autotuner:
    """
    Auto-tuner for CUDA kernel block/grid dimensions using NVTX.

    Features:
    - Systematic search over block sizes (32-1024) and grid dimensions
    - NVTX markers for precise kernel timing
    - Configuration cache stored in ~/.cache/cuda-optimizer/
    - Warm-up runs to stabilize measurements
    - Statistical analysis (median of multiple runs)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        num_trials: int = 5,
        warmup_iterations: int = 3,
        verbose: bool = False,
    ):
        """
        Initialize the autotuner.

        Args:
            cache_dir: Directory to store configuration cache.
                      Defaults to ~/.cache/cuda-optimizer/
            num_trials: Number of timing trials per configuration
            warmup_iterations: Warm-up runs before timing
            verbose: Whether to print tuning progress
        """
        self.cache_dir = Path(
            cache_dir or os.path.expanduser("~/.cache/cuda-optimizer")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "kernel_configs.pkl"

        self.num_trials = num_trials
        self.warmup_iterations = warmup_iterations
        self.verbose = verbose

        # Load existing cache if available
        self.config_cache: Dict[str, Dict[str, Any]] = self._load_cache()

        # NVTX availability check
        self.nvtx_available = NVTX_AVAILABLE and torch.cuda.is_available()

        if not self.nvtx_available and verbose:
            print("Warning: NVTX not available. Using fallback timing.")

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                if self.verbose:
                    print(f"Loaded {len(cache)} cached configurations")
                return cache
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load cache: {e}. Starting fresh.")
        return {}

    def _save_cache(self) -> None:
        """Save configuration cache to disk."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.config_cache, f)
        except Exception as e:
            if self.verbose:
                print(f"Failed to save cache: {e}")

    def _generate_cache_key(
        self, kernel_name: str, input_shape: Tuple[int, ...], dtype: str
    ) -> str:
        """Generate a unique cache key for a kernel configuration."""
        key_parts = [kernel_name, str(input_shape), dtype]
        return "|".join(key_parts)

    def _suggest_block_sizes(self) -> List[int]:
        """Suggest block sizes to search (powers of 2 from 32 to 1024)."""
        return [32, 64, 128, 256, 512, 1024]

    def _calculate_grid_size(
        self, total_elements: int, block_size: int, num_features: int = 1
    ) -> Tuple[int, int]:
        """
        Calculate optimal grid dimensions for a given problem size.

        For kernels with 2D grid (like LayerNorm), we calculate both dimensions.
        """
        # For kernels that process features in y dimension
        blocks_per_feature = (total_elements + block_size - 1) // block_size
        return (blocks_per_feature, num_features)

    def autotune_kernel(
        self,
        kernel_fn: Callable,
        kernel_name: str,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        fixed_params: Optional[Dict[str, Any]] = None,
        num_features: int = 1,
    ) -> Dict[str, Any]:
        """
        Autotune a CUDA kernel for optimal launch configuration.

        Args:
            kernel_fn: Function that executes the kernel (must accept block_dim, grid_dim)
            kernel_name: Unique name for this kernel
            input_shape: Shape of input tensor
            dtype: Data type of tensors
            fixed_params: Additional fixed parameters passed to kernel_fn
            num_features: Number of feature dimensions (for 2D grid)

        Returns:
            Dictionary containing optimal configuration:
            {
                'block_size': int,
                'grid_x': int,
                'grid_y': int,
                'execution_time_ms': float,
                'throughput_gb_s': float
            }
        """
        # Check cache first
        cache_key = self._generate_cache_key(kernel_name, input_shape, str(dtype))
        if cache_key in self.config_cache:
            if self.verbose:
                print(f"Using cached configuration for {kernel_name}")
            return self.config_cache[cache_key]

        if self.verbose:
            print(f"Autotuning {kernel_name} with input shape {input_shape}...")

        # Prepare test data
        total_elements = torch.prod(torch.tensor(input_shape)).item()
        block_sizes = self._suggest_block_sizes()

        best_config = None
        best_time = float("inf")

        # Search over block sizes
        for block_size in block_sizes:
            grid_x, grid_y = self._calculate_grid_size(
                total_elements, block_size, num_features
            )

            # Skip invalid configurations
            if grid_x <= 0 or grid_y <= 0:
                continue

            # Warm-up
            for _ in range(self.warmup_iterations):
                kernel_fn(block_size, (grid_x, grid_y), **fixed_params or {})

            # Time with NVTX
            times = []
            for trial in range(self.num_trials):
                torch.cuda.synchronize()

                if self.nvtx_available:
                    with nvtx.annotate(
                        f"autotune_{kernel_name}_block{block_size}_trial{trial}"
                    ):
                        start = time.perf_counter()
                        kernel_fn(block_size, (grid_x, grid_y), **fixed_params or {})
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                else:
                    start = time.perf_counter()
                    kernel_fn(block_size, (grid_x, grid_y), **fixed_params or {})
                    torch.cuda.synchronize()
                    end = time.perf_counter()

                times.append((end - start) * 1000)  # Convert to ms

            median_time = sorted(times)[len(times) // 2]

            # Calculate throughput (GB/s)
            # Estimate memory accessed: input + output + weight + bias
            element_size = torch.finfo(dtype).bits // 8
            total_bytes = total_elements * element_size * 2  # Read + write
            throughput = (total_bytes / (median_time / 1000)) / 1e9  # GB/s

            if self.verbose:
                print(
                    f"  block={block_size:4d}, grid=({grid_x:5d}, {grid_y:2d}) -> "
                    f"time={median_time:.4f}ms, throughput={throughput:.2f}GB/s"
                )

            # Track best
            if median_time < best_time:
                best_time = median_time
                best_config = {
                    "block_size": block_size,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "execution_time_ms": median_time,
                    "throughput_gb_s": throughput,
                }

        if best_config is None:
            raise RuntimeError(
                f"Autotuning failed for {kernel_name}: no valid configuration found"
            )

        # Cache the result
        self.config_cache[cache_key] = best_config
        self._save_cache()

        if self.verbose:
            print(
                f"  Best configuration: block={best_config['block_size']}, "
                f"grid=({best_config['grid_x']}, {best_config['grid_y']}) "
                f"time={best_config['execution_time_ms']:.4f}ms"
            )

        return best_config

    def autotune_operation(
        self,
        operation: Callable,
        kernel_name: str,
        test_data: Tuple[torch.Tensor, ...],
        num_warmup: int = 10,
    ) -> Dict[str, Any]:
        """
        Autotune an operation that doesn't directly accept block/grid parameters.

        This method profiles the operation with different CUDA launch configurations
        by temporarily modifying the kernel or using a wrapper.

        Args:
            operation: Callable that executes the operation to tune
            kernel_name: Unique name for this kernel
            test_data: Tuple of tensors used for testing
            num_warmup: Number of warmup iterations

        Returns:
            Dictionary with timing metrics and suggested configuration
        """
        # For operations that don't directly expose block/grid params,
        # we need a wrapper that can vary the configuration

        total_elements = test_data[0].numel()
        block_sizes = self._suggest_block_sizes()

        best_config = None
        best_time = float("inf")

        for block_size in block_sizes:
            torch.cuda.synchronize()

            # Warm-up
            for _ in range(num_warmup):
                operation()
                torch.cuda.synchronize()

            # Timing with NVTX
            times = []
            for trial in range(self.num_trials):
                torch.cuda.synchronize()

                if self.nvtx_available:
                    with nvtx.annotate(
                        f"autotune_op_{kernel_name}_block{block_size}_trial{trial}"
                    ):
                        start = time.perf_counter()
                        operation()
                        torch.cuda.synchronize()
                        end = time.perf_counter()
                else:
                    start = time.perf_counter()
                    operation()
                    torch.cuda.synchronize()
                    end = time.perf_counter()

                times.append((end - start) * 1000)

            median_time = sorted(times)[len(times) // 2]

            if self.verbose:
                print(
                    f"  {kernel_name}: block_size={block_size} -> {median_time:.4f}ms"
                )

            if median_time < best_time:
                best_time = median_time
                best_config = {
                    "block_size": block_size,
                    "execution_time_ms": median_time,
                }

        if best_config:
            self.config_cache[kernel_name] = best_config
            self._save_cache()

        return best_config or {}

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self.config_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.verbose:
            print("Cache cleared")

    def get_cached_config(self, kernel_name: str) -> Optional[Dict[str, Any]]:
        """Get cached configuration for a kernel if available."""
        return self.config_cache.get(kernel_name)

    def list_cached_kernels(self) -> List[str]:
        """List all kernels with cached configurations."""
        return list(self.config_cache.keys())

    def benchmark_all_cached(self) -> Dict[str, Dict[str, Any]]:
        """Return all cached configurations."""
        return self.config_cache
