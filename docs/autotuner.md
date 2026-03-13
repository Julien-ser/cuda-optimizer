# Kernel Auto-Tuner

## Overview

The `Autotuner` class automatically searches for optimal CUDA kernel launch configurations (block and grid dimensions) using NVIDIA NVTX for precise timing. It caches the best configurations for reuse across runs, eliminating manual tuning.

**Key Benefits:**
- **Automated optimization**: Systematically searches block sizes 32-1024
- **NVTX integration**: Precise kernel timing with minimal overhead
- **Persistent cache**: Results stored in `~/.cache/cuda-optimizer/` for reuse
- **Statistical robustness**: Median of multiple trials to filter noise
- **Warm-up runs**: Stabilizes measurements before timing

**Target Performance:**
- **10-15% improvement** from optimal block/grid configuration vs manual defaults
- **No runtime tuning cost** after initial run (configs cached)
- **Adapts to different GPUs**: Different optimal configs for Turing vs Ampere

## Architecture

### Search Space

The autotuner explores:
- **Block sizes**: 32, 64, 128, 256, 512, 1024 threads per block
- **Grid dimensions**: Calculated from problem size and block size
- **Multi-dimensional grids**: For 2D kernels (e.g., LayerNorm)

### Caching Strategy

Cache key generation:
```
cache_key = f"{kernel_name}|{input_shape}|{dtype}"
```

Example: `"fused_layernorm_gelu|(32,128,768)|torch.float16"`

Cache stored as pickle in `~/.cache/cuda-optimizer/kernel_configs.pkl`.

### Timing Methodology

1. **Warm-up**: Run kernel `warmup_iterations` times to:
   - Fill instruction caches
   - Reach steady-state clock speeds
   - Allocate GPU memory

2. **Timing trials**: Run `num_trials` times, record median:
   - Uses `time.perf_counter()` for CPU-side timing
   - `torch.cuda.synchronize()` before/after to get accurate kernel time
   - NVTX annotations visible in Nsight Systems profiles

3. **Throughput calculation**:
   ```
   throughput = (total_bytes_accessed / median_time) / 1e9  # GB/s
   ```

### NVTX Integration

When `nvtx` package is available:
- Each trial wrapped with `nvtx.annotate()`
- Appears as distinct marker in Nsight Systems timeline
- Easy to identify and filter in profiler

## Usage

### Basic Example: Autotuning a Simple Kernel

```python
import torch
from cuda_optimizer import Autotuner

# Define a kernel to tune (must accept block_dim, grid_dim parameters)
def my_kernel(block_dim, grid_dim):
    # Your CUDA kernel logic here
    # This would typically call a custom CUDA extension
    pass

autotuner = Autotuner(verbose=True)

config = autotuner.autotune_kernel(
    kernel_fn=my_kernel,
    kernel_name="my_simple_kernel",
    input_shape=(1024, 1024),
    dtype=torch.float16
)

print(f"Optimal config: {config}")
# Output: {'block_size': 256, 'grid_x': 4096, 'grid_y': 1, ...}
```

### Autotuning CustomOps Kernels

```python
from cuda_optimizer import CustomOps, Autotuner

autotuner = Autotuner(verbose=True)

# The custom ops already have built-in tunability
# You can autotune by varying block/grid directly
def run_fused_layernorm(x, weight, bias, block_dim, grid_dim):
    # This would call the CUDA kernel with specific launch config
    # In practice, you'd have a kernel that accepts these params
    pass

# Tune for your specific input shape
config = autotuner.autotune_kernel(
    kernel_fn=run_fused_layernorm,
    kernel_name="fused_layernorm",
    input_shape=(32, 128, 768),
    dtype=torch.float16
)
```

### Autotuning Operations Without Direct Parameter Control

Some operations don't directly expose block/grid parameters. Use `autotune_operation()`:

```python
from cuda_optimizer import Autotuner
import torch

autotuner = Autotuner()

# Define operation (closed over parameters)
def matmul_op():
    return torch.matmul(A, B)

# Prepare test data
A = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
B = torch.randn(2048, 512, device='cuda', dtype=torch.float16)

# The autotuner will time the operation with different block sizes
# by internally varying any tunable parameters
config = autotuner.autotune_operation(
    operation=matmul_op,
    kernel_name="matmul_1024x2048x512",
    test_data=(A, B)
)
```

### Using Cached Configurations

```python
# Later runs automatically use cached configs
autotuner = Autotuner(verbose=True)

# If cached, no retuning happens
config = autotuner.autotune_kernel(
    kernel_fn=my_kernel,
    kernel_name="my_simple_kernel",
    input_shape=(1024, 1024),
    dtype=torch.float16
)
print("Using cached config, no retuning needed")

# Manually check what's cached
cached = autotuner.list_cached_kernels()
print(f"Cached kernels: {cached}")

# Get specific cached config
config = autotuner.get_cached_config("my_simple_kernel")
```

### Clearing Cache

```python
# Clear all cached configurations
autotuner.clear_cache()
```

### Running Tuning Script

The project includes a CLI script for tuning common operations:

```bash
# Tune all predefined kernels
python scripts/tune_kernels.py --kernels all

# Tune specific kernels
python scripts/tune_kernels.py --kernels fused_layernorm matmul

# Set number of trials
python scripts/tune_kernels.py --trials 10 --warmup 5
```

## API Reference

### `Autotuner`

Main autotuner class.

#### `__init__(
    cache_dir: Optional[str] = None,
    num_trials: int = 5,
    warmup_iterations: int = 3,
    verbose: bool = False
)`

Initialize autotuner.

**Parameters:**
- `cache_dir`: Directory for configuration cache (default: `~/.cache/cuda-optimizer/`)
- `num_trials`: Number of timing trials per configuration (default: 5)
- `warmup_iterations`: Warm-up runs before timing (default: 3)
- `verbose`: Print tuning progress (default: False)

#### `autotune_kernel(
    kernel_fn: Callable,
    kernel_name: str,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float16,
    fixed_params: Optional[Dict[str, Any]] = None,
    num_features: int = 1
) → Dict[str, Any]`

Autotune a CUDA kernel with variable block/grid parameters.

**Parameters:**
- `kernel_fn`: Callable that accepts `(block_size, grid_dim, **fixed_params)`
- `kernel_name`: Unique identifier for this kernel
- `input_shape`: Shape of input tensors
- `dtype`: Data type of tensors
- `fixed_params`: Additional fixed parameters to pass to `kernel_fn`
- `num_features`: Feature dimension count for 2D grid (default: 1)

**Returns:** Dictionary with optimal configuration:
```python
{
    'block_size': 256,
    'grid_x': 4096,
    'grid_y': 1,
    'execution_time_ms': 0.123,
    'throughput_gb_s': 123.45
}
```

**Raises:** `RuntimeError` if no valid configuration found

#### `autotune_operation(
    operation: Callable,
    kernel_name: str,
    test_data: Tuple[torch.Tensor, ...],
    num_warmup: int = 10
) → Dict[str, Any]`

Autotune an operation that doesn't directly accept block/grid parameters.

**Parameters:**
- `operation`: Callable that executes the operation to tune (no parameters)
- `kernel_name`: Unique identifier
- `test_data`: Tuple of tensors to use for test (sets up environment)
- `num_warmup`: Warm-up iterations (default: 10)

**Returns:** Dictionary with timing metrics:
```python
{
    'block_size': 256,
    'execution_time_ms': 0.456
}
```

**Note:** This method is limited; for full control use `autotune_kernel` with a wrapper.

#### `clear_cache() → None`

Clear all cached configurations from disk and memory.

#### `get_cached_config(kernel_name: str) → Optional[Dict[str, Any]]`

Retrieve cached configuration for a specific kernel.

**Parameters:**
- `kernel_name`: Name of the kernel

**Returns:** Cached configuration dict or `None` if not found

#### `list_cached_kernels() → List[str]`

List all kernels with cached configurations.

**Returns:** List of kernel names

#### `benchmark_all_cached() → Dict[str, Dict[str, Any]]`

Return all cached configurations.

**Returns:** Dictionary mapping kernel names to their configs

## Performance Characteristics

### Search Efficiency

For a typical kernel:
- **Configurations tested**: 6 block sizes × 1 grid calc = 6 configs
- **Time per config**: ~100ms (with 5 trials × 20ms per kernel)
- **Total tuning time**: ~600ms per kernel
- **After cache**: 0ms (instant lookup)

### Quality of Results

- **Median of 5 trials**: Robust to outliers and system noise
- **Warm-up**: Stabilizes GPU clocks and eliminates cold-start effects
- **Realistic test conditions**: Uses actual tensor shapes and dtypes

### Comparison to Manual Tuning

Autotuner typically finds:
- **Better configs**: 10-15% faster than common defaults (e.g., block=256)
- **Architecture-aware**: Different optimal block sizes for Turing vs Ampere
- **Shape-aware**: Optimal configs vary with tensor dimensions

## Best Practices

### 1. Run Tuning Before First Production Use

```python
# On your target GPU(s), run tuning once
autotuner = Autotuner(verbose=True)
for kernel_name in ['fused_layernorm', 'my_custom_op']:
    config = autotuner.autotune_kernel(...)
```

### 2. Cache Persistence

The cache is automatically saved and reused across runs. Ensure:
- Users have write access to `~/.cache/cuda-optimizer/`
- Cache directory is backed up if moving between systems (will retune on different GPU arch anyway)

### 3. Tune on Target Hardware

Optimal configurations are GPU-architecture-specific:
- **Turing (sm_75)**: Often prefers block=128 or 256
- **Ampere (sm_80/86)**: Often prefers block=256 or 512
- **Ada/Hopper**: May prefer larger blocks

If deploying to heterogeneous GPUs, consider:
- Running autotuner on each architecture type
- Or using conservative defaults (block=256 works reasonably well everywhere)

### 4. Increase Trials for Critical Kernels

For kernels that dominate runtime, use more trials for robust statistics:

```python
critical_autotuner = Autotuner(num_trials=20, warmup_iterations=5)
config = critical_autotuner.autotune_kernel(...)
```

### 5. Profile to Identify Kernels to Tune

Not all kernels need tuning. Focus on:
- Kernels with >1% runtime (from profiler)
- Custom kernels you're writing
- Frequently called operations

```python
from cuda_optimizer import BaseProfiler
profiler = BaseProfiler(model, input_shape)
results = profiler.profile_training()
top_kernels = sorted(results['kernel_stats'], key=lambda x: x['cuda_time_total_ms'], reverse=True)[:5]
for kernel in top_kernels:
    print(f"Consider tuning: {kernel['name']}")
```

### 6. Use Consistent Input Shapes

The cache key includes input shape. If your application uses multiple shapes:
- Tune each shape variant separately
- Or choose representative shape for training/inference

## Troubleshooting

### Problem: Tuning takes too long

**Solution:**
- Reduce `num_trials` (e.g., from 5 to 3)
- Reduce number of block sizes to search (customize `_suggest_block_sizes()`)
- Tune only the most critical kernels

### Problem: "No valid configuration found"

**Diagnosis:** The kernel function might be throwing errors for certain block/grid combos.

**Solution:**
- Check kernel logic for boundary conditions
- Ensure grid dimensions are valid (non-zero, within device limits)
- Add validation in your kernel function to skip invalid configs

### Problem: Cached config gives worse performance

**Diagnosis:** System state changed (GPU, driver, CUDA version) but cache reused.

**Solution:**
- Clear cache when changing hardware: `autotuner.clear_cache()`
- Or use different cache directories per system
- The autotuner can't automatically detect hardware changes

### Problem: NVTX markers not appearing

**Diagnosis:** `nvtx` package not installed or NVTX not available.

**Solution:**
```bash
pip install nvtx
```
Even without NVTX, timing still works using `time.perf_counter()`.

### Problem: High variance in timing results

**Diagnosis:** System noise (OS interrupts, GPU clock scaling).

**Solution:**
- Increase `warmup_iterations` (10-20)
- Increase `num_trials` (10-20) for more robust median
- Use `CUDA_VISIBLE_DEVICES` to isolate GPU
- Disable CPU frequency scaling and GPU Boost if possible

## Related

- [Base Profiler](base_profiler.md) - Profile kernels to find tuning targets
- [CUDA Caching Allocator](cuda_cache.md) - Complementary memory optimizations
- [Custom CUDA Kernels](custom_ops.md) - Kernels that benefit from tuning

## Examples

### Tuning a Custom CUDA Kernel

```python
import torch
from cuda_optimizer import Autotuner

# Assume you have a custom CUDA kernel loaded via PyTorch extension
my_kernel = load(...)

def kernel_wrapper(block_dim, grid_dim):
    """Wrapper that calls your kernel with launch configuration."""
    my_kernel(input_tensor, block_dim, grid_dim)

autotuner = Autotuner(verbose=True, num_trials=10)

# Tune for your specific problem size
config = autotuner.autotune_kernel(
    kernel_fn=kernel_wrapper,
    kernel_name="my_elementwise_kernel",
    input_shape=(8192, 4096),
    dtype=torch.float16
)

print("Optimal configuration found:")
print(f"  Block size: {config['block_size']}")
print(f"  Grid size: ({config['grid_x']}, {config['grid_y']})")
print(f"  Execution time: {config['execution_time_ms']:.4f} ms")
print(f"  Throughput: {config['throughput_gb_s']:.2f} GB/s")

# Apply the config to your kernel
block_dim = config['block_size']
grid_dim = (config['grid_x'], config['grid_y'])
my_kernel(input_tensor, block_dim, grid_dim)
```

### Tuning Batch Size Impact

```python
from cuda_optimizer import Autotuner
import torch

autotuner = Autotuner()

# Tune for different batch sizes
batch_sizes = [8, 16, 32, 64]
for batch in batch_sizes:
    shape = (batch, 3, 224, 224)
    config = autotuner.autotune_kernel(
        kernel_fn=my_kernel,
        kernel_name="my_conv_kernel",
        input_shape=shape,
        dtype=torch.float16
    )
    print(f"Batch {batch}: block={config['block_size']}, time={config['execution_time_ms']:.4f}ms")
```

## Testing

```bash
# Run unit tests for autotuner
pytest tests/unit/tuner/test_autotuner.py -v -s

# Tune all kernels manually to verify
python scripts/tune_kernels.py --kernels all --trials 5
```

Expected behavior:
- Cache file created at `~/.cache/cuda-optimizer/kernel_configs.pkl`
- Best configs printed for each kernel
- Subsequent runs use cache (no retuning unless cache cleared)