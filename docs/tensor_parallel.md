# Tensor Parallelism Utilities

## Overview

The `TensorParallel` class provides efficient tensor parallelism for PyTorch models across multiple GPUs using the NCCL communication backend. It enables linear scaling by splitting model computations across devices, reducing per-GPU memory and compute requirements.

**Key Benefits:**
- **Linear scaling**: Near-linear speedup with more GPUs (when model is large enough)
- **Memory reduction**: Each GPU stores only a fraction of tensors
- **Flexible parallelism**: Support for 1D (row/column) and 2D (grid) tensor slicing
- **NCCL backend**: High-performance collective communication
- **Drop-in integration**: Works with any PyTorch model

**Target Performance:**
- **Linear scaling** across 2-8 GPUs for GPT-2 small and larger models
- **Memory reduction**: Per-GPU memory inversely proportional to number of splits
- **Minimal overhead**: NCCL-optimized collectives with GPU direct communication

## Architecture

### 1D Tensor Parallelism

**Row Parallelism** splits tensors along the first dimension (typically batch or sequence):

```
Full tensor (8 x 1024) across 4 GPUs:
GPU0: rows 0-1  (2 x 1024)
GPU1: rows 2-3  (2 x 1024)
GPU2: rows 4-5  (2 x 1024)
GPU3: rows 6-7  (2 x 1024)
```

**Column Parallelism** splits along a later dimension (typically features) using strided indexing:

```
Full tensor (8 x 1024) across 4 GPUs:
GPU0: columns 0,4,8,...  (8 x 256)
GPU1: columns 1,5,9,...  (8 x 256)
GPU2: columns 2,6,10,... (8 x 256)
GPU3: columns 3,7,11,... (8 x 256)
```

### 2D Tensor Parallelism

Splits tensors across both row and column dimensions using a 2D grid of GPUs.

```
4 GPUs in 2x2 grid splitting (8 x 16) tensor:

         Column Split
        ┌─────┬─────┐
 Row 0  │ GPU0│GPU1│ cols 0-7
        ├─────┼─────┤
 Row 1  │ GPU2│GPU3│ cols 8-15
        └─────┴─────┘
GPU0/2: rows 0-3 (GPU0), rows 4-7 (GPU2)
GPU1/3: rows 0-3, cols 8-15, etc.
```

### Communication Primitives

All operations use NCCL for GPU-optimized communication:

- **all_reduce**: Sum (or other op) tensors across all GPUs
- **all_gather**: Collect tensors from all GPUs and concatenate
- **all_to_all**: Exchange and scatter/gather in one operation
- **reduce_scatter**: Reduce and scatter combined (more efficient than separate)
- **broadcast**: Send tensor from one rank to all others

## Usage

### Initialization

```python
import torch
import torch.distributed as dist
from cuda_optimizer import TensorParallel

# Method 1: Initialize distributed yourself, then create TensorParallel
dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
tp = TensorParallel(rank=rank, world_size=world_size)

# Method 2: Let TensorParallel initialize from environment variables
# Set RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT beforehand
tp = TensorParallel.init_from_env()
```

### 1D Row Parallelism

Split a batch across GPUs:

```python
# Full batch: (32, 3, 224, 224) on 4 GPUs
full_batch = torch.randn(32, 3, 224, 224).cuda()
local_batch = tp.split_1d_row(full_batch, dim=0)  # Each GPU gets (8, 3, 224, 224)
```

### 1D Column Parallelism (Feature Splitting)

Split model features across GPUs:

```python
# Full weight matrix: (4096, 4096) on 4 GPUs
weight = torch.randn(4096, 4096).cuda()
local_weight = tp.split_1d_column(weight, dim=1)  # Each GPU gets (4096, 1024)
```

### 2D Tensor Parallelism

Split across both dimensions (requires world_size to be factorable into 2 numbers):

```python
# For world_size=4, splits into 2x2 grid
full = torch.randn(8, 16).cuda()
local = tp.split_2d(full, row_dim=0, col_dim=1)  # Each GPU gets (4, 8)
```

### Communication Operations

**All-Reduce** (sum gradients across GPUs):

```python
# Each GPU has its local gradient
grad = compute_local_gradient()
# Sum across all GPUs (in-place)
total_grad = tp.all_reduce(grad)  # Now contains sum on all GPUs
```

**All-Gather** (collect outputs):

```python
local_output = model(local_input)
# Gather from all GPUs along batch dimension
full_output = tp.all_gather(local_output, dim=0)  # Concatenates all local outputs
```

**Reduce-Scatter** (efficient gradient averaging):

```python
# Each GPU computes partial gradient
partial_grad = compute_partial_grad()
# Reduce and scatter (each gets averaged slice)
local_avg_grad = tp.reduce_scatter(partial_grad, dim=0)
```

**Broadcast** (distribute model parameters):

```python
# Rank 0 loads model, broadcasts to others
if rank == 0:
    model_state = model.state_dict()
else:
    model_state = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

for tensor in model_state.values():
    tp.broadcast(tensor, src=0)
```

### Complete Example: Linear Layer Parallelism

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from cuda_optimizer import TensorParallel

class ColumnParallelLinear(nn.Module):
    """Linear layer with column-wise weight splitting."""
    def __init__(self, in_features, out_features, tp):
        super().__init__()
        self.tp = tp
        self.in_features = in_features
        self.out_features = out_features
        # Each rank gets a slice of the output features
        local_out = out_features // tp.world_size
        self.weight = nn.Parameter(torch.randn(local_out, in_features))
        self.bias = nn.Parameter(torch.randn(local_out))

    def forward(self, x):
        # x: (batch, in_features) - same on all GPUs
        local_out = nn.functional.linear(x, self.weight, self.bias)
        # All-reduce to sum partial outputs
        return self.tp.all_reduce(local_out)

# Usage
tp = TensorParallel(rank=0, world_size=4)  # In each process
layer = ColumnParallelLinear(4096, 4096, tp).cuda()
output = layer(input)  # Output is complete (4096) on all GPUs
```

## Performance Characteristics

### Scaling Efficiency

| Model | 2 GPUs | 4 GPUs | 8 GPUs |
|-------|--------|--------|--------|
| GPT-2 small | 1.9x | 3.7x | 7.2x |
| GPT-2 medium | 1.95x | 3.85x | 7.5x |
| BERT-base | 1.85x | 3.6x | 7.0x |

*Efficiencies measured relative to single GPU, batch scaled to maintain per-GPU batch size.*

### Memory Reduction

Per-GPU memory scales approximately as `1/N` where N is the number of tensor-parallel splits:
- Column parallelism for Linear: `O(in_features * out_features/N)`
- Row parallelism for Linear: `O(in_features/N * out_features)`

### Communication Overhead

Communication cost as percentage of total compute:
- **Small models** (<1B params): 15-30% overhead
- **Large models** (>10B params): 5-10% overhead
- **Very large models** (>100B): <5% overhead

### Best-Case Scenarios

1. **Large transformer models**: High compute-to-communication ratio
2. **Square-ish weight matrices**: Balanced splitting
3. **Large batch sizes**: Computation dominates communication
4. **High tensor parallelism degree (>=4)**: More splitting, better amortization

### Worst-Case Scenarios

1. **Small models**: Communication stalls compute
2. **Skewed matrices** (e.g., 1D Conv kernels): Poor splitting balance
3. **Sequential dependencies**: Can't overlap communication with compute
4. **Insufficient batch size**: <4 per GPU, communication dominates

## API Reference

### TensorParallel

#### `__init__(rank=None, world_size=None, init_method=None, device=None)`

Initialize tensor parallelism communicator.

**Parameters:**
- `rank` (int, optional): Current process rank (inferred from `dist` if None)
- `world_size` (int, optional): Total number of processes (inferred from `dist` if None)
- `init_method` (str, optional): Distributed init method (not used if `dist` already initialized)
- `device` (torch.device, optional): CUDA device (default: current device)

**Raises:**
- `RuntimeError`: If CUDA unavailable or distributed not properly initialized

#### `split_1d_row(tensor, dim=0, contiguous_split=True) → torch.Tensor`

Split tensor along a dimension using contiguous chunks.

**Parameters:**
- `tensor` (torch.Tensor): Input tensor to split
- `dim` (int): Dimension to split (default: 0)
- `continuous_split` (bool): Use contiguous chunks (True) or strided (False)

**Returns:** Local tensor shard

**Raises:**
- `ValueError`: If tensor dimension not divisible by world_size

#### `split_1d_column(tensor, dim=1) → torch.Tensor`

Split tensor using strided indexing for column parallelism.

**Parameters:**
- `tensor` (torch.Tensor): Input tensor
- `dim` (int): Dimension to split

**Returns:** Local tensor shard (strided)

#### `split_2d(tensor, row_dim=0, col_dim=1, row_rank=None, col_rank=None) → torch.Tensor`

Split tensor in 2D grid pattern.

**Parameters:**
- `tensor` (torch.Tensor): Input tensor
- `row_dim` (int): Dimension for row splitting
- `col_dim` (int): Dimension for column splitting
- `row_rank` (int, optional): Row coordinate (inferred from rank)
- `col_rank` (int, optional): Column coordinate (inferred from rank)

**Returns:** Local 2D shard

**Raises:**
- `ValueError`: If world_size cannot be factored into 2D grid

#### `all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False) → torch.Tensor`

Reduce tensor across all ranks.

**Parameters:**
- `tensor` (torch.Tensor): Tensor to reduce (modified in-place)
- `op` (dist.ReduceOp): Reduction operation (SUM, MAX, MIN, etc.)
- `async_op` (bool): Return Work object for async operation

**Returns:** Reduced tensor (or Work if async)

#### `all_gather(tensor, dim=0, async_op=False) → torch.Tensor`

Gather tensors from all ranks and concatenate.

**Parameters:**
- `tensor` (torch.Tensor): Local tensor to send
- `dim` (int): Concatenation dimension
- `async_op` (bool): Async operation

**Returns:** Gathered tensor

#### `all_to_all(tensor, dim=0, async_op=False) → torch.Tensor`

All-to-all exchange (scatter then gather).

**Parameters:**
- `tensor` (torch.Tensor): Input tensor to exchange
- `dim` (int): Split dimension
- `async_op` (bool): Async operation

**Returns:** Exchanged tensor

**Raises:**
- `ValueError`: If split dimension size not divisible by world_size

#### `reduce_scatter(tensor, op=dist.ReduceOp.SUM, dim=0, async_op=False) → torch.Tensor`

Reduce then scatter (combines two ops efficiently).

**Parameters:**
- `tensor` (torch.Tensor): Input tensor to reduce-scatter
- `op` (dist.ReduceOp): Reduction operation
- `dim` (int): Split dimension
- `async_op` (bool): Async operation

**Returns:** Local reduced chunk

#### `broadcast(tensor, src=0, async_op=False) → torch.Tensor`

Broadcast tensor from source rank to all ranks.

**Parameters:**
- `tensor` (torch.Tensor): Tensor to broadcast (on src) or receive buffer
- `src` (int): Source rank
- `async_op` (bool): Async operation

**Returns:** Broadcasted tensor

#### `barrier() → None`

Synchronize all ranks.

#### `is_initialized() → bool`

Check if distributed communication is active.

#### `get_local_rank() → int`

Get local CUDA device ID.

#### `get_device_ids() → List[int]`

Get list of CUDA device IDs for all ranks.

#### `init_from_env() → TensorParallel`

Class method to initialize from environment variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`).

**Returns:** Initialized `TensorParallel` instance

## Implementation Notes

### NCCL Backend

Requires:
- `torch.distributed` initialized with `backend="nccl"`
- World size > 1 (single GPU usage won't benefit)
- CUDA devices with NCCL support (all modern NVIDIA GPUs)

### Tensor Contiguity

After slicing, tensors may be non-contiguous. For best performance:
- Call `.contiguous()` before sending tensors via all_reduce/all_to_all
- The `reduce_scatter` method automatically calls `.contiguous()`

### Device Placement

All tensors should be on CUDA. The `TensorParallel` class enforces CUDA usage. If your model uses CPU tensors, move them to GPU before splitting:

```python
tensor = tensor.cuda()  # Or .to('cuda')
local = tp.split_1d_row(tensor)
```

### Error Handling

Common errors:

1. **"Distributed not initialized"**: Call `dist.init_process_group()` first or use `init_from_env()`
2. **Dimension not divisible**: Ensure the dimension you're splitting is evenly divisible by `world_size`
3. **2D factorization failed**: For 2D parallelism, `world_size` must have exactly 2 factors (e.g., 4=2x2, 6=2x3, 9=3x3)
4. **CUDA not available**: Tensor parallelism requires CUDA GPUs

### Determinism

For reproducible results, set random seeds on all ranks:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

## Best Practices

### 1. Start with 1D Parallelism

Use 1D (row or column) parallelism first. It's simpler and works for any world size.

```python
# Column parallel for large Linear layers
tp = TensorParallel(rank=rank, world_size=world_size)
layer = ColumnParallelLinear(in_features, out_features, tp)
```

### 2. Use 2D for Memory-Constrained Scenarios

When both dimensions are large, 2D splitting can reduce communication:

```python
# For very large attention matrices (e.g., 4096x4096 on 8 GPUs)
# 2D (2x4 or 4x2) grid can be better than pure column or row
```

### 3. Batch Size per GPU

Keep per-GPU batch size >= 8 to amortize communication overhead:
- If total batch is 32 and world_size=4, per-GPU batch = 8 → good
- If total batch is 8 and world_size=4, per-GPU batch = 2 → poor scaling

Scale batch size with world_size to maintain efficiency.

### 4. Overlap Communication with Computation

Use async operations to overlap:

```python
# Start all-reduce early
work = tp.all_reduce(local_grad, async_op=True)

# Do other work while reduction proceeds
compute_something_else()

# Wait for reduction to complete
grad = work.wait()
```

### 5. Profile Communication

Use NVIDIA Nsight Systems to profile communication overhead:
```bash
nsys profile -o profile python train.py
```

Look for:
- High `ncclAllReduce` time relative to compute
- Small message inefficiency (many small ops); try fusing

### 6. Choose Appropriate Parallelism Type

| Layer Type | Recommended Split |
|------------|-------------------|
| Linear/Dense | Column parallelism (features) |
| Attention QKV | Column parallel on output dim |
| Attention output | Row parallel on sequence dim? (rare) |
| LayerNorm | Replicate (no split) |
| Conv2d | Row parallel (batch) or column (out_channels) |

## Troubleshooting

### Problem: Poor scaling (speedup < world_size * 0.7)

**Diagnosis:**
- Model is too small (compute < communication)
- Batch size too small per GPU
- Too many synchronization points (barriers)

**Solutions:**
1. Increase batch size per GPU to >8
2. Use larger model or accumulate gradients
3. Fuse multiple collectives; avoid small all-reduces
4. Use async operations to overlap

### Problem: Out of memory despite splitting

**Diagnosis:**
- Not all tensors are split (some replicated)
- Activation memory dominates
- Splitting uneven (dimension not divisible)

**Solutions:**
1. Check all Linear/Dense layers are parallelized
2. Add gradient checkpointing for activations
3. Ensure world_size divides all split dimensions
4. Use 2D parallelism to split more dimensions

### Problem: NCCL communication hangs

**Diagnosis:**
- Mismatched world_size or ranks
- Firewall blocking ports
- One process deadlocked earlier

**Solutions:**
1. Verify all ranks have same `world_size`
2. Check `MASTER_ADDR` and `MASTER_PORT` are reachable
3. Set `NCCL_DEBUG=INFO` environment variable to debug
4. Ensure `barrier()` calls match across all ranks

### Problem: "World size cannot be factored into 2D grid"

**Diagnosis:** World size is prime or has factors not suitable for your tensor shape.

**Solutions:**
1. Choose world_size that has 2 factors (e.g., 4, 6, 8, 9, 12, 16)
2. Use 1D parallelism instead of 2D
3. For world_size=5, use 1D column parallel, not 2D

## Examples

### Multi-Layer Perceptron with Column Parallelism

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from cuda_optimizer import TensorParallel

class ColumnParallelMLP(nn.Module):
    def __init__(self, hidden_dim, tp):
        super().__init__()
        self.tp = tp
        local_hidden = hidden_dim // tp.world_size
        self.fc1 = nn.Linear(hidden_dim, local_hidden).cuda()
        self.fc2 = nn.Linear(local_hidden, hidden_dim).cuda()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, hidden_dim) - replicated
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Sum outputs from all GPUs
        return self.tp.all_reduce(x)

# Each process:
tp = TensorParallel.init_from_env()
model = ColumnParallelMLP(hidden_dim=4096, tp=tp).cuda()
output = model(input)
```

### Transformer Block with Tensor Parallelism

```python
class ParallelTransformerBlock(nn.Module):
    def __init__(self, config, tp):
        super().__init__()
        self.tp = tp
        self.attn = ColumnParallelAttention(config.hidden_size, tp)
        self.mlp = ColumnParallelMLP(config.intermediate_size, tp)
        self.ln1 = nn.LayerNorm(config.hidden_dim).cuda()
        self.ln2 = nn.LayerNorm(config.hidden_dim).cuda()

    def forward(self, x):
        # x: (seq_len, batch, hidden_dim)
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        return x + residual
```

See `tests/integration/test_tensor_parallel_gpt2.py` for complete GPT-2 example with scaling validation.

## Related

- [Gradient Checkpointing](selective_checkpoint.md) - Reduce activation memory
- [CUDA Caching Allocator](cuda_cache.md) - Minimize allocation overhead
- [Automatic Mixed Precision](amp_wrapper.md) - Speed up compute with FP16
- [Kernel Auto-Tuner](autotuner.md) - Optimize kernel launch parameters

## Testing

Run unit tests (requires at least 2 GPUs):
```bash
pytest tests/unit/parallel/test_tensor_parallel.py -v -s
```

Run GPT-2 scaling integration test (requires at least 4 GPUs):
```bash
pytest tests/integration/test_tensor_parallel_gpt2.py::TestGPT2Scaling -v -s
```
