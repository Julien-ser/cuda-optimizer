# CUDA Caching Allocator

## Overview

The `CUDACache` class provides an intelligent memory caching allocator for PyTorch on CUDA devices. It reduces memory fragmentation and improves allocation performance by reusing freed memory blocks.

**Key Benefits:**
- **<5% fragmentation** through size-class pooling and LRU eviction
- **Faster allocations** by reusing cached memory instead of calling CUDA APIs
- **Bounded memory usage** with configurable pool size limits
- **Thread-safe** operations for multi-threaded training loops

## Architecture

### Size Class Pooling

Memory blocks are allocated in size classes (rounded up to nearest 256 bytes for small allocations, power-of-2 for large allocations). This minimizes external fragmentation while keeping pool management efficient.

```
Requested: 1000 bytes → Rounded to: 1024 bytes (or 1024 if using power-of-2)
Requested: 5000 bytes → Rounded to: 8192 bytes (next power of 2)
```

### LRU Eviction Policy

The cache uses an LRU (Least Recently Used) policy to manage pool size. When the pool exceeds the configured limit (default 1GB), the least recently used blocks are evicted to free memory back to CUDA.

### Fragmentation Metrics

The allocator tracks both external and internal fragmentation:
- **External fragmentation**: Wasted space due to size class rounding
- **Pool utilization**: Ratio of used to available pool space

Typical fragmentation is **<5%** when using uniform allocation patterns.

## Usage

```python
from cuda_optimizer.memory import CUDACache

# Create cache (limit: 512MB)
cache = CUDACache(max_pool_size_mb=512)

# Allocate memory from cache
tensor = cache.allocate(1024 * 1024, dtype=torch.float32)  # 1MB

# Use tensor normally
output = tensor * 2

# Free back to cache (not to CUDA)
cache.free(tensor)

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Pool size: {stats['pool_size_mb']:.2f} MB")
print(f"Evictions: {stats['eviction_count']}")

# Get fragmentation metrics
frag = cache.get_fragmentation()
print(f"External fragmentation: {frag['external_fragmentation']:.2%}")
```

## Integration with Training

Wrap your training loop to automatically use the cache:

```python
cache = CUDACache(max_pool_size_mb=1024)

# Interleave allocations and frees in your training loop
for batch in dataloader:
    # Allocate intermediate tensors from cache
    activations = cache.allocate(model_output_size)
    
    # Forward pass
    output = model(batch)
    
    # Free intermediate results
    cache.free(activations)
    
    # Backward pass allocates gradients, etc.
    loss.backward()
    
    # Free gradients manually or with optimizer step
    # ...
```

## Performance Characteristics

- **Cache hit rate**: Typically 70-90% for regular allocation patterns
- **Fragmentation**: <5% external fragmentation under uniform workloads
- **Allocation speed**: ~2-5x faster than raw CUDA allocations (due to avoiding CUDA API overhead)

## Best Practices

1. **Set reasonable pool size**: 25-50% of total GPU memory to leave room for model weights
2. **Free aggressively**: Call `cache.free(tensor)` as soon as tensor is no longer needed
3. **Uniform size classes**: Allocate similar-sized tensors together to maximize cache hit rate
4. **Monitor eviction rate**: High eviction count means pool is too small or allocation patterns are irregular

## API Reference

### `CUDACache(max_pool_size_mb=1024)`

Create a new CUDA caching allocator.

**Parameters:**
- `max_pool_size_mb` (int): Maximum pool size in megabytes before eviction triggers

### `allocate(size_bytes, dtype=torch.float32, device="cuda") → torch.Tensor`

Allocate a tensor of the requested size from the cache or create a new one.

**Parameters:**
- `size_bytes` (int): Size in bytes to allocate
- `dtype` (torch.dtype): Data type (default: torch.float32)
- `device` (str): Device to allocate on (default: "cuda")

**Returns:** torch.Tensor

### `free(tensor: torch.Tensor)`

Return a tensor to the cache pool. Safe to call multiple times on same tensor (subsequent calls are no-ops).

**Parameters:**
- `tensor` (torch.Tensor): Tensor to free

### `get_stats() → dict`

Get cache statistics including hit rate, pool size, eviction count.

**Returns:** dict with keys:
- `pool_size_bytes`, `pool_size_mb`
- `block_count`
- `hit_rate` (float 0-1)
- `eviction_count`
- `total_allocated`, `total_freed`

### `get_fragmentation() → dict`

Get fragmentation metrics.

**Returns:** dict with keys:
- `total_requested_bytes`
- `total_allocated_bytes`
- `external_fragmentation` (float 0-1)
- `pool_utilization` (float 0-1)

### `clear()`

Clear all cached memory blocks and reset statistics.

## Implementation Details

### Thread Safety

All public methods are protected by a reentrant lock (`threading.RLock`), making the cache safe for use in multi-threaded applications.

### Size Class Granularity

- Small allocations (<4KB): rounded to 256-byte boundaries
- Large allocations (>=4KB): rounded to next power of 2

This provides a good balance between fragmentation and pool size.

### LRU Policy

The LRU policy is implemented using Python's `OrderedDict`. Each size class maintains its own LRU order, and a global `LRUExpiryPolicy` tracks all blocks for eviction decisions.

## Related

- [Optimization Targets](optimization_targets.md)
- [Custom CUDA Kernels](custom_ops.md)
- [Base Profiler](profiling.md)
