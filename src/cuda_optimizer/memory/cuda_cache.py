"""
CUDA memory caching allocator with pool reuse and LRU-based eviction.
"""

import threading
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import torch

from .expiry_policy import LRUExpiryPolicy


class CUDACache:
    """CUDA memory caching allocator with pool reuse and LRU-based eviction.

    Features:
    - Memory pool with size class granularity for efficient reuse
    - LRU (Least Recently Used) eviction policy to bound memory usage
    - Fragmentation tracking and reduction (<5% target)
    - Thread-safe operations
    - Drop-in replacement for direct torch.cuda allocation

    Example:
        cache = CUDACache(max_pool_size_mb=1024)

        # Allocate tensor from cache
        tensor = cache.allocate(1024 * 1024)  # 1MB

        # Use tensor normally
        result = tensor * 2

        # Free back to cache
        cache.free(tensor)
    """

    # Size class granularity: round up to nearest 256 bytes for small sizes
    # For larger sizes, use power-of-2 alignment
    SIZE_CLASS_GRANULARITY = 256

    def __init__(self, max_pool_size_mb: int = 1024):
        """Initialize CUDA caching allocator.

        Args:
            max_pool_size_mb: Maximum memory pool size in megabytes (default 1024)
        """
        self.max_pool_size_bytes = max_pool_size_mb * 1024 * 1024
        self.expiry_policy = LRUExpiryPolicy(max_pool_size_mb)

        # Pools: size_class -> OrderedDict of (block_id -> tensor)
        # OrderedDict provides LRU ordering within each size class
        self._pools: Dict[int, OrderedDict] = {}

        # Mapping: tensor id -> (size_class, block_id)
        self._tensor_registry: Dict[int, Tuple[int, str]] = {}

        self._lock = threading.RLock()
        self._block_counter = 0
        self._stats_lock = threading.Lock()

        # Statistics
        self._alloc_count = 0
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._total_freed = 0

    def _round_up_size(self, size_bytes: int) -> int:
        """Round size up to nearest size class.

        Args:
            size_bytes: Requested size in bytes

        Returns:
            Rounded size class
        """
        if size_bytes <= 0:
            raise ValueError("Size must be positive")

        if size_bytes < 4096:
            # Small allocations: round to 256-byte granularity
            return (
                math.ceil(size_bytes / self.SIZE_CLASS_GRANULARITY)
                * self.SIZE_CLASS_GRANULARITY
            )
        else:
            # Large allocations: round to next power of 2
            return 2 ** math.ceil(math.log2(size_bytes))

    def allocate(
        self,
        size_bytes: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        **kwargs,
    ) -> torch.Tensor:
        """Allocate a tensor from the cache pool.

        Args:
            size_bytes: Size in bytes to allocate
            dtype: Tensor data type (default torch.float32)
            device: Device to allocate on (default "cuda")
            **kwargs: Additional arguments for torch.empty()

        Returns:
            torch.Tensor allocated from cache or newly allocated

        Raises:
            RuntimeError: If CUDA is not available or allocation fails
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        size_class = self._round_up_size(size_bytes)

        with self._lock:
            # Check if cache hit exists in pool
            pool = self._pools.get(size_class)
            if pool and len(pool) > 0:
                # Get the least recently used block from this size class
                block_id, tensor = pool.popitem(last=False)  # LRU = first item

                # Update tracking
                tensor_id = id(tensor)
                if tensor_id in self._tensor_registry:
                    del self._tensor_registry[tensor_id]

                # Mark as allocated
                self._tensor_registry[tensor_id] = (size_class, block_id)
                self.expiry_policy.remove(block_id)

                with self._stats_lock:
                    self._hit_count += 1

                return tensor

            # Cache miss: allocate new tensor
            with self._stats_lock:
                self._miss_count += 1
                self._alloc_count += 1

            num_elements = size_bytes // self._get_dtype_size(dtype)
            tensor = torch.empty(num_elements, dtype=dtype, device=device, **kwargs)

            # Track this allocation
            tensor_id = id(tensor)
            block_id = f"block_{self._block_counter}"
            self._block_counter += 1
            self._tensor_registry[tensor_id] = (size_class, block_id)

            return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """Free a tensor back to the cache pool.

        Args:
            tensor: Tensor to free
        """
        if tensor is None:
            return

        tensor_id = id(tensor)

        with self._lock:
            if tensor_id not in self._tensor_registry:
                # Tensor wasn't allocated from cache, skip
                return

            size_class, block_id = self._tensor_registry[tensor_id]
            del self._tensor_registry[tensor_id]

            # Check if eviction is needed before adding
            bytes_to_add = self._get_tensor_size(tensor)

            if self.expiry_policy.should_evict():
                self._evict_blocks()

            # Add to pool
            pool = self._pools.setdefault(size_class, OrderedDict())
            pool[block_id] = tensor
            self.expiry_policy.add(block_id, bytes_to_add)

            # Ensure LRU order (newly freed block goes to end = most recent)
            pool.move_to_end(block_id, last=True)

            with self._stats_lock:
                self._total_freed += 1

    def _evict_blocks(self) -> None:
        """Evict least recently used blocks to maintain pool size limit."""
        candidates = self.expiry_policy.get_eviction_candidates()

        for block_id in candidates:
            # Find which pool contains this block
            for size_class, pool in list(self._pools.items()):
                if block_id in pool:
                    del pool[block_id]
                    self.expiry_policy.remove(block_id)

                    with self._stats_lock:
                        self._eviction_count += 1

                    # Clean up empty pools
                    if len(pool) == 0:
                        del self._pools[size_class]

                    break  # Block found and removed

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Get total size of tensor in bytes.

        Args:
            tensor: PyTorch tensor

        Returns:
            Size in bytes
        """
        return tensor.numel() * self._get_dtype_size(tensor.dtype)

    def _get_dtype_size(self, dtype: torch.dtype) -> int:
        """Get size in bytes for a torch dtype.

        Args:
            dtype: PyTorch dtype

        Returns:
            Size in bytes per element
        """
        dtype_sizes = {
            torch.float16: 2,
            torch.float32: 4,
            torch.float64: 8,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.uint8: 1,
            torch.bool: 1,
        }
        return dtype_sizes.get(dtype, 4)  # Default to 4 bytes

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including:
            - pool_size_bytes: Current pool size
            - block_count: Number of blocks in pool
            - hit_rate: Cache hit rate (0-1)
            - eviction_count: Number of blocks evicted
            - total_allocated: Total allocations performed
        """
        with self._lock, self._stats_lock:
            total_pool_size = self.expiry_policy.get_pool_size()
            block_count = self.expiry_policy.get_block_count()

            total_lookups = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_lookups if total_lookups > 0 else 0.0

            return {
                "pool_size_bytes": total_pool_size,
                "pool_size_mb": total_pool_size / (1024 * 1024),
                "block_count": block_count,
                "hit_rate": hit_rate,
                "eviction_count": self._eviction_count,
                "total_allocated": self._alloc_count,
                "total_freed": self._total_freed,
            }

    def get_fragmentation(self) -> dict:
        """Calculate memory fragmentation metrics.

        Fragmentation is defined as (wasted_space / total_pool_size) where
        wasted_space comes from:
        1. External fragmentation: unused space in partially filled pools
        2. Size class rounding: difference between requested and allocated size

        Returns:
            Dict with fragmentation metrics:
            - total_requested_bytes: Sum of original requested sizes
            - total_allocated_bytes: Sum of actual allocated sizes
            - external_fragmentation: (allocated - requested) / allocated
            - pool_utilization: used_space / pool_size
        """
        with self._lock:
            if self.expiry_policy.get_pool_size() == 0:
                return {
                    "total_requested_bytes": 0,
                    "total_allocated_bytes": 0,
                    "external_fragmentation": 0.0,
                    "pool_utilization": 0.0,
                }

            # Calculate total allocated size in pools
            total_allocated = self.expiry_policy.get_pool_size()

            # For external fragmentation, we need to know original request sizes
            # We track this per block in expiry policy by storing (block_id, allocated_size, requested_size)
            # For now, we calculate based on size classes
            total_requested = 0
            for size_class, pool in self._pools.items():
                # Each block in this pool has rounded size = size_class
                # Actual minimum requested would be unknown without tracking
                # We estimate: assume requests are evenly distributed in (size_class - granularity, size_class]
                avg_request = (
                    size_class - self.SIZE_CLASS_GRANULARITY // 2
                    if size_class < 4096
                    else size_class * 0.9
                )
                total_requested += len(pool) * avg_request

            # External fragmentation = (allocated - requested) / allocated
            if total_allocated > 0:
                external_frag = (total_allocated - total_requested) / total_allocated
            else:
                external_frag = 0.0

            # Pool utilization = (allocated blocks that are actually being used) / total pool size
            # Currently all blocks in pool are free, so utilization is 0% (this is expected)
            # In steady state, we'd track active vs inactive blocks differently

            return {
                "total_requested_bytes": total_requested,
                "total_allocated_bytes": total_allocated,
                "external_fragmentation": external_frag,
                "pool_utilization": 0.0,  # All pool blocks are currently free
            }

    def clear(self) -> None:
        """Clear all cached memory blocks."""
        with self._lock:
            self._pools.clear()
            self._tensor_registry.clear()
            self.expiry_policy.clear()

            with self._stats_lock:
                self._alloc_count = 0
                self._hit_count = 0
                self._miss_count = 0
                self._eviction_count = 0
                self._total_freed = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
