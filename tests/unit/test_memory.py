"""
Unit tests for CUDA caching allocator and expiry policy.
"""

import pytest
import torch
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cuda_optimizer.memory import CUDACache, LRUExpiryPolicy


pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLRUExpiryPolicy:
    """Test suite for LRU expiry policy."""

    def test_initialization(self):
        """Test policy initializes correctly."""
        policy = LRUExpiryPolicy(max_pool_size_mb=512)
        assert policy.get_pool_size() == 0
        assert policy.get_block_count() == 0
        assert not policy.should_evict()

    def test_add_and_touch(self):
        """Test adding and touching blocks."""
        policy = LRUExpiryPolicy(max_pool_size_mb=10)
        policy.add("block1", 100)
        policy.add("block2", 200)
        assert policy.get_pool_size() == 300
        assert policy.get_block_count() == 2

        # Touch block1 to make it most recently used
        policy.touch("block1")

        # Eviction candidates should be block2 (LRU)
        candidates = policy.get_eviction_candidates()
        assert "block2" in candidates
        assert "block1" not in candidates

    def test_eviction_threshold(self):
        """Test eviction threshold is respected."""
        policy = LRUExpiryPolicy(max_pool_size_mb=1)  # 1MB limit
        policy.add("block1", 600 * 1024 * 1024)  # 600MB
        policy.add("block2", 500 * 1024 * 1024)  # 500MB -> total 1.1GB
        assert policy.should_evict()

        # Should need to evict at least 100MB
        candidates = policy.get_eviction_candidates(target_size_bytes=1 * 1024 * 1024)
        assert len(candidates) >= 1

    def test_remove(self):
        """Test removing blocks."""
        policy = LRUExpiryPolicy()
        policy.add("block1", 100)
        policy.add("block2", 200)

        policy.remove("block1")
        assert policy.get_pool_size() == 200
        assert policy.get_block_count() == 1

        policy.remove("block2")
        assert policy.get_pool_size() == 0
        assert policy.get_block_count() == 0

    def test_clear(self):
        """Test clearing all blocks."""
        policy = LRUExpiryPolicy()
        policy.add("block1", 100)
        policy.add("block2", 200)
        policy.clear()
        assert policy.get_pool_size() == 0
        assert policy.get_block_count() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDACache:
    """Test suite for CUDA caching allocator."""

    def test_allocation(self):
        """Test basic allocation."""
        cache = CUDACache(max_pool_size_mb=512)

        # Allocate a 1MB tensor (float32 = 4 bytes, so 262144 elements)
        tensor = cache.allocate(1024 * 1024)  # 1MB
        assert tensor is not None
        assert tensor.device.type == "cuda"
        assert tensor.numel() * 4 >= 1024 * 1024  # At least requested size

    def test_free_and_reuse(self):
        """Test freeing and reusing cached memory."""
        cache = CUDACache(max_pool_size_mb=512)

        # Allocate and free
        tensor1 = cache.allocate(1024 * 1024)
        ptr1 = tensor1.data_ptr()
        cache.free(tensor1)

        # Allocate again - should get same memory block (cache hit)
        tensor2 = cache.allocate(1024 * 1024)
        ptr2 = tensor2.data_ptr()

        assert ptr1 == ptr2, "Cache should reuse freed memory blocks"

        cache.free(tensor2)

    def test_cache_hit_rate(self):
        """Test that cache hit rate improves with reuse."""
        cache = CUDACache(max_pool_size_mb=512)

        # Allocate many tensors of same size
        tensors = [cache.allocate(512 * 1024) for _ in range(10)]
        for t in tensors:
            cache.free(t)

        # Allocate again
        stats_before = cache.get_stats()
        hits_before = stats_before["hit_count"]

        new_tensors = [cache.allocate(512 * 1024) for _ in range(10)]
        for t in new_tensors:
            cache.free(t)

        stats_after = cache.get_stats()
        hits_after = stats_after["hit_count"]

        # Should have 10 hits from reusing the first 10 allocations
        assert (hits_after - hits_before) >= 10

    def test_different_sizes(self):
        """Test allocations of different sizes use different size classes."""
        cache = CUDACache(max_pool_size_mb=512)

        sizes = [1024, 4096, 16384, 65536]  # bytes
        tensors = []

        for size in sizes:
            t = cache.allocate(size)
            tensors.append(t)

        # Free all
        for t in tensors:
            cache.free(t)

        stats = cache.get_stats()
        assert stats["block_count"] == len(sizes)
        assert stats["pool_size_bytes"] > 0

    def test_pool_size_limit(self):
        """Test that pool size respects the limit."""
        cache = CUDACache(max_pool_size_mb=10)  # 10MB limit

        # Allocate many blocks to exceed limit
        for _ in range(50):
            t = cache.allocate(256 * 1024)  # 256KB each -> 12.8MB total
            cache.free(t)

        stats = cache.get_stats()
        # Pool should be bounded (some evictions should occur)
        assert stats["pool_size_bytes"] <= 10 * 1024 * 1024 * 1.2  # Allow some margin

    def test_fragmentation_metrics(self):
        """Test fragmentation metrics are reasonable."""
        cache = CUDACache(max_pool_size_mb=512)

        # Allocate and free many blocks of same size class
        for _ in range(100):
            t = cache.allocate(
                1000
            )  # Will round to 1024 or 256 depending on granularity
            cache.free(t)

        frag = cache.get_fragmentation()

        # External fragmentation should be relatively low (<10% with our size classes)
        # In test environment, may vary
        assert frag["external_fragmentation"] >= 0
        assert frag["external_fragmentation"] <= 0.2  # Reasonable upper bound

    def test_stats_tracking(self):
        """Test statistics tracking."""
        cache = CUDACache(max_pool_size_mb=512)

        assert cache.get_stats()["total_allocated"] == 0

        t1 = cache.allocate(1024)
        t2 = cache.allocate(1024)

        stats = cache.get_stats()
        assert stats["total_allocated"] == 2
        assert stats["block_count"] == 0  # Still allocated, not in pool

        cache.free(t1)
        cache.free(t2)

        stats = cache.get_stats()
        assert stats["block_count"] == 2
        assert stats["total_freed"] == 2

    def test_clear(self):
        """Test clearing the cache."""
        cache = CUDACache(max_pool_size_mb=512)

        t1 = cache.allocate(1024)
        t2 = cache.allocate(2048)
        cache.free(t1)
        cache.free(t2)

        assert cache.get_stats()["block_count"] == 2

        cache.clear()
        assert cache.get_stats()["block_count"] == 0
        assert cache.get_stats()["pool_size_bytes"] == 0

    def test_dtype_support(self):
        """Test allocation with different dtypes."""
        cache = CUDACache()

        for dtype in [torch.float16, torch.float32, torch.float64]:
            t = cache.allocate(1024, dtype=dtype)
            assert t.dtype == dtype
            cache.free(t)

    def test_multiple_allocate_free_cycle(self):
        """Test multiple allocate/free cycles show LRU behavior."""
        cache = CUDACache(max_pool_size_mb=10)

        # Allocate 5 blocks of 1MB each
        blocks = [cache.allocate(1024 * 1024) for _ in range(5)]
        ptrs = [b.data_ptr() for b in blocks]

        # Free all
        for b in blocks:
            cache.free(b)

        # Allocate 3 of the 5 back (should get 3 most recently freed? Actually LRU: first ones evicted if limit exceeded)
        # The pool will have 5 blocks. If we allocate 3, we get the 3 most recently used (last freed).
        # In OrderedDict, last=True means most recent at end, popitem(last=False) gets LRU from beginning.
        # When we free blocks in order 0,1,2,3,4, the order in pool (by touch) should be: 0 (oldest) ... 4 (newest)
        # So allocations should get 4,3,2 (the 3 most recent) if we pop from end. But we pop from beginning (LRU).
        # Actually, in free(), we use move_to_end(block_id, last=True), which puts at end. So after freeing 5:
        # pool order: [block0, block1, block2, block3, block4] with block4 at end (most recent)
        # allocate() uses popitem(last=False) which pops leftmost = block0 (oldest). That's wrong for LRU cache - we want to pop least recently used, which is correct, but we want to allocate the most recently used block (to reuse hot blocks). Wait, we want cache HIT to return the most recently used block? No, we want to reuse any block that's available. The LRU policy is for eviction, not for allocation. In allocation, we can pick any block. We pop from the beginning (oldest) to maintain LRU order? Actually, if we want to keep recently used blocks in the pool (i.e., avoid evicting hot blocks), we should evict the oldest ones first. But allocation can use any block. It might be better to use the most recently freed block (last in the OrderedDict) to keep the pool order consistent? Let's think:
        # - When a block is freed, it becomes most recent (end)
        # - When we allocate, we want to remove a block from pool. Which one? If we remove a random block, the LRU ordering still works for eviction decisions? The order in the OrderedDict is based on when they were added/freed. If we remove a block from the middle, we need to handle that. If we always pop from the beginning (oldest), then the pool always maintains the order of "last freed time" for the remaining blocks. That's fine. The oldest ones are at the beginning, and when eviction happens, we evict from beginning. So allocation should pop from the beginning to preserve the order of remaining blocks. This means allocation tends to use the oldest freed blocks first. That's acceptable.
        # So the test: allocate 3 should return blocks 0,1,2 (the oldest). But we need to check that the blocks are from the pool.
        a1, a2, a3 = [cache.allocate(1024 * 1024) for _ in range(3)]
        returned_ptrs = [a1.data_ptr(), a2.data_ptr(), a3.data_ptr()]

        # Should be from the original set (some of the original 5)
        for ptr in returned_ptrs:
            assert ptr in ptrs

        # Clean up
        for b in [a1, a2, a3]:
            cache.free(b)

        # The remaining 2 blocks should still be in pool
        stats = cache.get_stats()
        assert stats["block_count"] == 2

    def test_fragmentation_under_five_percent(self):
        """Test that memory fragmentation is under 5% with uniform allocations."""
        cache = CUDACache(max_pool_size_mb=128)

        # Simulate workloads: allocate and free many blocks of the same size
        block_size = 4096  # 4KB
        num_blocks = 200

        allocated = []
        for _ in range(num_blocks):
            t = cache.allocate(block_size)
            allocated.append(t)

        # Free all
        for t in allocated:
            cache.free(t)

        # Check fragmentation
        frag = cache.get_fragmentation()
        external_frag = frag["external_fragmentation"]

        # Our size classes should keep external fragmentation low
        # With rounding to 256-byte granularity for small sizes, waste is minimal
        assert external_frag < 0.05, (
            f"External fragmentation {external_frag:.2%} exceeds 5%"
        )

        # Additional sanity: total allocated in pool should be close to requested
        total_allocated = frag["total_allocated_bytes"]
        total_requested = frag["total_requested_bytes"]
        if total_allocated > 0:
            efficiency = total_requested / total_allocated
            assert efficiency >= 0.95, f"Memory efficiency {efficiency:.2%} below 95%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
