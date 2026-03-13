"""
Memory block expiry policies for the caching allocator.
"""

import threading
import time
from collections import OrderedDict
from typing import Optional


class LRUExpiryPolicy:
    """Least Recently Used (LRU) expiry policy for memory blocks.

    Tracks memory blocks and evicts the least recently used ones when the pool
    exceeds the size limit. Uses an ordered dictionary to maintain access order.
    """

    def __init__(self, max_pool_size_mb: int = 1024):
        """Initialize LRU policy with maximum pool size in megabytes.

        Args:
            max_pool_size_mb: Maximum memory pool size in megabytes before eviction
        """
        self.max_pool_size_bytes = max_pool_size_mb * 1024 * 1024
        self._access_order = OrderedDict()
        self._lock = threading.RLock()
        self._total_pool_size = 0

    def touch(self, block_id: str) -> None:
        """Mark a block as recently used.

        Args:
            block_id: Unique identifier for the memory block
        """
        with self._lock:
            if block_id in self._access_order:
                self._access_order.move_to_end(block_id)
            else:
                self._access_order[block_id] = None

    def add(self, block_id: str, size_bytes: int) -> None:
        """Add a block to the pool tracking.

        Args:
            block_id: Unique identifier for the memory block
            size_bytes: Size of the block in bytes
        """
        with self._lock:
            self._access_order[block_id] = size_bytes
            self._total_pool_size += size_bytes
            self._access_order.move_to_end(block_id)

    def should_evict(self) -> bool:
        """Check if pool size exceeds limit and eviction is needed.

        Returns:
            True if pool size is over limit and eviction should occur
        """
        with self._lock:
            return self._total_pool_size > self.max_pool_size_bytes

    def get_eviction_candidates(self, target_size_bytes: Optional[int] = None) -> list:
        """Get list of block IDs to evict to reach target size.

        Args:
            target_size_bytes: Target pool size after eviction. If None, evict to max_pool_size.

        Returns:
            List of block IDs in eviction order (oldest first)
        """
        with self._lock:
            if target_size_bytes is None:
                target_size_bytes = self.max_pool_size_bytes

            candidates = []
            current_size = self._total_pool_size

            # Iterate from oldest (beginning of OrderedDict)
            for block_id, size in self._access_order.items():
                if current_size <= target_size_bytes:
                    break
                candidates.append(block_id)
                current_size -= size

            return candidates

    def remove(self, block_id: str) -> None:
        """Remove a block from the pool tracking.

        Args:
            block_id: Unique identifier for the memory block
        """
        with self._lock:
            if block_id in self._access_order:
                size = self._access_order[block_id]
                if size is not None:
                    self._total_pool_size -= size
                del self._access_order[block_id]

    def get_pool_size(self) -> int:
        """Get current total pool size in bytes.

        Returns:
            Total size of all blocks in bytes
        """
        with self._lock:
            return self._total_pool_size

    def get_block_count(self) -> int:
        """Get number of blocks in the pool.

        Returns:
            Number of tracked blocks
        """
        with self._lock:
            return len(self._access_order)

    def clear(self) -> None:
        """Clear all blocks from the policy."""
        with self._lock:
            self._access_order.clear()
            self._total_pool_size = 0
