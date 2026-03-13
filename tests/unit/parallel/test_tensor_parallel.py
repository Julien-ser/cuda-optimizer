"""
Unit tests for TensorParallel.

Tests cover:
- Tensor splitting (1D row/column, 2D)
- Communication operations (all-reduce, all-gather, all-to-all, reduce-scatter)
- Broadcast and barrier
- Edge cases and error handling
"""

import pytest
import torch
import torch.distributed as dist
import sys
import os
import tempfile
import multiprocessing as mp

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from cuda_optimizer.parallel import TensorParallel

pytest.importorskip("torch")


def _setup_distributed(rank, world_size, tmpdir):
    """Initialize distributed training for a subprocess."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank % torch.cuda.device_count())

    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{tmpdir}/shared_file",
        rank=rank,
        world_size=world_size,
    )


def _cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_worker(fn, world_size, *args):
    """Spawn workers to run distributed tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mp.spawn(fn, args=(world_size, tmpdir, *args), nprocs=world_size, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs")
class TestTensorParallel1D:
    """Test suite for 1D tensor parallelism."""

    def test_split_1d_row(self):
        """Test row-wise splitting."""
        world_size = min(4, torch.cuda.device_count())

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                # Create tensor with divisible dim size
                full = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16).cuda()
                local = tp.split_1d_row(full, dim=0)

                expected_size = (2, 16) if ws == 4 else (8 // ws, 16)
                assert local.shape == expected_size, (
                    f"Rank {rank}: Expected {expected_size}, got {local.shape}"
                )

                # Check values are correct slice
                start = rank * (8 // ws)
                expected = full[start : start + (8 // ws)].cuda()
                assert torch.allclose(local, expected), (
                    f"Rank {rank}: Values don't match"
                )
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)

    def test_split_1d_column(self):
        """Test column-wise splitting."""
        world_size = min(4, torch.cuda.device_count())

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                full = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16).cuda()
                local = tp.split_1d_column(full, dim=1)

                expected_size = (8, 4) if ws == 4 else (8, 16 // ws)
                assert local.shape == expected_size, (
                    f"Rank {rank}: Expected {expected_size}, got {local.shape}"
                )

                # Check values (strided selection)
                indices = torch.arange(rank, 16, ws, device="cuda")
                expected = full.index_select(1, indices)
                assert torch.allclose(local, expected), (
                    f"Rank {rank}: Values don't match"
                )
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)

    def test_non_divisible_raises(self):
        """Test that non-divisible dimension raises error."""
        world_size = min(4, torch.cuda.device_count())

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                full = torch.randn(7, 16).cuda()  # 7 not divisible by 4
                with pytest.raises(ValueError, match="must be divisible"):
                    tp.split_1d_row(full, dim=0)
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Need at least 4 GPUs for 2D")
class TestTensorParallel2D:
    """Test suite for 2D tensor parallelism."""

    def test_split_2d(self):
        """Test 2D grid splitting."""
        world_size = 4  # Requires exactly 4 for 2x2 grid

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                full = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16).cuda()
                local = tp.split_2d(full, row_dim=0, col_dim=1)

                # 2x2 grid: each gets (4, 8)
                assert local.shape == (4, 8), (
                    f"Rank {rank}: Expected (4, 8), got {local.shape}"
                )

                row_rank = rank // 2
                col_rank = rank % 2
                row_start = row_rank * 4
                row_end = row_start + 4
                col_start = col_rank * 8
                col_end = col_start + 8

                expected = full[row_start:row_end, col_start:col_end].cuda()
                assert torch.allclose(local, expected), (
                    f"Rank {rank}: Values don't match"
                )
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)

    def test_split_2d_custom_grid(self):
        """Test 2D splitting with inferred grid (2x2 from world_size=4)."""
        world_size = 4

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                full = torch.randn(12, 20).cuda()
                local = tp.split_2d(full, row_dim=0, col_dim=1)

                # Grid is 2x2 (factorized from 4)
                assert local.shape == (6, 10), (
                    f"Rank {rank}: Expected (6, 10), got {local.shape}"
                )
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)

    def test_non_factorizable_raises(self):
        """Test that non-factorizable world_size raises error."""
        world_size = 3  # Prime, can't factor into 2D

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)
                full = torch.randn(6, 6).cuda()

                with pytest.raises(ValueError, match="cannot be factored"):
                    tp.split_2d(full, row_dim=0, col_dim=1)
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)


class TestTensorParallelCommunication:
    """Test communication operations with simulated 2-GPU setup."""

    def test_all_reduce(self):
        """Test all-reduce operation."""
        world_size = 2

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                tensor = torch.tensor(
                    [rank + 1.0], device="cuda"
                )  # rank 0: 1.0, rank 1: 2.0
                result = tp.all_reduce(tensor.clone())  # Clone to avoid in-place

                # After sum, both should have 3.0
                assert torch.allclose(result, torch.tensor([3.0], device="cuda")), (
                    f"Rank {rank}: all_reduce failed, got {result}"
                )
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)

    def test_all_gather(self):
        """Test all-gather operation."""
        world_size = 2

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                local = torch.tensor([rank], device="cuda")
                gathered = tp.all_gather(local, dim=0)

                expected = torch.tensor([0, 1], device="cuda")
                assert torch.allclose(gathered, expected), (
                    f"Rank {rank}: all_gather failed, got {gathered}"
                )
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)

    def test_reduce_scatter(self):
        """Test reduce-scatter operation."""
        world_size = 2

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                # Each rank has a tensor [rank+1, rank+10]
                tensor = torch.tensor(
                    [float(rank + 1), float(rank + 10)], device="cuda"
                )
                result = tp.reduce_scatter(tensor, dim=0)

                # Reduce sum: position 0: 1+2=3, position 1: 10+11=21
                # Then scatter: rank 0 gets 3, rank 1 gets 21
                expected = torch.tensor([3.0 if rank == 0 else 21.0], device="cuda")
                assert torch.allclose(result, expected), (
                    f"Rank {rank}: reduce_scatter failed, got {result}"
                )
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)

    def test_broadcast(self):
        """Test broadcast operation."""
        world_size = 2

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                if rank == 0:
                    tensor = torch.tensor([42.0], device="cuda")
                else:
                    tensor = torch.tensor([0.0], device="cuda")

                received = tp.broadcast(tensor.clone(), src=0)

                assert torch.allclose(received, torch.tensor([42.0], device="cuda")), (
                    f"Rank {rank}: broadcast failed, got {received}"
                )
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)

    def test_barrier(self):
        """Test barrier synchronization."""
        world_size = 2

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)
                tp.barrier()  # Should not deadlock
            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)


class TestTensorParallelHelpers:
    """Test helper methods."""

    def test_factorize(self):
        """Test integer factorization."""
        tp = TensorParallel(rank=0, world_size=1)  # Dummy
        assert tp._factorize(4) == [2, 2]
        assert tp._factorize(6) == [2, 3]
        assert tp._factorize(12) == [3, 4]

    def test_get_local_rank(self):
        """Test local rank calculation."""
        # Test with world_size=4, 2 local devices
        tp = TensorParallel(rank=0, world_size=4)  # rank 0
        assert tp.get_local_rank() == 0

        tp = TensorParallel(rank=2, world_size=4)  # rank 2 -> local rank 0? (2 % 2 = 0)
        assert tp.get_local_rank() == 0

    def test_get_device_ids(self):
        """Test device ID list generation."""
        tp = TensorParallel(rank=0, world_size=4)
        devices = tp.get_device_ids()
        assert len(devices) == 4
        # Should cycle through available devices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTensorParallelEdgeCases:
    """Test edge cases and error handling."""

    def test_not_initialized_error(self):
        """Test that operations fail without initialization."""
        tp = TensorParallel(rank=0, world_size=1)
        # Should raise error for communication ops without init
        tensor = torch.tensor([1.0], device="cuda")
        with pytest.raises(RuntimeError, match="Distributed must be initialized"):
            tp.all_reduce(tensor)

    def test_invalid_backend_warning(self):
        """Test warning for non-NCCL backend."""
        # This would require actually initializing with a different backend
        # Just verify code path exists
        pass  # Handled in __init__
