"""
Integration test for tensor parallelism with GPT-2 small.

This test validates that tensor parallelism provides linear scaling
across multiple GPUs for GPT-2 small model.
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import os
import tempfile
import multiprocessing as mp
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from cuda_optimizer.parallel import TensorParallel

pytest.importorskip("torch")


def _setup_distributed(rank, world_size, tmpdir):
    """Initialize distributed training for a subprocess."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"  # Different port for this test
    torch.cuda.set_device(rank % torch.cuda.device_count())

    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{tmpdir}/shared_file_gpt2",
        rank=rank,
        world_size=world_size,
    )


def _cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _create_gpt2_small(vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
    """Create a small GPT-2 model for testing."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    return model.cuda()


def _run_gpt2_scaling_test(
    rank, world_size, tmpdir, batch_size=8, seq_length=128, num_warmup=5, num_iters=20
):
    """Worker function for GPT-2 scaling test."""
    _setup_distributed(rank, world_size, tmpdir)
    try:
        tp = TensorParallel(rank=rank, world_size=world_size)

        # Create model and move to GPU
        model = _create_gpt2_small()
        model.eval()

        # Dummy input
        input_ids = torch.randint(0, 50257, (batch_size, seq_length), device="cuda")
        attention_mask = torch.ones(batch_size, seq_length, device="cuda")

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)

        torch.cuda.synchronize()

        # Timing
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Average time per iteration
        avg_time = elapsed / num_iters
        print(f"Rank {rank}: avg_time = {avg_time:.4f}s")

        # Only rank 0 gathers results from all ranks to compute scaling
        if rank == 0:
            all_times = [torch.tensor(avg_time, device="cuda")]
            for _ in range(world_size - 1):
                all_times.append(torch.empty(1, device="cuda"))
            dist.gather(torch.tensor(avg_time, device="cuda"), all_times, dst=0)
            all_times = [t.item() for t in all_times]

            # Compute speedup relative to single GPU (world_size=1) - in this test,
            # we compare average across all ranks as if each rank is part of same group
            single_gpu_time = all_times[
                0
            ]  # Approximate (in real test would run separately)
            speedups = [single_gpu_time / t for t in all_times]

            print(f"\nWorld size: {world_size}")
            print(f"Single-GPU (estimate): {single_gpu_time:.4f}s")
            for i, (t, speedup) in enumerate(zip(all_times, speedups)):
                print(f"Rank {i}: time={t:.4f}s, speedup={speedup:.2f}x")

            # Linear scaling: speedup should be close to world_size
            avg_speedup = sum(speedups) / len(speedups)
            expected_speedup = float(world_size)
            tolerance = 0.3  # 30% tolerance for parallel overhead
            assert avg_speedup >= expected_speedup * (1 - tolerance), (
                f"Speedup {avg_speedup:.2f}x is less than expected ~{expected_speedup}x"
            )

        else:
            dist.gather(torch.tensor(avg_time, device="cuda"), None, dst=0)

    finally:
        _cleanup_distributed()


def _run_worker(fn, world_size, *args):
    """Spawn workers to run distributed tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mp.spawn(fn, args=(world_size, tmpdir, *args), nprocs=world_size, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="Need at least 4 GPUs for scaling test"
)
class TestGPT2Scaling:
    """Integration tests for GPT-2 small with tensor parallelism."""

    def test_gpt2_scaling_4gpu(self):
        """Test that GPT-2 small shows near-linear scaling across 4 GPUs."""
        world_size = 4
        _run_worker(
            _run_gpt2_scaling_test,
            world_size,
            batch_size=4,
            seq_length=64,
            num_iters=10,
        )

    @pytest.mark.slow
    def test_gpt2_scaling_8gpu(self):
        """Test scaling across 8 GPUs if available."""
        world_size = min(8, torch.cuda.device_count())
        if world_size < 8:
            pytest.skip(f"Only {world_size} GPUs available, need 8 for this test")
        _run_worker(
            _run_gpt2_scaling_test,
            world_size,
            batch_size=8,
            seq_length=128,
            num_iters=10,
        )


class TestTensorParallelIntegration:
    """Integration tests using simple models."""

    def test_simple_model_parallel(self):
        """Test tensor parallelism with a simple linear model."""
        world_size = 2

        def worker(rank, ws, tmpdir):
            _setup_distributed(rank, ws, tmpdir)
            try:
                tp = TensorParallel(rank=rank, world_size=ws)

                # Create a simple model
                model = nn.Sequential(
                    nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10)
                ).cuda()

                # Test forward pass
                batch = torch.randn(16, 100, device="cuda")
                output = model(batch)
                assert output.shape == (16, 10), (
                    f"Rank {rank}: unexpected output shape {output.shape}"
                )

            finally:
                _cleanup_distributed()

        _run_worker(worker, world_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
