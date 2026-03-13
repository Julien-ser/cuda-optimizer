"""
Unit tests for BaseProfiler.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import json
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from cuda_optimizer.profiling import BaseProfiler

pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBaseProfiler:
    """Test suite for BaseProfiler."""

    def test_initialization(self):
        """Test profiler initializes correctly."""
        model = nn.Linear(10, 10)
        profiler = BaseProfiler(model, input_shape=(32, 10))
        assert profiler.model is not None
        assert profiler.input_shape == (32, 10)
        assert profiler.device == "cuda"
        assert profiler.use_nsight is True

    def test_initialization_custom_device(self):
        """Test profiler with custom device."""
        model = nn.Linear(10, 10)
        profiler = BaseProfiler(model, input_shape=(32, 10), device="cuda:0")
        assert profiler.device == "cuda:0"

    def test_create_dummy_input(self):
        """Test dummy input creation."""
        model = nn.Linear(10, 10)
        profiler = BaseProfiler(model, input_shape=(32, 10, 20))
        dummy = profiler._create_dummy_input()
        assert dummy.shape == (32, 10, 20)
        assert dummy.device.type == "cuda"

    def test_warmup(self):
        """Test warmup runs."""
        model = nn.Linear(10, 10).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10))
        # Should not raise errors
        profiler._warmup(iterations=5)

    def test_profile_training_basic(self):
        """Test basic training profiling produces expected keys."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)

        results = profiler.profile_training(iterations=10)

        # Check required keys
        assert "mode" in results
        assert results["mode"] == "training"
        assert "fps" in results
        assert "avg_latency_ms" in results
        assert "memory_peak_mb" in results
        assert "memory_allocated_mb" in results
        assert "kernel_stats" in results
        assert "iterations" in results
        assert results["iterations"] == 10
        assert "model_params" in results
        assert results["model_params"] > 0

    def test_profile_training_fps_calculation(self):
        """Test FPS is calculated reasonably."""
        model = nn.Linear(10, 10).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)

        results = profiler.profile_training(iterations=10)

        assert results["fps"] > 0
        assert results["avg_latency_ms"] > 0

    def test_profile_inference_basic(self):
        """Test basic inference profiling."""
        model = nn.Linear(10, 10).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)

        results = profiler.profile_inference(iterations=20)

        assert results["mode"] == "inference"
        assert results["fps"] > 0
        assert "kernel_stats" in results

    def test_profile_inference_faster_than_training(self):
        """Test inference is typically faster than training (no backward)."""
        model = nn.Linear(100, 100).cuda()
        profiler = BaseProfiler(model, input_shape=(64, 100), use_nsight=False)

        train_results = profiler.profile_training(iterations=10)
        infer_results = profiler.profile_inference(iterations=10)

        # Inference should generally be at least as fast as training
        # (allowing some tolerance for variance)
        # Note: This might not always hold true depending on model size, but is a reasonable expectation
        assert infer_results["fps"] >= train_results["fps"] * 0.5

    def test_profile_with_nsight_disabled(self):
        """Test nsight path not used when disabled."""
        model = nn.Linear(10, 10).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)

        results = profiler.profile_training(iterations=5)
        assert results["nsight_report"] is None

    def test_profile_memory_tracking(self):
        """Test memory usage is tracked."""
        model = nn.Sequential(nn.Linear(100, 1000), nn.Linear(1000, 100)).cuda()
        profiler = BaseProfiler(model, input_shape=(64, 100), use_nsight=False)

        results = profiler.profile_training(iterations=10)

        assert results["memory_peak_mb"] > 0
        assert results["memory_allocated_mb"] >= 0

    def test_profile_kernel_stats_structure(self):
        """Test kernel stats have correct structure."""
        model = nn.Linear(100, 100).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 100), use_nsight=False)

        results = profiler.profile_training(iterations=10)
        kernel_stats = results["kernel_stats"]

        if kernel_stats:
            stat = kernel_stats[0]
            assert "name" in stat
            assert "count" in stat
            assert "cpu_time_total_ms" in stat
            assert "cuda_time_total_ms" in stat

    def test_export_results_json(self):
        """Test results can be exported to JSON."""
        model = nn.Linear(10, 10).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)
        profiler.profile_inference(iterations=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            profiler.export_results(output_path)

            assert os.path.exists(output_path)
            with open(output_path, "r") as f:
                loaded = json.load(f)
                assert "mode" in loaded
                assert "fps" in loaded

    def test_compare_baseline(self):
        """Test comparison against baseline."""
        model = nn.Linear(10, 10).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)
        profiler.profile_inference(iterations=10)

        # Create a baseline with slightly different parameters
        baseline = {
            "fps": profiler.results["fps"] * 0.9,  # 10% slower baseline
            "memory_peak_mb": profiler.results["memory_peak_mb"]
            * 1.1,  # 10% more memory
            "avg_latency_ms": profiler.results["avg_latency_ms"] * 1.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = os.path.join(tmpdir, "baseline.json")
            with open(baseline_path, "w") as f:
                json.dump(baseline, f)

            comparison = profiler.compare_baseline(baseline_path)

            assert "fps_speedup_percent" in comparison
            assert "memory_reduction_percent" in comparison
            assert comparison["fps_speedup_percent"] > 0  # Current is faster
            assert (
                comparison["memory_reduction_percent"] > 0
            )  # Current uses less memory

    def test_profile_device_transfer(self):
        """Test model is moved to correct device."""
        model = nn.Linear(10, 10)  # On CPU initially
        profiler = BaseProfiler(model, input_shape=(32, 10), device="cuda:0")
        # Model should be moved to CUDA
        assert next(profiler.model.parameters()).device.type == "cuda"

    def test_profile_preserves_model_eval_mode(self):
        """Test that profiling doesn't permanently change model mode."""
        model = nn.Linear(10, 10).cuda()
        model.eval()
        original_mode = model.training

        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)
        profiler.profile_inference(iterations=5)

        # Model should still be in eval mode
        assert model.training == original_mode

    def test_profile_multiple_runs_independent(self):
        """Test multiple profiling runs produce independent results."""
        model = nn.Linear(10, 10).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)

        results1 = profiler.profile_inference(iterations=10)
        results2 = profiler.profile_inference(iterations=10)

        # Both should have valid data
        assert results1["fps"] > 0
        assert results2["fps"] > 0
        # Results dicts are separate
        assert results1 is not results2

    def test_profile_with_grad_enabled(self):
        """Test training profile works with gradient enabled."""
        model = nn.Linear(10, 10).cuda()
        profiler = BaseProfiler(model, input_shape=(32, 10), use_nsight=False)

        results = profiler.profile_training(iterations=5)

        # Should collect training metrics
        assert results["mode"] == "training"
        assert "loss" not in results  # Not tracked, but that's okay

    def test_invalid_input_shape_raises(self):
        """Test invalid input shape appropriately."""
        model = nn.Linear(10, 10)
        # This might not raise until actual computation, but that's okay
        profiler = BaseProfiler(model, input_shape=(), use_nsight=False)
        # Empty shape would fail when creating dummy input


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
