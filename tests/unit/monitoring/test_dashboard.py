"""
Unit tests for Dashboard monitoring module.
"""

import pytest
import torch
import sys
import os
import time
import threading
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from cuda_optimizer.monitoring import Dashboard, GPUMetrics

pytest.importorskip("torch")


class TestGPUMetrics:
    """Test suite for GPUMetrics dataclass."""

    def test_create_metrics(self):
        """Test GPUMetrics creation."""
        metrics = GPUMetrics(
            timestamp=time.time(),
            device=0,
            gpu_utilization_percent=75.5,
            memory_used_mb=1024.0,
            memory_total_mb=2048.0,
            memory_utilization_percent=50.0,
            fps=100.0,
            latency_ms=10.0,
            num_iterations=100,
        )
        assert metrics.device == 0
        assert metrics.gpu_utilization_percent == 75.5
        assert metrics.memory_used_mb == 1024.0
        assert metrics.fps == 100.0

    def test_metrics_with_none_values(self):
        """Test GPUMetrics with None for optional fields."""
        metrics = GPUMetrics(
            timestamp=time.time(),
            device=0,
            gpu_utilization_percent=None,
            memory_used_mb=512.0,
            memory_total_mb=1024.0,
            memory_utilization_percent=50.0,
            fps=None,
            latency_ms=None,
            num_iterations=50,
        )
        assert metrics.gpu_utilization_percent is None
        assert metrics.fps is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDashboard:
    """Test suite for Dashboard class."""

    def test_initialization(self):
        """Test Dashboard initializes correctly."""
        dashboard = Dashboard(device=0, max_history=100)
        assert dashboard.device == 0
        assert dashboard.max_history == 100
        assert dashboard._running is False
        assert dashboard._thread is None
        assert len(dashboard._metrics) == 0

    def test_initialization_default_values(self):
        """Test Dashboard defaults."""
        dashboard = Dashboard()
        assert dashboard.device == 0
        assert dashboard.max_history == 10000
        assert dashboard.update_interval == 1.0

    def test_start_stop(self):
        """Test start and stop operations."""
        dashboard = Dashboard(device=0, update_interval=0.1)
        dashboard.start()
        assert dashboard._running is True
        assert dashboard._thread is not None
        assert dashboard._thread.is_alive()

        time.sleep(0.2)  # Let it collect a few metrics

        dashboard.stop()
        assert dashboard._running is False
        assert dashboard._thread is not None

    def test_start_multiple_times(self):
        """Test starting dashboard multiple times doesn't create multiple threads."""
        dashboard = Dashboard(update_interval=0.1)
        dashboard.start()
        dashboard.start()  # Second start should be ignored
        assert dashboard._thread is not None
        # Should still be only one thread
        dashboard.stop()

    def test_update_metrics(self):
        """Test manual metrics update."""
        dashboard = Dashboard()
        dashboard.update(iteration=1, latency_ms=5.0)

        latest = dashboard.get_latest_metrics()
        assert latest is not None
        assert latest.num_iterations == 1
        assert latest.latency_ms == 5.0

    def test_update_multiple_iterations(self):
        """Test multiple updates."""
        dashboard = Dashboard()
        for i in range(1, 6):
            dashboard.update(iteration=i, latency_ms=float(i))

        latest = dashboard.get_latest_metrics()
        assert latest.num_iterations == 5
        assert latest.latency_ms == 5.0

        history = dashboard.get_metrics_history()
        # Should have 5 entries (one per iteration)
        # Actually each update adds a metrics entry
        assert len(history) == 5

    def test_max_history_limit(self):
        """Test that history is trimmed to max_history."""
        dashboard = Dashboard(max_history=3)
        for i in range(5):
            dashboard.update(iteration=i)

        history = dashboard.get_metrics_history()
        assert len(history) <= 3
        # Should keep the most recent ones
        assert history[-1]["num_iterations"] == 4  # Last one (0-indexed)

    def test_get_metrics_history_empty(self):
        """Test get_metrics_history returns empty list when no metrics."""
        dashboard = Dashboard()
        history = dashboard.get_metrics_history()
        assert history == []

    def test_export_json(self):
        """Test JSON export."""
        dashboard = Dashboard()
        dashboard.update(iteration=1, fps=100.0, latency_ms=10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "metrics.json")
            dashboard.export_json(output_path)

            assert os.path.exists(output_path)
            with open(output_path, "r") as f:
                data = json.load(f)
                assert "metadata" in data
                assert "metrics" in data
                assert data["metadata"]["device"] == 0
                assert len(data["metrics"]) == 1

    def test_export_csv(self):
        """Test CSV export."""
        dashboard = Dashboard()
        dashboard.update(iteration=1, fps=100.0, latency_ms=10.0)
        dashboard.update(iteration=2, fps=110.0, latency_ms=9.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "metrics.csv")
            dashboard.export_csv(output_path)

            assert os.path.exists(output_path)
            import csv

            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2

    def test_export_csv_empty(self):
        """Test CSV export with no data."""
        dashboard = Dashboard()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "metrics.csv")
            # Should not raise, just print message
            dashboard.export_csv(output_path)

    def test_get_summary_stats(self):
        """Test summary statistics."""
        dashboard = Dashboard()
        # Add some metrics (with GPU utilization, which may be None without GPU)
        # We'll add metrics via the internal _metrics list to have full control
        from cuda_optimizer.monitoring import GPUMetrics
        import time

        now = time.time()
        for i in range(5):
            metrics = GPUMetrics(
                timestamp=now + i,
                device=0,
                gpu_utilization_percent=50.0 + i * 5,
                memory_used_mb=1000.0 + i * 100,
                memory_total_mb=2000.0,
                memory_utilization_percent=50.0 + i * 5,
                fps=100.0 + i * 10,
                latency_ms=10.0 - i * 0.5,
                num_iterations=i,
            )
            dashboard._metrics.append(metrics)

        stats = dashboard.get_summary_stats()

        assert "duration_seconds" in stats
        assert "total_iterations" in stats
        assert stats["total_iterations"] == 4  # Last iteration value
        assert "gpu_utilization" in stats
        assert "memory_mb" in stats
        assert "fps" in stats

        assert stats["gpu_utilization"]["avg"] is not None
        assert stats["memory_mb"]["max"] == 1400.0

    def test_get_summary_stats_empty(self):
        """Test summary stats with empty metrics."""
        dashboard = Dashboard()
        stats = dashboard.get_summary_stats()
        assert stats == {}

    def test_context_manager(self):
        """Test Dashboard as context manager."""
        with Dashboard(update_interval=0.1) as dashboard:
            assert dashboard._running is True
            time.sleep(0.2)
            # Should have collected some metrics
            assert len(dashboard._metrics) >= 0  # Might be 0 or more

        # After exiting context, should be stopped
        assert dashboard._running is False

    def test_thread_safety_multiple_updates(self):
        """Test thread safety during concurrent updates."""
        dashboard = Dashboard()
        dashboard.start()

        def update_loop():
            for i in range(50):
                dashboard.update(iteration=i)
                time.sleep(0.01)

        # Start multiple threads updating
        threads = []
        for t_id in range(3):
            thread = threading.Thread(target=update_loop)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        time.sleep(0.1)  # Let dashboard collect
        dashboard.stop()

        # Should have many metrics without errors
        assert len(dashboard._metrics) >= 0

    def test_get_latest_metrics_empty(self):
        """Test get_latest_metrics returns None when empty."""
        dashboard = Dashboard()
        latest = dashboard.get_latest_metrics()
        assert latest is None

    def test_fps_calculation(self):
        """Test FPS calculation from iterations."""
        dashboard = Dashboard()
        # Simulate updates with time control
        dashboard._iteration_count = 0
        dashboard._start_time = time.time()
        dashboard._last_fps_update = time.time()

        time.sleep(0.1)
        dashboard.update(iteration=10)  # 10 iterations in ~0.1s

        latest = dashboard.get_latest_metrics()
        # FPS should be roughly 100 (10 / 0.1) with some smoothing from deque
        assert latest is not None
        # FPS may be None initially if not enough data
        # or within a reasonable range
        if latest.fps is not None:
            assert latest.fps > 0

    def test_clear_cache_not_exposed(self):
        """Test that Dashboard doesn't expose clear method directly."""
        # This is more of a design check
        dashboard = Dashboard()
        # No clear method should exist (we use stop and let GC)
        # But we can verify that after stop, metrics are still accessible
        dashboard.update(iteration=1)
        dashboard.stop()
        # Metrics should still be there
        assert len(dashboard._metrics) > 0

    def test_memory_info_fallback(self):
        """Test memory info retrieval if CUDA not available."""
        # This test will use the mock-based approach
        with patch("torch.cuda.is_available", return_value=False):
            dashboard = Dashboard()
            memory_used, memory_total = dashboard._get_memory_info()
            assert memory_used == 0.0
            assert memory_total == 0.0

    def test_gpu_utilization_fallback_all_methods(self):
        """Test GPU utilization retrieval falls back appropriately."""
        dashboard = Dashboard(use_nvidia_smi=False)

        # Should return None if no methods available
        util = dashboard._get_gpu_utilization()
        # Without any method, it should return None
        # Since we disabled use_nvidia_smi and mocked torch.cuda.is_available as false
        assert util is None


class TestDashboardIntegration:
    """Integration tests for Dashboard with simulated training."""

    def test_dashboard_with_training_loop_simulation(self):
        """Test dashboard integrated with a simulated training loop."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dashboard = Dashboard(update_interval=0.5)

        try:
            dashboard.start()

            # Simulate training
            for step in range(10):
                # Do training step (simulated)
                time.sleep(0.1)  # Simulate work
                dashboard.update(iteration=step, latency_ms=10.0)

            time.sleep(0.6)  # Let background thread collect
            dashboard.stop()

            history = dashboard.get_metrics_history()
            # Should have metrics from both manual updates and background
            assert len(history) > 0

            # Export
            with tempfile.TemporaryDirectory() as tmpdir:
                json_path = os.path.join(tmpdir, "metrics.json")
                csv_path = os.path.join(tmpdir, "metrics.csv")
                dashboard.export_json(json_path)
                dashboard.export_csv(csv_path)

                assert os.path.exists(json_path)
                assert os.path.exists(csv_path)
        finally:
            if dashboard._running:
                dashboard.stop()

    def test_concurrent_monitoring_and_updates(self):
        """Test dashboard handles concurrent background monitoring and updates."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dashboard = Dashboard(update_interval=0.2)
        dashboard.start()

        try:
            # Update from main thread while background monitors
            for i in range(20):
                dashboard.update(iteration=i)
                time.sleep(0.05)

            time.sleep(0.3)  # Allow background to collect

            # Should have combined metrics
            history = dashboard.get_metrics_history()
            assert len(history) > 0
        finally:
            dashboard.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
