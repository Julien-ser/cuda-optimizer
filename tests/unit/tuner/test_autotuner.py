"""
Unit tests for the kernel autotuner.

Tests cover:
- Cache management (load/save/clear)
- Configuration key generation
- Block size and grid size calculations
- Autotuning logic (with mocked CUDA operations)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import torch

from cuda_optimizer.tuner import Autotuner


class TestAutotunerCache:
    """Test cache management functionality."""

    def setup_method(self):
        """Set up test fixture."""
        self.temp_dir = tempfile.mkdtemp()
        self.autotuner = Autotuner(cache_dir=self.temp_dir, verbose=False)

    def teardown_method(self):
        """Clean up test fixture."""
        shutil.rmtree(self.temp_dir)

    def test_cache_init_creates_directory(self):
        """Cache directory should be created on init."""
        assert Path(self.temp_dir).exists()

    def test_cache_save_and_load(self):
        """Configurations should be saved and loaded correctly."""
        # Add a configuration manually
        test_config = {
            "block_size": 256,
            "grid_x": 100,
            "grid_y": 1,
            "execution_time_ms": 1.5,
        }
        self.autotuner.config_cache["test_kernel"] = test_config
        self.autotuner._save_cache()

        # Create new autotuner to test loading
        new_autotuner = Autotuner(cache_dir=self.temp_dir, verbose=False)
        assert "test_kernel" in new_autotuner.config_cache
        assert new_autotuner.config_cache["test_kernel"] == test_config

    def test_cache_clear(self):
        """Clear should remove all cached configurations."""
        self.autotuner.config_cache = {"kernel1": {}, "kernel2": {}}
        self.autotuner._save_cache()
        assert self.autotuner.cache_file.exists()

        self.autotuner.clear_cache()
        assert len(self.autotuner.config_cache) == 0
        assert not self.autotuner.cache_file.exists()

    def test_list_cached_kernels(self):
        """Should list all cached kernel names."""
        self.autotuner.config_cache = {
            "kernel_a": {},
            "kernel_b": {},
            "kernel_c": {},
        }
        kernels = self.autotuner.list_cached_kernels()
        assert len(kernels) == 3
        assert "kernel_a" in kernels

    def test_get_cached_config(self):
        """Should return configuration for existing kernel, None otherwise."""
        test_config = {"block_size": 128}
        self.autotuner.config_cache["existing"] = test_config

        assert self.autotuner.get_cached_config("existing") == test_config
        assert self.autotuner.get_cached_config("nonexistent") is None


class TestAutotunerConfigKey:
    """Test configuration key generation."""

    def setup_method(self):
        self.autotuner = Autotuner(verbose=False)

    def test_generate_cache_key_consistency(self):
        """Same inputs should produce same key."""
        key1 = self.autotuner._generate_cache_key(
            "test_kernel", (8, 256, 768), "float16"
        )
        key2 = self.autotuner._generate_cache_key(
            "test_kernel", (8, 256, 768), "float16"
        )
        assert key1 == key2

    def test_generate_cache_key_different_shapes(self):
        """Different shapes should produce different keys."""
        key1 = self.autotuner._generate_cache_key(
            "test_kernel", (8, 256, 768), "float16"
        )
        key2 = self.autotuner._generate_cache_key(
            "test_kernel", (16, 256, 768), "float16"
        )
        assert key1 != key2

    def test_generate_cache_key_different_kernels(self):
        """Different kernel names should produce different keys."""
        key1 = self.autotuner._generate_cache_key("kernel_a", (8, 256, 768), "float16")
        key2 = self.autotuner._generate_cache_key("kernel_b", (8, 256, 768), "float16")
        assert key1 != key2

    def test_generate_cache_key_different_dtypes(self):
        """Different dtypes should produce different keys."""
        key1 = self.autotuner._generate_cache_key(
            "test_kernel", (8, 256, 768), "float16"
        )
        key2 = self.autotuner._generate_cache_key(
            "test_kernel", (8, 256, 768), "float32"
        )
        assert key1 != key2


class TestAutotunerBlockGridCalculation:
    """Test block size and grid size calculations."""

    def setup_method(self):
        self.autotuner = Autotuner(verbose=False)

    def test_suggest_block_sizes(self):
        """Should return expected block size list."""
        sizes = self.autotuner._suggest_block_sizes()
        assert sizes == [32, 64, 128, 256, 512, 1024]

    def test_calculate_grid_size_simple(self):
        """Should calculate correct grid size for 1D problem."""
        grid = self.autotuner._calculate_grid_size(total_elements=1000, block_size=256)
        assert grid == (4, 1)  # ceil(1000/256) = 4

    def test_calculate_grid_size_with_features(self):
        """Should calculate correct 2D grid size."""
        grid = self.autotuner._calculate_grid_size(
            total_elements=1000, block_size=256, num_features=10
        )
        # batch_blocks = ceil(1000/256) = 4, num_features = 10
        assert grid == (4, 10)

    def test_calculate_grid_size_rounding(self):
        """Should round up correctly."""
        grid = self.autotuner._calculate_grid_size(total_elements=256, block_size=256)
        assert grid == (1, 1)

        grid = self.autotuner._calculate_grid_size(total_elements=257, block_size=256)
        assert grid == (2, 1)


class TestAutotunerAutotuneOperation:
    """Test the autotune_operation method with mocked CUDA."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.autotuner = Autotuner(cache_dir=self.temp_dir, verbose=False, num_trials=3)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autotune_operation_real_cuda(self):
        """Test autotuning with real CUDA operations if available."""

        # Create a simple CUDA operation
        def simple_op():
            a = torch.randn(1000, 1000, device="cuda", dtype=torch.float16)
            b = torch.randn(1000, 1000, device="cuda", dtype=torch.float16)
            return torch.matmul(a, b)

        config = self.autotuner.autotune_operation(
            operation=simple_op,
            kernel_name="test_matmul",
            test_data=(torch.randn(1000, 1000, device="cuda"),),
        )

        assert "execution_time_ms" in config
        assert "block_size" in config
        assert config["execution_time_ms"] > 0

    def test_autotune_operation_mocked(self):
        """Test autotuning with mocked timing."""
        with patch("time.perf_counter") as mock_time:
            # Configure mock to return increasing times
            mock_time.side_effect = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]

            def mock_op():
                pass

            config = self.autotuner.autotune_operation(
                operation=mock_op,
                kernel_name="mocked_op",
                test_data=(),
            )

        assert "execution_time_ms" in config
        assert "block_size" in config

    def test_autotune_operation_caches_result(self):
        """Autotune should cache the best configuration."""
        call_count = 0

        def counting_op():
            nonlocal call_count
            call_count += 1

        self.autotuner.autotune_operation(
            operation=counting_op,
            kernel_name="counting_op",
            test_data=(),
        )

        # Check that the config was cached
        cached = self.autotuner.get_cached_config("counting_op")
        assert cached is not None
        assert "execution_time_ms" in cached

        # Running again should use cached config (fewer actual runs)
        call_count_after = call_count
        self.autotuner.autotune_operation(
            operation=counting_op,
            kernel_name="counting_op",
            test_data=(),
        )
        # Should have returned cached result without re-running
        assert call_count == call_count_after


class TestAutotunerIntegration:
    """Integration tests with mock kernels."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.autotuner = Autotuner(cache_dir=self.temp_dir, verbose=False)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_autotune_with_multiple_configs(self):
        """Test autotuning across multiple block sizes."""
        times = [0.01, 0.009, 0.011, 0.008, 0.012, 0.007]  # Mock times
        time_iter = iter(times)

        def varying_op():
            # Simulate different execution times based on block size
            pass

        with patch("time.perf_counter", side_effect=lambda: next(time_iter)):
            config = self.autotuner.autotune_operation(
                operation=varying_op,
                kernel_name="varying_op",
                test_data=(),
            )

        assert config is not None
        # Best config should have smallest time
        assert config["execution_time_ms"] == min(times)

    def test_benchmark_all_cached(self):
        """Should return all cached configurations."""
        self.autotuner.config_cache = {
            "kernel1": {"execution_time_ms": 1.0},
            "kernel2": {"execution_time_ms": 2.0},
        }

        all_configs = self.autotuner.benchmark_all_cached()
        assert len(all_configs) == 2
        assert all_configs["kernel1"]["execution_time_ms"] == 1.0
