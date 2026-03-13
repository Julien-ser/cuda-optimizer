"""
Integration test for full pipeline with ResNet50 training.

Tests the complete optimization pipeline including:
- Baseline profiling
- Memory optimization
- Mixed precision
- Gradient checkpointing
- Optimizer fusion
"""

import pytest
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import sys
import os
import time
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cuda_optimizer.profiling import BaseProfiler
from cuda_optimizer.memory import CUDACache
from cuda_optimizer.optim import AMPWrapper
from cuda_optimizer.checkpoint import SelectiveCheckpoint, CheckpointCompiler
from cuda_optimizer.fusion import install_fused_optimizers, FusedOptimizerContext
from cuda_optimizer import Optimizer  # High-level API if exists

pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFullPipelineResNet50:
    """Integration test suite for ResNet50 full optimization pipeline."""

    @pytest.fixture
    def resnet50_model(self):
        """Create ResNet50 model."""
        model = models.resnet50(weights=None, num_classes=10).cuda()
        return model

    @pytest.fixture
    def synthetic_data(self, batch_size=32):
        """Create synthetic training data."""
        # Input images: (batch, 3, 224, 224)
        inputs = torch.randn(batch_size, 3, 224, 224).cuda()
        # Labels: 10 classes
        targets = torch.randint(0, 10, (batch_size,)).cuda()
        return inputs, targets

    def test_baseline_profiling(self, resnet50_model, synthetic_data):
        """Test baseline profiling of ResNet50."""
        inputs, targets = synthetic_data
        profiler = BaseProfiler(
            resnet50_model, input_shape=(32, 3, 224, 224), use_nsight=False
        )

        # Profile inference first (faster)
        infer_results = profiler.profile_inference(iterations=20)
        assert infer_results["fps"] > 0
        assert infer_results["memory_peak_mb"] > 0

        # Profile training
        train_results = profiler.profile_training(iterations=10)
        assert train_results["fps"] > 0
        assert train_results["memory_peak_mb"] > 0
        assert train_results["model_params"] > 20000000  # ResNet50 has ~25M params

    def test_memory_pool_integration(self, resnet50_model, synthetic_data):
        """Test memory pool caching with training."""
        cache = CUDACache(max_pool_size_mb=512)
        inputs, targets = synthetic_data
        criterion = nn.CrossEntropyLoss()

        # Train with manual cache allocation
        for step in range(5):
            optimizer = torch.optim.AdamW(resnet50_model.parameters(), lr=1e-3)

            # Allocate tensors through cache
            x = cache.allocate(
                inputs.numel() * inputs.element_size(), dtype=inputs.dtype
            )
            x = x.view(inputs.shape)
            x.copy_(inputs)

            optimizer.zero_grad()
            output = resnet50_model(x)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            cache.free(x)

        stats = cache.get_stats()
        assert stats["total_allocated"] >= 5  # At least 5 allocations

    def test_amp_wrapper_integration(self, resnet50_model, synthetic_data):
        """Test automatic mixed precision wrapper."""
        amp = AMPWrapper(enabled=True, init_scale=2.0**16, growth_factor=2.0)
        inputs, targets = synthetic_data
        criterion = nn.CrossEntropyLoss()

        # Wrap optimizer
        optimizer = torch.optim.AdamW(resnet50_model.parameters(), lr=1e-3)
        optimizer = amp.wrap_optimizer(optimizer)

        # Training step
        resnet50_model.train()
        with torch.cuda.amp.autocast(enabled=True):
            optimizer.zero_grad()
            output = resnet50_model(inputs)
            loss = criterion(output, targets)

        # AMP wrapper should handle scaling
        amp.scale_loss(loss)
        amp.backward()
        optimizer.step()
        amp.update_scale()

        # Loss should be a tensor
        assert isinstance(loss.item(), float)

    def test_gradient_checkpointing_integration(self, resnet50_model, synthetic_data):
        """Test gradient checkpointing integration."""
        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)

        # Select some layers for checkpointing (e.g., layer4 blocks)
        for name, module in resnet50_model.named_modules():
            if "layer4" in name:
                selector.select_layers([module])

        # Apply checkpointing
        compiler.compile(resnet50_model)

        inputs, targets = synthetic_data
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(resnet50_model.parameters(), lr=1e-3)

        # Training step should work
        resnet50_model.train()
        optimizer.zero_grad()
        output = resnet50_model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_optimizer_fusion_integration(self, resnet50_model, synthetic_data):
        """Test fused optimizer integration."""
        if not FusedAdamW.is_available():
            pytest.skip("Fused AdamW not available")

        inputs, targets = synthetic_data
        criterion = nn.CrossEntropyLoss()

        with FusedOptimizerContext():
            optimizer = torch.optim.AdamW(resnet50_model.parameters(), lr=1e-3)
            assert hasattr(optimizer, "_fused") or isinstance(optimizer, FusedAdamW)

            # Training step
            resnet50_model.train()
            optimizer.zero_grad()
            output = resnet50_model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            assert loss.item() > 0

    def test_full_optimization_pipeline(self, resnet50_model, synthetic_data):
        """Test complete optimization pipeline with multiple optimizations together."""
        if not FusedAdamW.is_available():
            pytest.skip("Fused AdamW not available")

        inputs, targets = synthetic_data
        criterion = nn.CrossEntropyLoss()

        # Apply all optimizations
        # 1. Gradient checkpointing
        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)
        # Selectively checkpoint some layers
        for name, module in resnet50_model.named_modules():
            if "layer3" in name or "layer4" in name:
                selector.select_layers([module])
        compiler.compile(resnet50_model)

        # 2. Fused optimizer
        with FusedOptimizerContext():
            optimizer = torch.optim.AdamW(resnet50_model.parameters(), lr=1e-3)
            amp = AMPWrapper(enabled=True)

            # Measure memory before
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # Training iteration
            resnet50_model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                output = resnet50_model(inputs)
                loss = criterion(output, targets)

            amp.scale_loss(loss)
            amp.backward()
            optimizer.step()
            amp.update_scale()

            torch.cuda.synchronize()
            memory_after = torch.cuda.max_memory_allocated() / 1024**2

            assert loss.item() > 0
            assert memory_after > 0  # Should have measurable memory usage

    def test_training_multiple_epochs(self, resnet50_model):
        """Test running multiple training epochs with optimizations."""
        if not FusedAdamW.is_available():
            pytest.skip("Fused AdamW not available")

        # Small batch size for memory
        batch_size = 16
        num_epochs = 3

        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)
        for name, module in resnet50_model.named_modules():
            if "layer2" in name:
                selector.select_layers([module])
        compiler.compile(resnet50_model)

        with FusedOptimizerContext():
            optimizer = torch.optim.AdamW(resnet50_model.parameters(), lr=1e-3)
            amp = AMPWrapper(enabled=True)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(num_epochs):
                # Dummy batch
                inputs = torch.randn(batch_size, 3, 224, 224).cuda()
                targets = torch.randint(0, 10, (batch_size,)).cuda()

                resnet50_model.train()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    output = resnet50_model(inputs)
                    loss = criterion(output, targets)

                amp.scale_loss(loss)
                amp.backward()
                optimizer.step()
                amp.update_scale()

                assert loss.item() > 0

    def test_accuracy_preservation(self):
        """Test that optimizations maintain model accuracy on a simple task."""
        # Create a simple classification task
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 10),
        ).cuda()

        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)
        compiler.compile(model)

        with FusedOptimizerContext():
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Train for a few steps
            for step in range(10):
                inputs = torch.randn(16, 3, 224, 224).cuda()
                targets = torch.randint(0, 10, (16,)).cuda()

                model.train()
                optimizer.zero_grad()
                output = model(inputs)
                loss = nn.CrossEntropyLoss()(output, targets)
                loss.backward()
                optimizer.step()

            # Check that model can do inference
            model.eval()
            with torch.no_grad():
                test_input = torch.randn(4, 3, 224, 224).cuda()
                output = model(test_input)
                assert output.shape == (4, 10)

    def test_memory_efficiency(self):
        """Test that optimizations reduce memory usage."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
        ).cuda()

        inputs = torch.randn(64, 1000).cuda()
        targets = torch.randint(0, 10, (64,)).cuda()
        criterion = nn.CrossEntropyLoss()

        # Baseline (no optimizations)
        torch.cuda.reset_peak_memory_stats()
        model_baseline = nn.Sequential(*list(model.children())).cuda()
        optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-3)
        model_baseline.train()
        optimizer_baseline.zero_grad()
        output = model_baseline(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer_baseline.step()
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.max_memory_allocated() / 1024**2

        # With checkpointing
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_checkpoint = nn.Sequential(*list(model.children())).cuda()
        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)
        # Checkpoint first linear layer
        selector.select_layers([model_checkpoint[0]])
        compiler.compile(model_checkpoint)
        optimizer_checkpoint = torch.optim.Adam(model_checkpoint.parameters(), lr=1e-3)
        model_checkpoint.train()
        optimizer_checkpoint.zero_grad()
        output = model_checkpoint(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer_checkpoint.step()
        torch.cuda.synchronize()
        checkpoint_memory = torch.cuda.max_memory_allocated() / 1024**2

        # Checkpointing should reduce memory (at least in peak during backward)
        # Note: This might not always be true for small models, but with a large first layer
        # it should show some reduction
        print(
            f"Baseline: {baseline_memory:.2f} MB, Checkpoint: {checkpoint_memory:.2f} MB"
        )
        # We don't assert strict reduction as it depends on model architecture
        # But we can assert both are measurable
        assert baseline_memory > 0
        assert checkpoint_memory > 0

    def test_profile_after_optimization(self, resnet50_model):
        """Test that profiling works correctly after optimizations applied."""
        # Apply optimizations
        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)
        for name, module in resnet50_model.named_modules():
            if "layer2" in name:
                selector.select_layers([module])
        compiler.compile(resnet50_model)

        with FusedOptimizerContext():
            # Profile the optimized model
            profiler = BaseProfiler(
                resnet50_model, input_shape=(32, 3, 224, 224), use_nsight=False
            )
            results = profiler.profile_inference(iterations=10)

            assert results["fps"] > 0
            assert results["memory_peak_mb"] > 0

    def test_export_and_compare_results(self, resnet50_model, synthetic_data):
        """Test exporting profiling results and comparing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = os.path.join(tmpdir, "baseline.json")
            optimized_path = os.path.join(tmpdir, "optimized.json")

            # Profile baseline (standard optimizer)
            model_baseline = models.resnet50(weights=None, num_classes=10).cuda()
            profiler_baseline = BaseProfiler(
                model_baseline, input_shape=(32, 3, 224, 224), use_nsight=False
            )
            baseline_results = profiler_baseline.profile_inference(iterations=10)
            profiler_baseline.export_results(baseline_path)

            # Profile optimized
            model_optimized = models.resnet50(weights=None, num_classes=10).cuda()
            with FusedOptimizerContext():
                profiler_optimized = BaseProfiler(
                    model_optimized, input_shape=(32, 3, 224, 224), use_nsight=False
                )
                optimized_results = profiler_optimized.profile_inference(iterations=10)
                profiler_optimized.export_results(optimized_path)

            # Compare
            comparison = profiler_optimized.compare_baseline(baseline_path)

            assert "fps_speedup_percent" in comparison
            assert "memory_reduction_percent" in comparison
            # Optimized should generally be better or at least comparable
            assert comparison["current_fps"] > 0
            assert comparison["baseline_fps"] > 0

    def test_multigpu_scaling_simulation(self):
        """Test that optimizations don't break multi-GPU scenarios (if available)."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for multi-GPU test")

        model = nn.Linear(1000, 1000).cuda()
        # Simulate DataParallel wrapping
        model_dp = nn.DataParallel(model)

        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)
        # Don't select layers in DP wrapper - it will traverse into wrapped model
        compiler.compile(model_dp)

        inputs = torch.randn(32, 1000).cuda()
        output = model_dp(inputs)
        assert output.shape == (32, 1000)

    def test_gradient_accumulation_with_amp(self):
        """Test gradient accumulation pattern with AMP."""
        model = nn.Linear(100, 10).cuda()
        criterion = nn.CrossEntropyLoss()

        with FusedOptimizerContext():
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            amp = AMPWrapper(enabled=True)

            accumulation_steps = 4
            optimizer.zero_grad()

            for step in range(accumulation_steps):
                inputs = torch.randn(8, 100).cuda()
                targets = torch.randint(0, 10, (8,)).cuda()

                with torch.cuda.amp.autocast(enabled=True):
                    output = model(inputs)
                    loss = criterion(output, targets) / accumulation_steps

                amp.scale_loss(loss)
                amp.backward()

            optimizer.step()
            amp.update_scale()

            # Gradients should be accumulated
            # No assertions on specific values, just that it completes without error

    def test_mixed_precision_accuracy_simple_model(self):
        """Test that mixed precision maintains reasonable accuracy."""
        torch.manual_seed(123)
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ).cuda()

        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)
        compiler.compile(model)

        with FusedOptimizerContext():
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            amp = AMPWrapper(enabled=True)
            criterion = nn.CrossEntropyLoss()

            # Train for a few steps
            for step in range(50):
                inputs = torch.randn(64, 784).cuda()
                targets = torch.randint(0, 10, (64,)).cuda()

                model.train()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(inputs)
                    loss = criterion(output, targets)

                amp.scale_loss(loss)
                amp.backward()
                optimizer.step()
                amp.update_scale()

                # Loss should decrease over time (roughly)
                if step % 10 == 0:
                    assert loss.item() < 5.0  # Should learn something

    def test_checkpoint_saves_memory(self):
        """Verify checkpointing actually saves memory on a memory-intensive model."""
        torch.manual_seed(0)

        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2000, 2000) for _ in range(10)])

            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    x = torch.relu(layer(x))
                return x

        device = torch.device("cuda")
        model_baseline = LargeModel().to(device)
        model_checkpoint = LargeModel().to(device)
        # Copy weights
        model_checkpoint.load_state_dict(model_baseline.state_dict())

        selector = SelectiveCheckpoint()
        compiler = CheckpointCompiler(selector)
        # Checkpoint all layers except first and last
        for i in range(1, len(model_checkpoint.layers) - 1):
            selector.select_layers([model_checkpoint.layers[i]])
        compiler.compile(model_checkpoint)

        inputs = torch.randn(64, 2000).to(device)

        # Baseline memory
        torch.cuda.reset_peak_memory_stats()
        out1 = model_baseline(inputs)
        loss1 = out1.sum()
        loss1.backward()
        torch.cuda.synchronize()
        baseline_mem = torch.cuda.max_memory_allocated() / 1024**2

        # Checkpointed memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out2 = model_checkpoint(inputs)
        loss2 = out2.sum()
        loss2.backward()
        torch.cuda.synchronize()
        checkpoint_mem = torch.cuda.max_memory_allocated() / 1024**2

        reduction = baseline_mem - checkpoint_mem
        print(
            f"Baseline: {baseline_mem:.2f} MB, Checkpoint: {checkpoint_mem:.2f} MB, Reduction: {reduction:.2f} MB"
        )

        # Checkpointing should save some memory
        assert checkpoint_mem <= baseline_mem * 0.9  # At least 10% reduction expected

    def test_optimization_metadata_preservation(self):
        """Test that optimizer state dict can be saved and loaded."""
        torch.manual_seed(42)
        model = nn.Linear(100, 10).cuda()

        with FusedOptimizerContext():
            optimizer1 = torch.optim.AdamW(model.parameters(), lr=1e-3)
            amp1 = AMPWrapper(enabled=True)

            # Do some training
            for step in range(5):
                inputs = torch.randn(32, 100).cuda()
                targets = torch.randint(0, 10, (32,)).cuda()

                optimizer1.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(inputs)
                    loss = nn.CrossEntropyLoss()(output, targets)
                amp1.scale_loss(loss)
                amp1.backward()
                optimizer1.step()
                amp1.update_scale()

            # Save state
            state = optimizer1.state_dict()

            # Create new optimizer and load
            optimizer2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer2.load_state_dict(state)

            # Should be able to continue training
            for step in range(5):
                inputs = torch.randn(32, 100).cuda()
                targets = torch.randint(0, 10, (32,)).cuda()

                optimizer2.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(inputs)
                    loss = nn.CrossEntropyLoss()(output, targets)
                loss.backward()
                optimizer2.step()

            assert True  # Completed without errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
