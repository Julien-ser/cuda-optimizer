"""
Unit tests for AMPWrapper and LayerAwareLossScaler.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cuda_optimizer.optim import AMPWrapper, LayerAwareLossScaler


pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLayerAwareLossScaler:
    """Test suite for LayerAwareLossScaler."""

    def test_initialization(self):
        """Test scaler initializes correctly."""
        scaler = LayerAwareLossScaler()
        assert scaler.get_scale() > 0
        assert scaler._global_scale == 2.0**16

    def test_individual_scaling(self):
        """Test per-layer scaling works."""
        scaler = LayerAwareLossScaler(layer_names=["layer1", "layer2"])

        scale1 = scaler.get_scale("layer1.weight")
        scale2 = scaler.get_scale("layer2.weight")

        assert scale1 == scale2  # Initially same
        assert scaler.get_scale("layer1.bias") == scaler.get_scale("layer1.weight")

    def test_gradient_norm_tracking(self):
        """Test gradient norm statistics are tracked."""
        scaler = LayerAwareLossScaler(layer_names=["layer1"])

        # Update gradient norms for different layers
        scaler.update_gradient_norm("layer1.weight", 0.5)
        scaler.update_gradient_norm("layer1.weight", 0.7)
        scaler.update_gradient_norm("layer2.weight", 1.0)

        # Check statistics
        stats = scaler._grad_norm_sums
        assert stats["layer1"] > 0
        assert stats["other"] > 0

    def test_overflow_tracking(self):
        """Test overflow counting."""
        scaler = LayerAwareLossScaler(layer_names=["layer1"])

        scaler.check_overflow("layer1.weight", True)
        scaler.check_overflow("layer1.weight", False)
        scaler.check_overflow("layer2.weight", True)

        assert scaler._overflow_counts["layer1"] == 1
        assert scaler._overflow_counts["layer2"] == 1

    def test_state_dict_serialization(self):
        """Test state dict save/load."""
        scaler = LayerAwareLossScaler(layer_names=["layer1"])
        scaler.update_gradient_norm("layer1.weight", 0.5)
        scaler._step = 100

        state = scaler.state_dict()

        # Create new scaler and load state
        scaler2 = LayerAwareLossScaler()
        scaler2.load_state_dict(state)

        assert scaler2._step == 100
        assert scaler2._global_scale == scaler._global_scale
        assert scaler2._grad_norm_sums["layer1"] == 0.5

    def test_scale_update_growth(self):
        """Test scale grows when gradients are small and stable."""
        scaler = LayerAwareLossScaler(
            initial_scale=1024.0,
            growth_factor=2.0,
            growth_interval=10,
            layer_names=["layer1"],
        )

        initial = scaler.get_scale("layer1.weight")

        # Simulate many steps with small gradients and no overflow
        for _ in range(15):
            scaler.update_gradient_norm("layer1.weight", 0.1)
            scaler.check_overflow("layer1.weight", False)
            scaler.step()

        # Scale should have grown (after 10 steps)
        new_scale = scaler.get_scale("layer1.weight")
        assert new_scale > initial

    def test_scale_update_backoff(self):
        """Test scale reduces on overflow."""
        scaler = LayerAwareLossScaler(
            initial_scale=1024.0,
            backoff_factor=0.5,
            growth_interval=1,
            layer_names=["layer1"],
        )

        # Simulate overflow
        scaler.check_overflow("layer1.weight", True)
        scaler.step()

        new_scale = scaler.get_scale("layer1.weight")
        assert new_scale < 1024.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAMPWrapper:
    """Test suite for AMPWrapper."""

    def test_initialization(self):
        """Test AMPWrapper initializes correctly."""
        model = nn.Linear(100, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        wrapper = AMPWrapper(
            model=model, optimizer=optimizer, accumulation_steps=1, enabled=True
        )

        assert wrapper.enabled is True
        assert wrapper.accumulation_steps == 1
        assert wrapper.model is model
        assert wrapper.optimizer is optimizer

    def test_fp32_fallback(self):
        """Test FP32 training works when AMP is disabled."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        wrapper = AMPWrapper(model=model, optimizer=optimizer, enabled=False)

        # Create dummy batch
        x = torch.randn(8, 10).cuda()
        y = torch.randn(8, 5).cuda()

        # Single training step
        loss_fn = nn.MSELoss()
        metrics = wrapper.train_step((x, y), loss_fn)

        assert "loss" in metrics
        assert metrics["scale"] == 1.0
        assert wrapper.metrics["total_steps"] == 1

    def test_amp_training_step(self):
        """Test AMP training step with minimal model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        wrapper = AMPWrapper(
            model=model, optimizer=optimizer, accumulation_steps=1, enabled=True
        )

        # Create dummy batch
        x = torch.randn(16, 10).cuda()
        y = torch.randn(16, 5).cuda()
        loss_fn = nn.MSELoss()

        # Training step
        metrics = wrapper.train_step((x, y), loss_fn)

        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert "scale" in metrics
        assert metrics["scale"] > 0
        assert wrapper.metrics["total_steps"] == 1

    def test_gradient_accumulation(self):
        """Test gradient accumulation works correctly."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        wrapper = AMPWrapper(
            model=model, optimizer=optimizer, accumulation_steps=4, enabled=True
        )

        x = torch.randn(8, 10).cuda()
        y = torch.randn(8, 5).cuda()
        loss_fn = nn.MSELoss()

        # Run 4 accumulation steps without optimizer step
        for i in range(4):
            metrics = wrapper.train_step((x, y), loss_fn, apply_optimizer_step=False)
            assert metrics["step_taken"] is False
            assert metrics["accumulation_step"] == i + 1

        # Next step should trigger optimizer update
        metrics = wrapper.train_step((x, y), loss_fn)
        assert metrics["step_taken"] is True
        assert metrics["accumulation_step"] == 1  # Reset after step

    def test_scale_loss_method(self):
        """Test scale_loss method."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        wrapper = AMPWrapper(model=model, optimizer=optimizer, enabled=True)

        loss = torch.tensor(1.5).cuda()
        scaled_loss = wrapper.scale_loss(loss)

        expected_scale = wrapper.scaler.get_scale()
        assert scaled_loss.item() == loss.item() * expected_scale

        # Test with disabled AMP
        wrapper.enabled = False
        scaled_loss = wrapper.scale_loss(loss)
        assert scaled_loss.item() == loss.item()

    def test_state_save_load(self):
        """Test state can be saved and loaded."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        wrapper = AMPWrapper(model=model, optimizer=optimizer)

        # Run a few steps
        x = torch.randn(8, 10).cuda()
        y = torch.randn(8, 5).cuda()
        loss_fn = nn.MSELoss()

        for _ in range(3):
            wrapper.train_step((x, y), loss_fn)

        # Save state
        state = wrapper.state_dict()

        # Create new wrapper with same model/optimizer
        wrapper2 = AMPWrapper(model=model, optimizer=optimizer)
        wrapper2.load_state_dict(state)

        # Check state restored
        assert wrapper2.metrics["total_steps"] == wrapper.metrics["total_steps"]
        assert wrapper2.scaler._step == wrapper.scaler._step

    def test_metrics_tracking(self):
        """Test performance metrics are tracked."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        wrapper = AMPWrapper(model=model, optimizer=optimizer, enabled=True)

        x = torch.randn(8, 10).cuda()
        y = torch.randn(8, 5).cuda()
        loss_fn = nn.MSELoss()

        for _ in range(5):
            wrapper.train_step((x, y), loss_fn)

        metrics = wrapper.get_metrics()
        assert metrics["total_steps"] == 5
        assert "overflow_rate" in metrics
        assert metrics["overflow_rate"] >= 0
        assert metrics["overflow_rate"] <= 1

    def test_get_scaling_stats(self):
        """Test scaling statistics retrieval."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        wrapper = AMPWrapper(
            model=model, optimizer=optimizer, layers_for_individual_scaling=["layer"]
        )

        stats = wrapper.get_scaling_stats()
        assert "global_scale" in stats
        assert "per_layer_scales" in stats
        assert isinstance(stats["per_layer_scales"], dict)

    def test_zero_grad(self):
        """Test zero_grad passes through to optimizer."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        wrapper = AMPWrapper(model=model, optimizer=optimizer)

        # Create some gradients
        x = torch.randn(8, 10).cuda()
        y = torch.randn(8, 5).cuda()
        loss_fn = nn.MSELoss()
        loss = loss_fn(model(x), y)
        loss.backward()

        # Check gradients exist
        assert any(p.grad is not None for p in model.parameters())

        # Zero gradients
        wrapper.zero_grad()

        # All gradients should be zero
        assert all(p.grad is None or p.grad.norm() == 0 for p in model.parameters())

    def test_individual_layer_scaling(self):
        """Test individual layer scaling feature."""
        model = nn.Sequential(
            nn.Linear(10, 20), nn.Linear(20, 30), nn.Linear(30, 5)
        ).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        wrapper = AMPWrapper(
            model=model,
            optimizer=optimizer,
            layers_for_individual_scaling=["0", "2"],  # First and last linear layers
        )

        # Check different layers have potentially different scales
        scale0 = wrapper.scaler.get_scale("0.weight")
        scale1 = wrapper.scaler.get_scale("1.weight")  # Should fall back to "other"
        scale2 = wrapper.scaler.get_scale("2.weight")

        assert scale0 > 0
        assert scale1 > 0
        assert scale2 > 0

    def test_validation_accuracy(self):
        """Test accuracy validation compares AMP vs FP32."""
        model = nn.Linear(10, 5).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        wrapper = AMPWrapper(model=model, optimizer=optimizer, enabled=True)

        # Create simple validation loader
        val_data = [
            (torch.randn(16, 10).cuda(), torch.randint(0, 5, (16,)).cuda())
            for _ in range(5)
        ]

        # This should not raise and should return two accuracy values
        amp_acc, fp32_acc = wrapper.validate_accuracy(val_data, max_batches=3)

        assert 0 <= amp_acc <= 1
        assert 0 <= fp32_acc <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAMPWrapperIntegration:
    """Integration tests for AMPWrapper with realistic models."""

    def test_simple_cnn_training(self):
        """Test AMP wrapper with a simple CNN."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10),
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        wrapper = AMPWrapper(model=model, optimizer=optimizer, enabled=True)

        # Dummy training for a few steps
        for step in range(5):
            x = torch.randn(16, 3, 32, 32).cuda()
            y = torch.randint(0, 10, (16,)).cuda()
            loss_fn = nn.CrossEntropyLoss()

            metrics = wrapper.train_step((x, y), loss_fn)
            assert "loss" in metrics
            assert metrics["loss"] > 0

            if step == 4:
                # Check metrics accumulated
                all_metrics = wrapper.get_metrics()
                assert all_metrics["total_steps"] == 5

    def test_accumulation_steps_integration(self):
        """Test gradient accumulation with realistic batch sizes."""
        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10)).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        wrapper = AMPWrapper(
            model=model, optimizer=optimizer, accumulation_steps=2, enabled=True
        )

        loss_fn = nn.CrossEntropyLoss()

        # Simulate small batch size via accumulation
        x = torch.randn(8, 100).cuda()
        y = torch.randint(0, 10, (8,)).cuda()

        # First step - accumulate
        m1 = wrapper.train_step((x, y), loss_fn, apply_optimizer_step=False)
        assert m1["step_taken"] is False

        # Second step - should trigger
        m2 = wrapper.train_step((x, y), loss_fn, apply_optimizer_step=False)
        assert m2["step_taken"] is False

        # Third step - should trigger optimizer
        m3 = wrapper.train_step((x, y), loss_fn)
        assert m3["step_taken"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
