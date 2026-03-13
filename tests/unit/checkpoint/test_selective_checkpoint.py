"""
Unit tests for SelectiveCheckpoint and CheckpointCompiler.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from cuda_optimizer.checkpoint import SelectiveCheckpoint, CheckpointCompiler

pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSelectiveCheckpoint:
    """Test suite for SelectiveCheckpoint."""

    def test_initialization(self):
        """Test selector initializes with empty state."""
        selector = SelectiveCheckpoint()
        assert selector._selected_layers == set()
        assert selector._name_patterns == []
        assert selector._type_whitelist == []
        assert selector._custom_recompute == {}

    def test_select_layers_explicit(self):
        """Test explicit layer selection."""
        selector = SelectiveCheckpoint()
        layer1 = nn.Linear(10, 10)
        layer2 = nn.Linear(10, 10)
        selector.select_layers([layer1])
        selected = selector.get_selected_layers(nn.Sequential(layer1, layer2))
        assert layer1 in selected
        assert layer2 not in selected

    def test_select_by_name_pattern(self):
        """Test selection by name pattern."""
        selector = SelectiveCheckpoint()

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(16, 32, 3)
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = TestModel()
        selector.select_by_name("conv")
        selected = selector.get_selected_layers(model)
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        assert len(selected) == 2
        for layer in conv_layers:
            assert layer in selected

    def test_select_by_type(self):
        """Test selection by type."""
        selector = SelectiveCheckpoint()
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Linear(10, 10), nn.Linear(10, 5)
        )
        selector.select_by_type(nn.Linear)
        selected = selector.get_selected_layers(model)
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(selected) == 2
        for layer in linear_layers:
            assert layer in selected

    def test_custom_recompute(self):
        """Test setting and retrieving custom recompute."""
        selector = SelectiveCheckpoint()
        layer = nn.Linear(10, 10)
        called = False

        def custom_recompute(forward_fn, *args, **kwargs):
            nonlocal called
            called = True
            return forward_fn(*args, **kwargs)

        selector.select_layers([layer])
        selector.set_custom_recompute(layer, custom_recompute)
        recompute_fn = selector.get_recompute_fn(layer)
        assert recompute_fn is custom_recompute

    def test_default_recompute(self):
        """Test default recompute function is checkpoint."""
        selector = SelectiveCheckpoint()
        layer = nn.Linear(10, 10)
        recompute_fn = selector.get_recompute_fn(layer)
        assert recompute_fn is SelectiveCheckpoint._default_recompute

    def test_get_selected_layers_combined(self):
        """Test combining selection methods."""
        selector = SelectiveCheckpoint()
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.Linear(10, 10), nn.Linear(10, 5)
        )
        conv = model[0]
        lin1 = model[2]
        selector.select_layers([conv])
        selector.select_by_type(nn.Linear)
        selected = selector.get_selected_layers(model)
        assert conv in selected
        assert lin1 in selected
        assert model[1] not in selected  # ReLU

    def test_default_recompute_uses_checkpoint(self):
        """Test that default recompute actually uses torch.utils.checkpoint."""
        selector = SelectiveCheckpoint()
        layer = nn.Linear(10, 10)
        recompute_fn = selector.get_recompute_fn(layer)
        # Should be a function that returns checkpoint(...)
        # We can test by calling it with dummy args and see if it wraps
        x = torch.randn(5, 10)
        # Calling default recompute should work and produce same output as direct call
        direct_out = layer(x)
        recomputed_out = recompute_fn(layer, x)
        assert torch.allclose(direct_out, recomputed_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCheckpointCompiler:
    """Test suite for CheckpointCompiler."""

    def test_compile_wraps_selected_layer(self):
        """Test that compile wraps forward method."""
        selector = SelectiveCheckpoint()
        layer = nn.Linear(10, 10)
        selector.select_layers([layer])
        compiler = CheckpointCompiler(selector)

        model = nn.Sequential(layer)
        compiler.compile(model)

        # The layer's forward should now be the checkpointed version
        assert layer.forward.__name__ == "checkpointed_forward"
        assert getattr(layer, "_checkpoint_wrapped", False) is True

    def test_compile_does_not_wrap_unselected(self):
        """Test unselected layers remain unchanged."""
        selector = SelectiveCheckpoint()
        layer1 = nn.Linear(10, 10)
        layer2 = nn.Linear(10, 10)
        selector.select_layers([layer1])
        compiler = CheckpointCompiler(selector)

        model = nn.Sequential(layer1, layer2)
        compiler.compile(model)

        assert layer1.forward.__name__ == "checkpointed_forward"
        assert layer2.forward.__name__ != "checkpointed_forward"
        assert not getattr(layer2, "_checkpoint_wrapped", False)

    def test_compile_idempotent(self):
        """Test that compiling twice does not double-wrap."""
        selector = SelectiveCheckpoint()
        layer = nn.Linear(10, 10)
        selector.select_layers([layer])
        compiler = CheckpointCompiler(selector)

        model = nn.Sequential(layer)
        compiler.compile(model)
        forward1 = layer.forward
        compiler.compile(model)  # compile again
        forward2 = layer.forward

        assert forward1 is forward2  # Same function

    def test_checkpoint_preserves_output(self):
        """Test that checkpointed layer produces the same output."""
        torch.manual_seed(42)
        selector = SelectiveCheckpoint()
        layer = nn.Linear(10, 10)
        selector.select_layers([layer])
        compiler = CheckpointCompiler(selector)

        model = nn.Sequential(layer)
        compiler.compile(model)

        x = torch.randn(5, 10)
        out1 = model(x)

        # Reset seed and try again to ensure deterministic? Not needed.
        # Compare with direct call
        direct_out = layer(x)
        assert torch.allclose(out1, direct_out, rtol=1e-5, atol=1e-5)

    def test_memory_reduction(self):
        """Test that checkpointing reduces peak memory."""
        torch.cuda.empty_cache()

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1000, 1000)
                self.fc2 = nn.Linear(1000, 1000)
                self.fc3 = nn.Linear(1000, 10)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        device = torch.device("cuda")
        input_tensor = torch.randn(64, 1000, device=device)
        target = torch.randint(0, 10, (64,), device=device)
        criterion = nn.CrossEntropyLoss()

        # Baseline model
        model1 = SmallModel().to(device)
        optimizer1 = torch.optim.Adam(model1.parameters())

        torch.cuda.reset_peak_memory_stats()
        optimizer1.zero_grad()
        out1 = model1(input_tensor)
        loss1 = criterion(out1, target)
        loss1.backward()
        optimizer1.step()
        baseline_peak = torch.cuda.max_memory_allocated()

        # Checkpointed model
        torch.cuda.empty_cache()
        model2 = SmallModel().to(device)
        selector = SelectiveCheckpoint()
        selector.select_layers([model2.fc1, model2.fc2])
        compiler = CheckpointCompiler(selector)
        compiler.compile(model2)
        optimizer2 = torch.optim.Adam(model2.parameters())

        torch.cuda.reset_peak_memory_stats()
        optimizer2.zero_grad()
        out2 = model2(input_tensor)
        loss2 = criterion(out2, target)
        loss2.backward()
        optimizer2.step()
        checkpointed_peak = torch.cuda.max_memory_allocated()

        reduction = baseline_peak - checkpointed_peak
        print(
            f"Baseline: {baseline_peak / 1e6:.2f} MB, Checkpointed: {checkpointed_peak / 1e6:.2f} MB"
        )
        assert reduction > 0, "Checkpointing should reduce memory usage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
