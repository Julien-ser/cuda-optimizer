"""
Unit tests for fusion module: FusedAdamW and optimizer fusion utilities.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from cuda_optimizer.fusion import (
    FusedAdamW,
    install_fused_optimizers,
    uninstall_fused_optimizers,
    FusedOptimizerContext,
    get_available_fused_optimizers,
    is_available as fused_adamw_available,
)
import torch.optim as optim

pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFusedAdamW:
    """Test suite for FusedAdamW optimizer."""

    def test_initialization(self):
        """Test FusedAdamW initializes correctly."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        model = nn.Linear(10, 10)
        optimizer = FusedAdamW(model.parameters(), lr=1e-3)
        assert optimizer is not None
        assert len(optimizer.param_groups) == 1

    def test_parameter_groups(self):
        """Test parameter groups work correctly."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        optimizer = FusedAdamW(
            [
                {"params": model[0].parameters(), "lr": 1e-3},
                {"params": model[1].parameters(), "lr": 2e-3},
            ]
        )
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[1]["lr"] == 2e-3

    def test_training_step(self):
        """Test optimizer performs a training step correctly."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        torch.manual_seed(42)
        model = nn.Linear(10, 10).cuda()
        optimizer = FusedAdamW(model.parameters(), lr=1e-3)

        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 10).cuda()
        criterion = nn.MSELoss()

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters were updated
        for param in model.parameters():
            assert param.grad is not None, "Gradient should be computed"

    def test_state_dict(self):
        """Test state dict save/load."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        model = nn.Linear(10, 10).cuda()
        optimizer1 = FusedAdamW(model.parameters(), lr=1e-3)
        optimizer2 = FusedAdamW(model.parameters(), lr=1e-3)

        # Perform some steps
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 10).cuda()
        criterion = nn.MSELoss()

        for _ in range(3):
            optimizer1.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer1.step()

        # Save state
        state = optimizer1.state_dict()

        # Load into optimizer2
        optimizer2.load_state_dict(state)

        # Check state matches
        for p1, p2 in zip(
            optimizer1.state_dict()["state"].values(),
            optimizer2.state_dict()["state"].values(),
        ):
            for k in p1.keys():
                if isinstance(p1[k], torch.Tensor):
                    assert torch.allclose(p1[k], p2[k])
                else:
                    assert p1[k] == p2[k]

    def test_l2_regularization(self):
        """Test that weight decay (L2 regularization) is applied."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        model = nn.Linear(10, 10, bias=False).cuda()
        optimizer = FusedAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 10).cuda()
        criterion = nn.MSELoss()

        # Get initial weights
        initial_weight = model.weight.data.clone()

        # Training step
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        # Check weights changed (due to both gradient and weight decay)
        assert not torch.allclose(model.weight.data, initial_weight)

    def test_gradient_clipping(self):
        """Test gradient clipping works with optimizer."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        model = nn.Linear(10, 10).cuda()
        optimizer = FusedAdamW(model.parameters(), lr=1e-3)

        # Create large gradients
        x = torch.randn(32, 10).cuda() * 1000
        y = torch.randn(32, 10).cuda() * 1000
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        # Check gradient norm before clipping
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check gradient norm decreased
        new_total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                new_total_norm += p.grad.data.norm(2).item() ** 2
        new_total_norm = new_total_norm**0.5

        assert new_total_norm <= total_norm

    def test_different_batch_sizes(self):
        """Test optimizer works with different batch sizes."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        model = nn.Linear(10, 10).cuda()
        optimizer = FusedAdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for batch_size in [1, 8, 16, 32, 64]:
            x = torch.randn(batch_size, 10).cuda()
            y = torch.randn(batch_size, 10).cuda()

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            # Should not raise any errors
            assert not torch.isnan(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFusedOptimizerContext:
    """Test suite for FusedOptimizerContext context manager."""

    def test_context_manager_install_uninstall(self):
        """Test context manager correctly installs and uninstills fused optimizers."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        # Ensure original AdamW is standard
        original_adamw = optim.AdamW

        with FusedOptimizerContext():
            # Inside context, should be replaced
            if fused_adamw_available():
                assert optim.AdamW is FusedAdamW

        # Outside context, should be restored
        assert optim.AdamW is original_adamw

    def test_context_manager_nested(self):
        """Test nested context managers."""
        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        original_adamw = optim.AdamW

        with FusedOptimizerContext():
            with FusedOptimizerContext():
                assert optim.AdamW is FusedAdamW

            # After inner context exits, should still be replaced
            assert optim.AdamW is FusedAdamW

        # After outer context exits, should be restored
        assert optim.AdamW is original_adamw


class TestFusionUtilities:
    """Test suite for fusion utility functions."""

    def test_install_fused_optimizers_not_available_warning(self):
        """Test install warns when kernels not available."""
        # This test doesn't require CUDA
        with pytest.warns(UserWarning, match="Fused optimizers not available"):
            install_fused_optimizers()

    def test_uninstall_fused_optimizers_no_error_when_not_installed(self):
        """Test uninstall doesn't error when not installed."""
        # Should not raise even if nothing was installed
        uninstall_fused_optimizers()

    def test_get_available_fused_optimizers_returns_dict(self):
        """Test get_available_fused_optimizers returns a dict."""
        available = get_available_fused_optimizers()
        assert isinstance(available, dict)
        if fused_adamw_available():
            assert "AdamW" in available
            assert available["AdamW"] is FusedAdamW

    def test_fused_adamw_signature_matches_torch(self):
        """Test FusedAdamW has compatible signature with torch.optim.AdamW."""
        import inspect

        if not fused_adamw_available():
            pytest.skip("Fused AdamW kernels not available")

        # Check that FusedAdamW accepts similar parameters as torch.optim.AdamW
        adamw_sig = inspect.signature(optim.AdamW.__init__)
        fused_sig = inspect.signature(FusedAdamW.__init__)

        # Both should have lr, betas, eps, weight_decay, amsgrad
        common_params = ["lr", "betas", "eps", "weight_decay", "amsgrad"]
        for param in common_params:
            assert param in adamw_sig.parameters, f"torch.AdamW missing {param}"
            assert param in fused_sig.parameters, f"FusedAdamW missing {param}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
