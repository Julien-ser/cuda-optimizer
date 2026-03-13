"""
Unit tests for custom CUDA kernels.
"""

import pytest
import torch
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from cuda_optimizer.kernels import CustomOps


pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCustomOps:
    """Test suite for custom CUDA kernels."""

    def test_custom_ops_available(self):
        """Test that custom ops are available when CUDA is present."""
        if torch.cuda.is_available():
            # CustomOps should be loadable (may fail if CUDA toolkit not installed)
            # We just check that the class exists and has the right methods
            assert hasattr(CustomOps, "fused_layernorm_gelu")
            assert hasattr(CustomOps, "fused_layernorm_relu")
            assert hasattr(CustomOps, "is_available")

    def test_layernorm_gelu_correctness(self):
        """Test fused LayerNorm + GELU matches native implementation."""
        if not CustomOps.is_available():
            pytest.skip("Custom ops not available - CUDA extension not built")

        torch.manual_seed(42)
        batch_size, seq_len, hidden = 2, 64, 256
        input = torch.randn(
            batch_size, seq_len, hidden, dtype=torch.float32, device="cuda"
        )
        weight = torch.ones(hidden, dtype=torch.float32, device="cuda")
        bias = torch.zeros(hidden, dtype=torch.float32, device="cuda")

        # Custom fused operation
        custom_out = CustomOps.fused_layernorm_gelu(input, weight, bias, eps=1e-5)

        # Native operation
        normalized = torch.nn.functional.layer_norm(
            input, normalized_shape=[hidden], weight=weight, bias=bias, eps=1e-5
        )
        native_out = torch.nn.functional.gelu(normalized)

        # Check close
        assert torch.allclose(custom_out, native_out, rtol=1e-3, atol=1e-3), (
            f"Max diff: {(custom_out - native_out).abs().max().item()}"
        )

    def test_layernorm_relu_correctness(self):
        """Test fused LayerNorm + ReLU matches native implementation."""
        if not CustomOps.is_available():
            pytest.skip("Custom ops not available - CUDA extension not built")

        torch.manual_seed(42)
        batch_size, seq_len, hidden = 2, 64, 256
        input = torch.randn(
            batch_size, seq_len, hidden, dtype=torch.float32, device="cuda"
        )
        weight = torch.ones(hidden, dtype=torch.float32, device="cuda")
        bias = torch.zeros(hidden, dtype=torch.float32, device="cuda")

        custom_out = CustomOps.fused_layernorm_relu(input, weight, bias, eps=1e-5)
        normalized = torch.nn.functional.layer_norm(
            input, normalized_shape=[hidden], weight=weight, bias=bias, eps=1e-5
        )
        native_out = torch.nn.functional.relu(normalized)

        assert torch.allclose(custom_out, native_out, rtol=1e-3, atol=1e-3), (
            f"Max diff: {(custom_out - native_out).abs().max().item()}"
        )

    def test_fused_layernorm_gelu_speedup(self):
        """Test that fused operation is faster than native (at least 0% for now)."""
        if not CustomOps.is_available():
            pytest.skip("Custom ops not available - CUDA extension not built")

        torch.manual_seed(42)
        batch_size, seq_len, hidden = 8, 512, 1024
        input = torch.randn(
            batch_size, seq_len, hidden, dtype=torch.float16, device="cuda"
        )
        weight = torch.ones(hidden, dtype=torch.float16, device="cuda")
        bias = torch.zeros(hidden, dtype=torch.float16, device="cuda")

        # Benchmark custom
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            _ = CustomOps.fused_layernorm_gelu(input, weight, bias)
        end.record()
        torch.cuda.synchronize()
        custom_time = start.elapsed_time(end) / 10

        # Benchmark native
        torch.cuda.synchronize()
        start.record()
        for _ in range(10):
            normalized = torch.nn.functional.layer_norm(
                input, normalized_shape=[hidden], weight=weight, bias=bias, eps=1e-5
            )
            _ = torch.nn.functional.gelu(normalized)
        end.record()
        torch.cuda.synchronize()
        native_time = start.elapsed_time(end) / 10

        speedup = native_time / custom_time
        print(
            f"Custom: {custom_time:.4f}ms, Native: {native_time:.4f}ms, Speedup: {speedup:.2f}x"
        )

        # We expect some speedup due to kernel fusion (at least 1.0x, ideally 1.2x)
        # The 20% target is in benchmark script; unit test just checks it doesn't regress badly
        assert speedup > 0.9, f"Custom ops are slower than native: {speedup:.2f}x"

    def test_gradient_flow(self):
        """Test that gradients flow correctly through fused operation."""
        if not CustomOps.is_available():
            pytest.skip("Custom ops not available - CUDA extension not built")

        torch.manual_seed(42)
        batch_size, seq_len, hidden = 2, 32, 128
        input = torch.randn(
            batch_size,
            seq_len,
            hidden,
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        weight = torch.ones(
            hidden, dtype=torch.float32, device="cuda", requires_grad=True
        )
        bias = torch.zeros(
            hidden, dtype=torch.float32, device="cuda", requires_grad=True
        )

        output = CustomOps.fused_layernorm_gelu(input, weight, bias)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are finite
        assert input.grad is not None
        assert weight.grad is not None
        assert bias.grad is not None
        assert torch.all(torch.isfinite(input.grad))
        assert torch.all(torch.isfinite(weight.grad))
        assert torch.all(torch.isfinite(bias.grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
