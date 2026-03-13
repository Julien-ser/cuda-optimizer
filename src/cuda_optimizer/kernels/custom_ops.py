"""
Custom CUDA operations (fused kernels, etc.).
"""

import torch
from torch.utils.cpp_extension import load
import os

# Get the directory of this file
_kernel_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_src = os.path.join(_kernel_dir, "custom_ops.cu")

# Load the CUDA extension
try:
    _custom_ops = load(
        name="custom_ops_cuda",
        sources=[_cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode",
            "arch=compute_75,code=sm_75",  # Turing (RTX 20xx, A100)
            "-gencode",
            "arch=compute_80,code=sm_80",  # Ampere (A100, A40)
            "-gencode",
            "arch=compute_86,code=sm_86",  # Ampere (RTX 30xx)
            "-gencode",
            "arch=compute_89,code=sm_89",  # Ada Lovelace (RTX 40xx)
            "-gencode",
            "arch=compute_90,code=sm_90",  # Hopper (H100)
        ],
        verbose=True,
    )
except Exception as e:
    print(f"Warning: Failed to load CUDA extension: {e}")
    print("CustomOps will not be available. Please ensure CUDA toolkit is installed.")
    _custom_ops = None


class CustomOps:
    """Custom CUDA operations for fused activation+layernorm, etc."""

    @staticmethod
    def fused_layernorm_gelu(input, weight, bias, eps=1e-5):
        """
        Fused LayerNorm + GELU activation.

        Args:
            input: Tensor of shape [..., features] on CUDA
            weight: 1D tensor of size [features]
            bias: 1D tensor of size [features]
            eps: epsilon for LayerNorm

        Returns:
            Output tensor after LayerNorm + GELU
        """
        if _custom_ops is None:
            raise RuntimeError(
                "CUDA custom operations not loaded. Build extension first."
            )
        return _custom_ops.fused_layernorm_gelu(input, weight, bias, eps)[0]

    @staticmethod
    def fused_layernorm_relu(input, weight, bias, eps=1e-5):
        """
        Fused LayerNorm + ReLU activation.

        Args:
            input: Tensor of shape [..., features] on CUDA
            weight: 1D tensor of size [features]
            bias: 1D tensor of size [features]
            eps: epsilon for LayerNorm

        Returns:
            Output tensor after LayerNorm + ReLU
        """
        if _custom_ops is None:
            raise RuntimeError(
                "CUDA custom operations not loaded. Build extension first."
            )
        return _custom_ops.fused_layernorm_relu(input, weight, bias, eps)[0]

    @staticmethod
    def is_available():
        """Check if custom ops are available."""
        return _custom_ops is not None
