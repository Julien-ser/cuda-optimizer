"""
CUDA Optimizer for PyTorch

A specialized toolkit for optimizing PyTorch neural networks on CUDA devices.
"""

__version__ = "0.1.0"
__author__ = "CUDA Optimizer Team"

# Import main components (to be implemented in later phases)
from cuda_optimizer.profiling.base_profiler import BaseProfiler
from cuda_optimizer.kernels.custom_ops import CustomOps
from cuda_optimizer.memory.cuda_cache import CUDACache
from cuda_optimizer.optim.amp_wrapper import AMPWrapper
from cuda_optimizer.tuner.autotuner import Autotuner
from cuda_optimizer.checkpoint.selective_checkpoint import SelectiveCheckpoint
from cuda_optimizer.parallel.tensor_parallel import TensorParallel
from cuda_optimizer.fusion.adam_fused import AdamFused
from cuda_optimizer.monitoring.dashboard import Dashboard

__all__ = [
    "BaseProfiler",
    "CustomOps",
    "CUDACache",
    "AMPWrapper",
    "Autotuner",
    "SelectiveCheckpoint",
    "TensorParallel",
    "AdamFused",
    "Dashboard",
]
