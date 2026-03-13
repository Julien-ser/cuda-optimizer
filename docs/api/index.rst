# API Reference

API documentation for the CUDA Optimizer package.

## Modules

.. toctree::
   :maxdepth: 2

   cuda_optimizer
   cuda_optimizer.kernels
   cuda_optimizer.memory
   cuda_optimizer.optim
   cuda_optimizer.tuner
   cuda_optimizer.checkpoint
   cuda_optimizer.parallel
   cuda_optimizer.fusion
   cuda_optimizer.monitoring
   cuda_optimizer.profiling

## Key Classes

- :class:`cuda_optimizer.BaseProfiler` - Profiling infrastructure
- :class:`cuda_optimizer.CustomOps` - Custom CUDA kernels
- :class:`cuda_optimizer.CUDACache` - Memory caching allocator
- :class:`cuda_optimizer.AMPWrapper` - Mixed precision training
- :class:`cuda_optimizer.Autotuner` - Kernel auto-tuning
- :class:`cuda_optimizer.SelectiveCheckpoint` - Gradient checkpointing
- :class:`cuda_optimizer.TensorParallel` - Multi-GPU parallelism
- :class:`cuda_optimizer.FusedAdamW` - Fused optimizer
- :class:`cuda_optimizer.Dashboard` - Monitoring dashboard