# CUDA Optimizer Documentation

Welcome to the CUDA Optimizer for PyTorch documentation. This toolkit provides specialized optimizations for running PyTorch neural networks on CUDA devices with minimal code changes.

## Quick Links

- **Getting Started**: :doc:`quickstart`
- **API Reference**: :doc:`api/index`
- **Migration Guide**: :doc:`migration_guide`
- **Troubleshooting**: :doc:`troubleshooting`
- **Optimization Targets**: :doc:`optimization_targets`

## Core Features

.. toctree::
   :maxdepth: 2
   
   quickstart
   migration_guide
   troubleshooting
   optimization_targets

## Modules

.. toctree::
   :maxdepth: 2
   
   custom_ops
   cuda_cache
   amp_wrapper
   autotuner
   selective_checkpoint
   tensor_parallel
   adam_fused
   dashboard
   base_profiler

## API Reference

.. toctree::
   :maxdepth: 2
   
   api/index