# cuda_optimizer.profiling API Reference

.. module:: cuda_optimizer.profiling

Profiling infrastructure using torch.profiler and NVIDIA Nsight.

## Classes

.. autoclass:: cuda_optimizer.profiling.BaseProfiler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: cuda_optimizer.profiling.NsightProfiler
   :members:
   :undoc-members:
   :show-inheritance:

## Functions

.. autofunction:: cuda_optimizer.profiling.profile_model
.. autofunction:: cuda_optimizer.profiling.analyze_bottlenecks
.. autofunction:: cuda_optimizer.profiling.generate_report
