# Troubleshooting Guide

Common issues and solutions when using CUDA Optimizer.

## Installation Issues

### CUDA Version Mismatch

**Error:** `RuntimeError: CUDA driver version is insufficient for CUDA runtime version`

**Solution:** Ensure your CUDA toolkit matches your driver:
```bash
# Check driver version
nvidia-smi

# Check CUDA runtime version
nvcc --version

# CUDA 11.8+ required. Update driver if needed.
```

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install PyTorch with CUDA support first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

### Build Failures (Custom CUDA Kernels)

**Error:** `error: command 'nvcc' failed with exit status 1`

**Solution:** Ensure CUDA binaries are in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Error:** `torch.ops.load_library() failed: libc10_cuda.so: cannot open shared object file`

**Solution:** PyTorch CUDA version mismatch. Reinstall PyTorch with matching CUDA version.

---

## GPU & CUDA Detection

### CUDA Not Available

**Error:** `RuntimeError: CUDA is not available`

**Solutions:**
1. Check GPU accessibility:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be > 0
```

2. Verify NVIDIA driver is loaded:
```bash
nvidia-smi
```

3. Check CUDA installation:
```bash
python -c "import torch; print(torch.version.cuda)"
```

### Insufficient GPU Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Enable memory caching (should be automatic):
```python
from cuda_optimizer import CUDACache
model = CUDACache(pool_size_gb=0.8 * torch.cuda.get_device_properties(0).total_memory/1e9).optimize(model)
```

2. Reduce batch size:
```python
batch_size = max(1, original_batch // 2)
```

3. Enable gradient checkpointing:
```python
from cuda_optimizer import SelectiveCheckpoint
checkpoint = SelectiveCheckpoint(compute_ratio=0.3)
model = checkpoint.apply(model)
```

4. Use mixed precision:
```python
from cuda_optimizer import AMPWrapper
amp = AMPWrapper()
model, optimizer = amp.prepare(model, optimizer)
```

---

## Performance Issues

### No Speedup Observed

**Problem:** Model runs at same speed after optimization

**Diagnosis:**
```python
from cuda_optimizer import BaseProfiler

# Profile before and after
profile_model(model, input_shape=(32, 3, 224, 224), num_iters=100)
optimized_model = Optimizer.optimize(model)
profile_model(optimized_model, input_shape=(32, 3, 224, 224), num_iters=100)
```

**Common Causes & Solutions:**

1. **Kernel auto-tuner not run**: Some custom kernels need one-time tuning:
```python
from cuda_optimizer import Autotuner
autotuner = Autotuner(cache_dir='~/.cache/cuda-optimizer/')
autotuner.tune_operations(model, input_shape=(32, 3, 224, 224))
```

2. **Small batch size**: Optimizations have overhead. Use batch_size >= 16 for CNNs, >= 8 for transformers.

3. **Model too small**: For tiny models (<1M parameters), overhead may outweigh benefits. Consider batching multiple samples.

4. **CPU bottleneck**: Data loading may be limiting factor:
```python
# Increase num_workers, use pin_memory
dataloader = DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True)
```

5. **Incorrect optimization applied**: Check which optimizations are active:
```python
from cuda_optimizer import get_active_optimizations
print(get_active_optimizations())
```

### Slow Training After Checkpointing

**Problem:** Checkpointing causes slowdown despite memory savings

**Solution:** Checkpointing trades compute for memory. Ensure:
- GPU utilization remains high (>80%)
- Recommended for memory-bound scenarios, not compute-bound
- Tune `compute_ratio` parameter (0.2-0.4 typically optimal)

---

## Accuracy & Correctness

### Outputs Don't Match Baseline

**Problem:** Model outputs differ from vanilla PyTorch

**Diagnosis:**
```python
# Set torch to deterministic mode for fair comparison
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)
```

**Common Causes:**

1. **AMP precision loss**: Mixed precision introduces small numerical differences. Check if within tolerance:
```python
diff = (output_original - output_optimized).abs().max()
print(f"Max diff: {diff}")  # Should be < 1e-3 typically
```

If diff > 0.01, disable AMP temporarily:
```python
amp = AMPWrapper(enabled=False)
```

2. **Checkpointing recomputation order**: Some layers may compute differently due to recompute. Check layer-specific checkpointing:
```python
# Exclude sensitive layers
model = checkpoint.apply(model, layers=['layer1', 'layer2'], exclude=['norm'])
```

3. **Fused kernel precision**: Custom fused ops may use different reduction order. Compare element-wise:
```python
# Manual verification
from cuda_optimizer.kernels import CustomOps
original_layer = nn.LayerNorm(...)
fused_layer = CustomOps.FusedLayerNormActivation(...)

# Test with same input
x = torch.randn(..., device='cuda')
out1 = original_layer(x)
out2 = fused_layer(x)
print(torch.allclose(out1, out2, rtol=1e-3, atol=1e-3))
```

**Validation:** See [Migration Guide](migration_guide.md) checklist for acceptable tolerances.

---

## Dashboard & Monitoring Issues

### Streamlit Dashboard Won't Start

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:** Install optional dependencies:
```bash
pip install streamlit pandas plotly
```

**Error:** Dashboard shows "No data" or empty plots

**Solution:** Ensure dashboard is imported and started:
```python
from cuda_optimizer import Dashboard

dashboard = Dashboard()
dashboard.start()  # Starts in background thread

# Must keep Python process alive
import time
time.sleep(3600)  # Or use dashboard.wait()
```

**Performance impact:** Dashboard adds <1% overhead. To disable logging:
```python
import logging
logging.getLogger('cuda_optimizer.monitoring').setLevel(logging.WARNING)
```

---

## Multi-GPU & Tensor Parallelism

### NCCL Communication Errors

**Error:** `RuntimeError: NCCL error in: ...`

**Solutions:**
1. Ensure all GPUs are same model and compute capability:
```bash
nvidia-smi  # Check all GPUs visible
```

2. Set NCCL environment variables:
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # or your network interface
export NCCL_IB_DISABLE=1  # disable InfiniBand if issues
```

3. Check for GPU memory fragmentation. Use `CUDACache`:
```python
cache = CUDACache()
model = cache.optimize(model)
```

4. Verify GPU connectivity (for multi-node):
```bash
python -c "import torch.distributed as dist; dist.init_process_group(backend='nccl')"
```

### Poor Scaling Across GPUs

**Problem:** Speedup < 1.8x on 2 GPUs

**Diagnosis:**
1. Profile communication overhead:
```bash
nvprof --profile-from-start off -o profile.nvvp python train.py
# Open in Nsight Systems
```

2. Check batch size per GPU:
- Should be >= 16 for good utilization
- Total batch = batch_per_gpu * num_gpus

3. Verify tensor parallelism strategy:
```python
# For small models, 1D is fastest
parallel = TensorParallel(strategy='1d')

# For very large models (GPT, large transformers), try 2D
parallel = TensorParallel(strategy='2d')
```

---

## Specific Component Issues

### Autotuner Hangs or Takes Too Long

**Problem:** Tuning takes >1 hour per operation

**Solution:**
1. Reduce search space:
```python
autotuner = Autotuner(
    max_trials=50,          # default: 100
    timeout_seconds=300    # per operation
)
```

2. Use cached configurations from similar GPUs:
```python
autotuner = Autotuner(
    cache_dir='~/.cache/cuda-optimizer/',
    load_cached=True
)
```

3. Skip tuning for already-tuned ops:
```python
autotuner.tune_operations(model, input_shape, skip_tuned=True)
```

### Custom Ops Fail to Load

**Error:** `RuntimeError: Found 'cuda' but can't load custom op library`

**Solutions:**
1. Rebuild custom ops:
```bash
cd src/cuda_optimizer/kernels
python setup.py install
```

2. Check CUDA architecture compatibility:
```bash
# In custom_ops.py, ensure correct compute capability
# For A100: compute capability 8.0
# For V100: 7.0
# For RTX 3090: 8.6
```

3. Verify CUDA toolkit version matches PyTorch build:
```python
print(torch.version.cuda)  # e.g., '11.8'
print(nvcc --version)      # Should match major.minor
```

### FusedAdamW State Dict Issues

**Problem:** `RuntimeError: Error(s) in loading state_dict for FusedAdamW`

**Solution:** FusedAdamW has different state layout. When loading checkpoints:
```python
# Option 1: Wrap optimizer
optimizer = FusedAdamW(...)
optimizer.load_state_dict(torch.load('checkpoint.pth')['optimizer'])

# Option 2: Convert from standard AdamW
std_opt = torch.optim.AdamW(...)
state = torch.load('checkpoint.pth')
std_opt.load_state_dict(state['optimizer'])
optimizer = FusedAdamW.from_standard(std_opt)  # converts state
```

---

## Data Type & Device Issues

### Mixed Dtype Errors

**Error:** `RuntimeError: expected scalar type Half but found Float`

**Solution:** AMP wrapper handles this automatically. If manual issues:
```python
# Ensure model and data are on same device/dtype
model = model.cuda().half()  # or .bfloat16() for Ampere+
data = data.cuda().half()

# Or use autocast
with torch.cuda.amp.autocast():
    output = model(data)
```

### CPU ↔ GPU Transfer Slows Training

**Problem:** `DataLoader` transfers are bottleneck

**Solutions:**
1. Use `pin_memory=True`:
```python
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=4)
```

2. Prefetch to GPU:
```python
from cuda_optimizer import CUDACache
cache = CUDACache(prefetch=True)  # async memory transfers
```

3. Use `non_blocking=True` for transfers:
```python
data = data.cuda(non_blocking=True)
labels = labels.cuda(non_blocking=True)
```

---

## Debugging & Diagnostics

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Or for specific module:
logging.getLogger('cuda_optimizer.kernels').setLevel(logging.DEBUG)
```

### Inspect Active Optimizations

```python
from cuda_optimizer import get_optimization_stats
stats = get_optimization_stats()
print(stats)
# Output shows which optimizations are active, memory savings, etc.
```

### Profile with Nsight Systems

```bash
# Install Nsight Systems (from CUDA toolkit)
nsys profile -o profile_report python train.py

# View GUI
nsys-ui profile_report.qdrep
```

### Check CUDA Errors

```python
torch.cuda.set_device(0)
torch.cuda.synchronize()

# After CUDA operations
if torch.cuda.is_available():
    print(torch.cuda.memory_summary())
```

---

## Environment-Specific Issues

### Docker Container

**Issue:** CUDA not accessible in container

**Solution:** Run with `--gpus all`:
```bash
docker run --gpus all -it cuda-optimizer
```

Add environment variables:
```bash
docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility ...
```

### Slurm / HPC Clusters

**Issue:** Multiple jobs conflict on same GPU

**Solution:** Use CUDA_VISIBLE_DEVICES:
```bash
# In job script
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
```

**Issue:** NCCL fails in multi-node

**Solution:** Set network interface:
```bash
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand
# or
export NCCL_SOCKET_IFNAME=eth0  # Ethernet
```

---

## Known Limitations

1. **Custom CUDA extensions**: Not automatically optimized. Manually apply `CustomOps` or report issue.
2. **In-place operations**: May conflict with checkpointing. Disable for affected layers.
3. **Weight tying**: Shared parameters across modules may need manual handling. See checkpointing docs.
4. **Dynamic control flow**: Checkpointing assumes static graph. Models with dynamic shapes may not work.
5. **CPU offloading**: Not supported. All parameters must fit in GPU memory (with optimizations).

---

## Getting Additional Help

1. **Check documentation**: Each module has detailed docs in `docs/`
2. **Run validation tests**:
```bash
pytest tests/unit/ -v  # unit tests
pytest tests/integration/ -v  # full pipeline
```

3. **Create minimal reproducible example**: Use [this template](https://github.com/.../minimal-repro)

4. **Open an issue**: Include:
   - CUDA version (`nvcc --version`)
   - PyTorch version (`torch.__version__`)
   - GPU model (`nvidia-smi`)
   - Full error traceback
   - Minimal reproduction script

5. **Community**: Join discussions on [GitHub Discussions](https://github.com/.../discussions)

---

## Performance Tuning Tips

1. **Profile first**: Always run `profile_model()` before optimizing to identify bottlenecks.
2. **One change at a time**: Test each optimization separately to measure impact.
3. **Monitor GPU utilization**: Use `nvidia-smi dmon` or dashboard to ensure high utilization (>80%).
4. **Tune batch size**: Larger batches generally improve throughput until memory limits.
5. **Use correct data type**: For Ampere+ GPUs (A100, RTX 30/40 series), `bfloat16` is faster than `float16`.
6. **Pinned memory**: Always use `pin_memory=True` in DataLoader for GPU training.
7. **Compile bottlenecks**: For critical paths, consider compiling with `torch.compile()`:
```python
model = torch.compile(model, mode='max-autotune')
```

---

## Quick Reference: Common Commands

```bash
# Check CUDA setup
nvcc --version && nvidia-smi && python -c "import torch; print(torch.version.cuda)"

# Run tests
pytest tests/ -v --tb=short

# Profile model
python scripts/run_baseline.py --model resnet50 --iters 100

# Build docs
cd docs && make html

# Clean cache
rm -rf ~/.cache/cuda-optimizer/

# Reinstall from source
pip install -e . --no-deps --force-reinstall
```

---

## Release-Specific Issues

### v0.1.x

- **Issue**: Python 3.8 compatibility
  - **Fix**: Upgrade to Python 3.9+ (PyTorch 2.0+ requires 3.9+)

- **Issue**: RTX 20-series (Turing) compute capability 7.5 support
  - **Fix**: When building custom ops, add `-gencode arch=compute_75,code=sm_75` to CUDA flags

- **Issue**: cuDNN 9.x not supported yet
  - **Fix**: Use cuDNN 8.x (tested with 8.9)
